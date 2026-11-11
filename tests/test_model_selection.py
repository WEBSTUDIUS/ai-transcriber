"""Policy boundaries and integration with local model startup; no model downloads."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_transcriber.llm.model_selection import GIB, Memory, select_model, configured_selection


@pytest.mark.parametrize('total,available,expected,fits', [
    (4, 3, '4b', False),
    (8, 6, '4b', True),
    (16, 12, '8b', True),
    (32, 24, '14b', True),
    (64, 32, '14b', True),
    (32, 5, '4b', True),
    (32, 2, '4b', False),
    (16, 7.49, '4b', True),
    (16, 7.5, '8b', True),
])
def test_memory_tiers_and_pressure(total, available, expected, fits):
    result = select_model(Memory(int(total * GIB), int(available * GIB)))
    assert result.profile.id == expected
    assert result.fits is fits
    if fits:
        assert result.profile.required_gib * GIB <= result.budget


def test_unknown_memory_does_not_guess():
    assert not select_model(None).fits


def test_context_and_precision_reduce_headroom():
    memory = Memory(16 * GIB, 8 * GIB)
    assert select_model(memory).profile.id == '8b'
    assert select_model(memory, context=32768).profile.id == '4b'
    assert not select_model(memory, context=65536, cache_type='f16').fits


def test_explicit_profile_does_not_silently_downgrade():
    result = select_model(Memory(8 * GIB, 6 * GIB), mode='14b')
    assert result.profile.id == '14b'
    assert not result.fits


def test_custom_model_validation_and_legacy_config(monkeypatch):
    monkeypatch.delenv('LLAMA_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LLAMA_MODEL_REPO', 'owner/custom')
    monkeypatch.setenv('LLAMA_MODEL_FILE', 'custom.gguf')
    assert configured_selection().mode == 'manual'
    monkeypatch.setenv('LLAMA_MODEL_PROFILE', 'auto')
    assert configured_selection().profile.id != 'manual'
    with pytest.raises(ValueError, match='both'):
        select_model(None, mode='manual', repo='owner/model')
    with pytest.raises(ValueError, match='filename'):
        select_model(None, mode='manual', repo='owner/model', filename='../outside.gguf')


def test_model_download_uses_selected_profile(web, monkeypatch, tmp_path):
    from ai_transcriber.llm import local_server
    selection = select_model(Memory(16 * GIB, 12 * GIB))
    monkeypatch.setattr(local_server, 'configured_selection', lambda **kw: selection)
    server = local_server.LocalLLMServer()
    server.cache_dir = tmp_path
    download = Mock()
    monkeypatch.setattr(server, '_download', download)
    server.ensure_model()
    url, path, label = download.call_args.args
    assert 'Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf' in url
    assert path == tmp_path / 'models' / selection.profile.filename
    assert label == server.model_name == selection.profile.filename


def test_memory_rechecked_before_download_and_process_start(web, monkeypatch, tmp_path):
    from ai_transcriber.llm import local_server
    selection = select_model(Memory(16 * GIB, 12 * GIB))
    monkeypatch.setattr(local_server, 'configured_selection', lambda **kw: selection)
    monkeypatch.setattr(local_server, 'detect_memory', lambda: Memory(16 * GIB, 2 * GIB))
    server = local_server.LocalLLMServer()
    prepare = Mock()
    monkeypatch.setattr(server, 'ensure_binary', prepare)
    with pytest.raises(RuntimeError, match='RAM'):
        server.prepare()
    prepare.assert_not_called()
    server.binary_path = tmp_path / 'llama-server'
    server.model_path = tmp_path / 'model.gguf'
    start = Mock()
    monkeypatch.setattr(local_server.subprocess, 'Popen', start)
    with pytest.raises(RuntimeError, match='RAM'):
        server.start()
    start.assert_not_called()


def test_local_model_api_exposes_live_budget(web, monkeypatch):
    module, client = web
    status = select_model(Memory(16 * GIB, 12 * GIB)).as_dict()
    monkeypatch.setattr(module, 'local_model_status', lambda: status)
    response = client.get('/api/local-model')
    assert response.status_code == 200
    assert response.json()['profile'] == '8b'
    assert response.json()['available_gib'] == 12
    assert 'Автовыбор' in client.get('/').text or 'localModelPlan' in client.get('/').text


def test_cuda_vram_is_not_confused_with_host_ram(web, monkeypatch):
    import torch
    from ai_transcriber.llm.local_server import _gpu_layers_for_memory
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (2 * GIB, 4 * GIB), raising=False)
    assert _gpu_layers_for_memory(99, 7) == 0
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (12 * GIB, 16 * GIB))
    assert _gpu_layers_for_memory(99, 7) == 99


def test_loaded_model_status_does_not_suggest_a_different_model(web, monkeypatch):
    from types import SimpleNamespace
    from ai_transcriber.llm import local_server
    selected = select_model(Memory(32 * GIB, 24 * GIB))
    monkeypatch.setattr(local_server, '_shared_server', SimpleNamespace(selection=selected))
    monkeypatch.setattr(local_server, 'detect_memory', lambda: Memory(32 * GIB, 2 * GIB))
    result = local_server.local_model_status()
    assert result['profile'] == '14b'
    assert result['loaded'] is True
    assert result['fits'] is True


def test_cpu_fallback_passes_zero_gpu_layers_explicitly(web, monkeypatch, tmp_path):
    from ai_transcriber.llm import local_server
    selected = select_model(Memory(8 * GIB, 6 * GIB))
    monkeypatch.setattr(local_server, 'configured_selection', lambda **kw: selected)
    monkeypatch.setattr(local_server, 'detect_memory', lambda: selected.memory)
    monkeypatch.setattr(local_server, '_gpu_layers_for_memory', lambda *args: 0)
    server = local_server.LocalLLMServer(ngpu=99)
    server.binary_path = tmp_path / 'llama-server'
    server.model_path = tmp_path / 'model.gguf'
    process = Mock()
    monkeypatch.setattr(local_server.subprocess, 'Popen', process)
    monkeypatch.setattr(local_server.atexit, 'register', Mock())
    monkeypatch.setattr(server, '_write_pid_file', Mock())
    server.start()
    command = process.call_args.args[0]
    assert command[command.index('--n-gpu-layers') + 1] == '0'


def test_failed_shared_startup_cleans_up_child(web, monkeypatch):
    from ai_transcriber.llm import local_server
    server = Mock()
    server.prepare_and_start.side_effect = RuntimeError('startup failed')
    monkeypatch.setattr(local_server, '_shared_server', None)
    monkeypatch.setattr(local_server, 'LocalLLMServer', Mock(return_value=server))
    with pytest.raises(RuntimeError, match='startup failed'):
        local_server.get_shared_server()
    server.stop.assert_called_once()
    assert local_server._shared_server is None
