"""Web/queue regression tests. ML engines are mocked; no model downloads."""
import importlib
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient




def test_page_and_config_validation(web):
    module, client = web
    assert client.get("/").status_code == 200
    assert client.post("/api/config", json={"model": {"bad": 1}}).status_code == 400
    assert client.post("/api/config", json={"send": "yes"}).status_code == 400
    response = client.post("/api/config", json={"language": "ru", "unknown": "ignored"})
    assert response.json()["language"] == "ru"
    assert "unknown" not in response.json()


def test_invalid_uploads_create_no_jobs(web):
    module, client = web
    for filename, content, data in [
        ("bad.exe", b"x", {}),
        ("empty.wav", b"", {}),
        ("note.txt", b"hello", {"summary": "false", "send": "true"}),
        ("note.txt", b"hello", {"llm_endpoint": "file:///etc/passwd"}),
        ("note.txt", b"hello", {"summary_prompt_type": "custom", "custom_prompt": ""}),
    ]:
        assert client.post("/api/jobs", files={"file": (filename, content)}, data=data).status_code == 400
    assert not module.jobs.active_ids()
    assert not list(module.UPLOAD_DIR.iterdir())


def test_cleanup_protects_all_active_jobs(web):
    module, _ = web
    active = [module.jobs.create(f"{i}.wav") for i in range(9)]
    for job in active:
        (module.UPLOAD_DIR / f"{job.id}.wav").write_bytes(b"audio")
        (module.OUTPUT_DIR / job.id).mkdir()
    module._cleanup_old_outputs(active[-1].id, keep_last=1)
    assert len(list(module.UPLOAD_DIR.iterdir())) == 9
    assert len(list(module.OUTPUT_DIR.iterdir())) == 9


def test_queue_status_and_failure():
    from ai_transcriber.web.jobs import JobManager
    manager = JobManager()
    first, second = manager.create("a.wav"), manager.create("b.wav")
    started, release = threading.Event(), threading.Event()
    def block():
        started.set()
        release.wait(3)
    def fail():
        raise ValueError("expected failure")
    manager.run(first.id, block)
    assert started.wait(2)
    manager.run(second.id, fail)
    try:
        assert second.status == "queued"
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while second.status != "failed" and time.monotonic() < deadline:
        time.sleep(.01)
    assert first.status == "completed"
    assert second.status == "failed"
    assert second.error == "expected failure"


def test_eviction_skips_active_oldest():
    from ai_transcriber.web.jobs import JobManager
    manager = JobManager()
    manager.MAX_JOBS = 3
    active = manager.create("active.wav")
    for i in range(8):
        job = manager.create(f"{i}.wav")
        job.status = "completed"
    assert manager.get(active.id) is active
    assert len(manager._jobs) <= 3


def test_text_processing_and_download(web, monkeypatch):
    module, client = web
    processor = importlib.import_module("ai_transcriber.web.processor")
    monkeypatch.setattr(processor, "_summarize_with_llm", lambda *args: "Test summary")
    response = client.post("/api/jobs", files={"file": ("../meeting.txt", b"hello world")})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(.01)
    assert job["status"] == "completed", job
    assert "transcript_text" not in job
    assert client.get(f"/api/jobs/{job_id}/file").text == "hello world"
    assert "Test summary" in client.get(f"/api/jobs/{job_id}/file?kind=summary").text
    assert client.get(f"/api/jobs/{job_id}/file?kind=bad").status_code == 400


def test_compute_and_mps_disabled_threshold(web, monkeypatch):
    runtime = importlib.import_module("ai_transcriber.utils.runtime")
    assert runtime.normalize_compute_type("float16", "int8", "cpu") == "float32"
    monkeypatch.setattr(runtime, "detect_runtime", lambda: ("cpu", "mps", "int8"))
    monkeypatch.setattr(runtime, "MPS_LONG_FILE_THRESHOLD_SECONDS", 0)
    assert runtime.resolve_runtime("auto", 9999)[1] == "mps"
    monkeypatch.setattr(runtime, "MPS_LONG_FILE_THRESHOLD_SECONDS", 600)
    assert runtime.resolve_runtime("auto", 9999)[1] == "cpu"


def test_bitrix_mapping_and_preflight(web, tmp_path, monkeypatch):
    exporter = importlib.import_module("ai_transcriber.exporters.webhook_exporter")
    monkeypatch.setattr(exporter, "ENTITY_TYPE_ID", 42)
    monkeypatch.setenv("BITRIX_FIELDS_JSON", '{"MY_SUMMARY":"summary","MY_USERS":"assignees","MY_VALUE":[1,2]}')
    monkeypatch.setenv("BITRIX_ASSIGNEE_IDS", '[7,8]')
    summary, transcript = tmp_path / "summary.md", tmp_path / "transcript.txt"
    summary.write_text("# Summary\nDone")
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        payload = {"result": {"item": {"id": 123}}} if len(calls) == 1 else {"result": 456}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload, status_code=200)
    monkeypatch.setattr(exporter.requests, "post", post)
    with pytest.raises(FileNotFoundError):
        exporter.create_smart_process_item("https://portal.example/rest/test", summary_file=str(summary), file=str(transcript))
    assert not calls
    transcript.write_text("hello")
    result = exporter.create_smart_process_item("https://portal.example/rest/test", summary_file=str(summary), file=str(transcript))
    assert result["item_id"] == 123
    assert calls[0][0].endswith("/test/crm.item.add.json")
    fields = calls[0][1]["json"]["fields"]
    assert "Done" in fields["MY_SUMMARY"]
    assert fields["MY_USERS"] == [7, 8]
    assert fields["MY_VALUE"] == [1, 2]


def test_archive_rejects_sibling_escape(web, tmp_path):
    import zipfile
    from ai_transcriber.llm.local_server import LocalLLMServer
    archive = tmp_path / "bad.zip"
    destination = tmp_path / "target"
    destination.mkdir()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../target-sibling/escape.txt", "bad")
    with pytest.raises(RuntimeError, match="Unsafe path"):
        LocalLLMServer._extract(archive, destination)
    assert not (tmp_path / "target-sibling").exists()


def test_text_chunking_rejects_zero_and_preserves_content(web):
    from ai_transcriber.llm.client import split_transcript
    with pytest.raises(ValueError):
        split_transcript("test", 0)
    chunks = split_transcript("a" * 105, 20)
    assert max(map(len, chunks)) <= 20
    assert "".join(chunks) == "a" * 105


def test_release_discovery_skips_metadata_only_releases(web, monkeypatch):
    from ai_transcriber.llm import local_server
    releases = [
        {"tag_name": "metadata-only", "assets": [{"name": "nightly-tag.txt"}]},
        {"tag_name": "b11985", "assets": [{"name": "llama-b11985-bin-macos-arm64.tar.gz", "browser_download_url": "https://example.test/archive"}]},
    ]
    monkeypatch.setattr(local_server.requests, "get", lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: releases))
    assert local_server._resolve_asset("macos-arm64", ".tar.gz") == ("b11985", "https://example.test/archive")


def test_empty_summary_is_rejected(web):
    from ai_transcriber.llm.client import LMStudioClient
    client = LMStudioClient(model="test")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=""))]))))
    with pytest.raises(ValueError, match="empty summary"):
        client.summarize("test", "meeting")


def test_queue_is_bounded_and_recovers_after_completion():
    from ai_transcriber.web.jobs import JobManager
    manager = JobManager()
    manager.MAX_JOBS = 2
    first = manager.create('first.wav')
    manager.create('second.wav')
    with pytest.raises(RuntimeError, match='queue is full'):
        manager.create('third.wav')
    first.status = 'completed'
    assert manager.create('third.wav').status == 'queued'


def test_web_rejects_cross_site_requests(web):
    _, client = web
    assert client.get('/api/config', headers={'Origin': 'https://foreign.example'}).status_code == 403
    assert client.post('/api/config', json={}, headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    assert client.post('/api/config', json={}, headers={'Origin': 'http://testserver'}).status_code == 200


def test_webhook_presets_do_not_expose_tokens(web, monkeypatch):
    module, client = web
    secret_url = 'https://portal.example/rest/1/' + 'a' * 16 + '/'
    monkeypatch.setenv('WEBHOOK_IT_TASK', secret_url)
    assert module.available_send_endpoints()[0] == {'name': 'it_task', 'url': 'it_task'}
    module.save_config({**module.DEFAULT_CONFIG, 'send_endpoint': secret_url})
    assert client.get('/api/config').json()['send_endpoint'] == 'it_task'
    assert secret_url not in client.get('/').text


def test_config_save_failure_does_not_strand_job(web, monkeypatch):
    module, client = web
    processor = importlib.import_module('ai_transcriber.web.processor')
    monkeypatch.setattr(processor, '_summarize_with_llm', lambda *a: 'Done')
    def fail(*a):
        raise OSError('disk error')
    monkeypatch.setattr(module, 'save_config', fail)
    response = client.post('/api/jobs', files={'file': ('note.txt', b'hello')})
    assert response.status_code == 200
    job_id = response.json()['job_id']
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] in ('completed', 'failed'):
            break
        time.sleep(.01)
    assert job['status'] == 'completed', job


def test_full_queue_rejects_upload_without_files(web):
    module, client = web
    module.jobs.MAX_JOBS = 1
    module.jobs.create('pending.wav')
    response = client.post('/api/jobs', files={'file': ('note.txt', b'hello')})
    assert response.status_code == 429
    assert not list(module.UPLOAD_DIR.iterdir())


def test_custom_prompt_preserves_literal_json_braces(web):
    from ai_transcriber.llm.client import LMStudioClient
    client = LMStudioClient(model='test')
    captured = {}
    def create(**kw):
        captured.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='Done'))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.summarize('transcript', 'custom', 'Return {"title": "..."} for {text}')
    assert captured['messages'][1]['content'] == 'Return {"title": "..."} for transcript'


def test_llm_model_discovery_uses_auth(web, monkeypatch):
    from ai_transcriber.llm.client import LMStudioClient
    client = LMStudioClient(model='test', api_key='unit-test-placeholder')
    captured = {}
    class FakeHTTP:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kw):
            captured.update(kw)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'data': [{'id': 'model'}]})
    monkeypatch.setattr('ai_transcriber.llm.client.httpx.Client', FakeHTTP)
    assert client.list_models() == ['model']
    assert captured['headers']['Authorization'] == 'Bearer unit-test-placeholder'
