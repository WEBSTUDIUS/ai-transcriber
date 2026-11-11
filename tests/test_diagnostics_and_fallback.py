"""User-visible errors and LLM fallback must not leak secrets or hide config errors."""
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, BadRequestError
from rich.console import Console
from typer.testing import CliRunner

from ai_transcriber.utils.redaction import redact_diagnostic
from ai_transcriber.llm.errors import endpoint_unavailable


def test_redaction_handles_environment_tokens_urls_and_relative_paths(monkeypatch):
    secret = 'unit-secret-' + '123456789'
    monkeypatch.setenv('HF_TOKEN', secret)
    message = f'{secret} https://user:pass@portal.example/rest/1/private-token/ /rest/1/other-token/ Bearer bearer-token'
    cleaned = redact_diagnostic(message)
    for value in (secret, 'private-token', 'other-token', 'bearer-token', 'user:pass'):
        assert value not in cleaned
    assert 'redacted' in cleaned


@pytest.mark.parametrize('status,expected', [(400, False), (401, False), (403, False),
    (404, False), (429, False), (500, False), (502, True), (503, True), (504, True)])
def test_fallback_http_status_policy(status, expected):
    response = httpx.Response(status, request=httpx.Request('GET', 'http://localhost/models'))
    assert endpoint_unavailable(httpx.HTTPStatusError('error', request=response.request, response=response)) is expected


def test_fallback_only_for_network_failures():
    assert endpoint_unavailable(httpx.ConnectError('offline'))
    assert endpoint_unavailable(APIConnectionError(request=httpx.Request('GET', 'http://localhost')))
    assert not endpoint_unavailable(ValueError('empty summary'))
    assert not endpoint_unavailable(RuntimeError('invalid model'))


def test_cli_errors_hide_secrets(web, monkeypatch, tmp_path):
    import ai_transcriber.cli as cli
    secret = 'unit-secret-' + '987654321'
    monkeypatch.setenv('OPENAI_API_KEY', secret)
    def fail(*a, **kw):
        raise RuntimeError(f'Failed https://portal.example/rest/1/private-token/ {secret}')
    monkeypatch.setattr(cli, '_run_summarize', fail)
    transcript = tmp_path / 'test.txt'
    transcript.write_text('hello')
    result = CliRunner().invoke(cli.app, ['summarize', str(transcript)])
    assert result.exit_code == 1
    assert 'private-token' not in result.output
    assert secret not in result.output
    assert 'redacted' in result.output


@pytest.mark.parametrize('failure', ['auth', 'bad_request', 'empty', 'no_models'])
def test_cli_does_not_start_local_model_for_configuration_errors(web, monkeypatch, failure):
    import ai_transcriber.cli as cli
    request = httpx.Request('GET', 'http://localhost/models')
    response = httpx.Response(401 if failure == 'auth' else 400, request=request)
    client = Mock()
    if failure == 'auth':
        client.list_models.side_effect = AuthenticationError('denied', response=response, body=None)
    elif failure == 'bad_request':
        client.list_models.side_effect = BadRequestError('bad request', response=response, body=None)
    elif failure == 'empty':
        client.list_models.side_effect = ValueError('bad JSON')
    else:
        client.list_models.return_value = []
    factory = Mock(return_value=client)
    server = Mock()
    monkeypatch.setattr(cli, 'LMStudioClient', factory)
    monkeypatch.setattr(cli, 'LocalLLMServer', server)
    with pytest.raises(Exception):
        cli.resolve_llm(Console(file=StringIO()), pinned='explicit-model')
    server.assert_not_called()


def test_cli_network_failure_uses_local_server_even_with_pinned_model(web, monkeypatch):
    import ai_transcriber.cli as cli
    primary, fallback = Mock(), Mock()
    primary.list_models.side_effect = httpx.ConnectError('offline')
    server = Mock(base_url='http://127.0.0.1:9999/v1', model_name='local-model', ctx_size=8192)
    monkeypatch.setattr(cli, 'LMStudioClient', Mock(side_effect=[primary, fallback]))
    monkeypatch.setattr(cli, 'LocalLLMServer', Mock(return_value=server))
    resolved, cleanup = cli.resolve_llm(Console(file=StringIO()), pinned='explicit-model')
    assert resolved is fallback
    server.prepare_and_start.assert_called_once()
    cleanup()
    server.stop.assert_called_once()


@pytest.mark.parametrize('failure,uses_fallback', [(ValueError('empty summary'), False),
    (httpx.ConnectError('offline'), True)])
def test_web_fallback_policy(web, monkeypatch, failure, uses_fallback):
    from ai_transcriber.web import processor
    from ai_transcriber.web.models import Job, JobOptions
    primary, fallback = Mock(), Mock()
    primary.summarize_long.side_effect = failure
    fallback.summarize_long.return_value = 'local summary'
    factory = Mock(side_effect=[primary, fallback])
    server = Mock(base_url='http://127.0.0.1:9999/v1', model_name='local-model', ctx_size=8192)
    local = Mock(return_value=server)
    monkeypatch.setattr(processor, 'LMStudioClient', factory)
    monkeypatch.setattr(processor, 'get_shared_server', local)
    if uses_fallback:
        assert processor._summarize_with_llm(Job('id', 'file'), JobOptions(), 'text') == 'local summary'
        local.assert_called_once()
    else:
        with pytest.raises(ValueError, match='empty summary'):
            processor._summarize_with_llm(Job('id', 'file'), JobOptions(), 'text')
        local.assert_not_called()


def test_transcriber_initialization_failure_is_redacted_and_cleaned(web, monkeypatch, tmp_path):
    import ai_transcriber.cli as cli
    from ai_transcriber.utils import runtime
    cleaned = Mock()
    monkeypatch.setattr(runtime, 'clear_asr_models', cleaned)
    monkeypatch.setattr(cli, 'get_audio_duration', lambda *a: 1)
    def fail(*a, **kw):
        raise RuntimeError('Download failed https://example.test/rest/1/private-credential/')
    monkeypatch.setattr(cli, 'get_transcriber_service', fail)
    audio = tmp_path / 'test.wav'
    audio.write_bytes(b'placeholder')
    result = CliRunner().invoke(cli.app, ['transcribe', str(audio)])
    assert result.exit_code == 1
    assert 'private-credential' not in result.output
    assert 'redacted' in result.output
    cleaned.assert_called_once()


def test_job_logs_redact_credentials_at_entry(monkeypatch):
    from ai_transcriber.web.models import Job
    secret = 'test-secret-for-job-log'
    monkeypatch.setenv('OPENAI_API_KEY', secret)
    job = Job('id', 'file')
    job.log(f'failed {secret} /rest/1/private-token/')
    assert secret not in job.logs[0]
    assert 'private-token' not in job.logs[0]
