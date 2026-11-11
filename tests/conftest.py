import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_TRANSCRIBER_DATA_DIR", str(tmp_path))
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_asr = ModuleType("ai_transcriber.transcriber.service")
    fake_asr.Transcriber = type("Transcriber", (), {})
    fake_asr.clear_model_cache = lambda: None
    fake_wx = ModuleType("ai_transcriber.transcriber.whisperx_service")
    fake_wx.WhisperXTranscriber = type("WhisperXTranscriber", (), {})
    fake_wx.clear_whisperx_caches = lambda: None
    with patch.dict(sys.modules, {"torch": torch,
        "ai_transcriber.transcriber.service": fake_asr,
        "ai_transcriber.transcriber.whisperx_service": fake_wx}):
        for name in ("ai_transcriber.config", "ai_transcriber.utils.runtime",
                     "ai_transcriber.web.processor", "ai_transcriber.web.app"):
            sys.modules.pop(name, None)
        module = importlib.import_module("ai_transcriber.web.app")
        yield module, TestClient(module.app)
