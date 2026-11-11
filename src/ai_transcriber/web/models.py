
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_transcriber.utils.redaction import redact_diagnostic


@dataclass
class JobOptions:
    transcribe: bool = True
    diarization: bool = False
    summary: bool = True
    send: bool = False
    normalize: bool = False

    model: str = "large-v3-turbo"
    backend: str = "auto"
    device: str = "auto"
    language: str | None = None
    compute_type: str = "auto"

    summary_prompt_type: str = "meeting"
    custom_prompt: str | None = None
    llm_endpoint: str = "http://127.0.0.1:1234/v1"
    llm_model: str | None = None

    unload_model: bool = True

    send_endpoint: str | None = None


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"
    progress: int = 0
    current_step: str = "Queued"
    logs: list[str] = field(default_factory=list)
    error: str | None = None

    transcript_file: str | None = None
    summary_file: str | None = None
    source_file: str | None = None
    output_dir: str | None = None
    transcript_text: str | None = None
    summary_text: str | None = None
    send_result: Any = None

    def log(self, message: str) -> None:
        self.logs.append(redact_diagnostic(message))
        # Не даём логам раздувать RAM: держим хвост.
        if len(self.logs) > 500:
            del self.logs[: len(self.logs) - 500]
