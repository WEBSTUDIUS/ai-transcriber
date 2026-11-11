from __future__ import annotations

from pathlib import Path

from ai_transcriber.transcriber.models import Transcript
from ai_transcriber.utils.time import format_time


def transcript_to_text(transcript: Transcript) -> str:
    """Canonical TXT format shared by CLI, web summaries and legacy writers."""
    lines = []
    for segment in transcript.segments:
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        lines.append(
            f"[{format_time(segment.start, include_ms=True)} - "
            f"{format_time(segment.end, include_ms=True)}] {speaker}{segment.text}"
        )
    return "\n".join(lines)


class TxtExporter:
    def export(self, transcript: Transcript, filename: str) -> Path:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = transcript_to_text(transcript)
        output_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        return output_path

    def _format_time(self, seconds: float) -> str:
        return format_time(seconds, include_ms=True)
