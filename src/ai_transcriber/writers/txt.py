from pathlib import Path

from ai_transcriber.transcriber.models import Transcript
from ai_transcriber.utils.time import format_time


def write_txt(
    transcript: Transcript,
    output_file: str,
) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for segment in transcript.segments:
        lines.append(
            f"[{format_time(segment.start, include_ms=True)} - "
            f"{format_time(segment.end, include_ms=True)}]"
        )
        lines.append(segment.text)
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
