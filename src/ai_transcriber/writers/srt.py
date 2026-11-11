from pathlib import Path

from ai_transcriber.transcriber.models import Transcript


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds_part = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{ms:03}"


def write_srt(transcript: Transcript, output_file: str) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for index, segment in enumerate(transcript.segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}"
        )
        lines.append(segment.text)
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
