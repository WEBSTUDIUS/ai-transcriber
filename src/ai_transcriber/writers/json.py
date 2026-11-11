import json
from pathlib import Path

from ai_transcriber.transcriber.models import Transcript
from ai_transcriber.utils.time import format_time


def write_json(transcript: Transcript, output_file: str) -> Path:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "start_time": format_time(segment.start, include_ms=True),
                "end_time": format_time(segment.end, include_ms=True),
                "text": segment.text,
            }
            for segment in transcript.segments
        ],
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
