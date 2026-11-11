from __future__ import annotations

from pathlib import Path

from ai_transcriber.transcriber.models import Transcript


class TxtExporter:

    def export(
        self,
        transcript: Transcript,
        filename: str,
    ) -> Path:

        output_path = Path(filename)

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            for segment in transcript.segments:

                start = self._format_time(
                    segment.start
                )

                speaker = (
                    segment.speaker
                    or "UNKNOWN"
                )

                f.write(
                    f"[{start}] {speaker}:\n"
                )

                f.write(
                    f"{segment.text}\n\n"
                )

        return output_path

    def _format_time(
        self,
        seconds: float,
    ) -> str:

        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

        return (
            f"{minutes:02d}:{seconds:02d}"
        )
