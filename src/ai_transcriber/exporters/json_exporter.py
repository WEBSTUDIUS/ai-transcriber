from __future__ import annotations

import json
from dataclasses import asdict

from ai_transcriber.transcriber.models import Transcript


class JsonExporter:

    def export(
        self,
        transcript: Transcript,
        filename: str,
    ) -> None:

        data = asdict(transcript)

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
