from __future__ import annotations

from rich.console import Console
from whisperx.diarize import DiarizationPipeline

from ai_transcriber.config import HF_TOKEN

console = Console()

class Diarizer:

    def __init__(
        self,
        device: str,
        token: str = HF_TOKEN,
    ) -> None:

        console.print("[green]Loading diarization model[/green]")
        console.print("[yellow]Parameters: device - " + device + "[/yellow]")

        self._model = DiarizationPipeline(
            # model_name="pyannote/speaker-diarization-3.1",
            device=device,
            token=token,
        )

    def diarize(
        self,
        audio,
    ):
        return self._model(audio)
