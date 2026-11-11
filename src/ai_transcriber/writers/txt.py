"""Compatibility entry point for the shared TXT exporter."""
from pathlib import Path

from ai_transcriber.exporters.txt_exporter import TxtExporter
from ai_transcriber.transcriber.models import Transcript


def write_txt(transcript: Transcript, output_file: str) -> Path:
    return TxtExporter().export(transcript, output_file)
