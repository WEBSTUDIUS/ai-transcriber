"""Summarize a user-supplied transcript using the project's normal CLI pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from ai_transcriber.cli import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transcript', type=Path)
    parser.add_argument('--prompt', choices=('call', 'meeting'), default='meeting')
    parser.add_argument('--llm', help='Model ID; omit to select interactively')
    args = parser.parse_args()
    if not args.transcript.is_file():
        parser.error('Transcript file does not exist')
    summarize(file=args.transcript, prompt=args.prompt, send=None, llm=args.llm, unload_model=True)


if __name__ == '__main__':
    main()
