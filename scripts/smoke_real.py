"""Run real, offline ASR checks on local audio and an already downloaded model.

Writes private artifacts only to the requested local output directory. Never
sends a webhook or downloads a model. Long coverage uses speech at both ends
of an hour of silence; it checks timestamps, not hour-long speech accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import wave

from ai_transcriber.web.models import Job, JobOptions
from ai_transcriber.web.processor import process_job


def write_fixture(source: Path, target: Path, long: bool = False) -> None:
    with wave.open(str(source), 'rb') as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Use a 16 kHz mono PCM16 WAV source')
        stream.setpos(min(60 * 16000, max(0, stream.getnframes() - 30 * 16000)))
        clip = stream.readframes(30 * 16000)
    with wave.open(str(target), 'wb') as output:
        output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        output.writeframes(clip)
        if long:
            # Keep allocation bounded: append silence a second at a time.
            silence = b'\0' * 32000
            for _ in range(3660 - len(clip) // 32000):
                output.writeframes(silence)
            output.writeframes(clip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/real-smoke'))
    parser.add_argument('--include-long', action='store_true')
    args = parser.parse_args()
    if not (args.model_path / 'model.bin').is_file():
        parser.error('--model-path must contain an already downloaded model.bin')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in (['short', 'long'] if args.include_long else ['short']):
        audio = args.output / f'{name}.wav'
        write_fixture(args.audio, audio, long=name == 'long')
        job = Job(id=name, filename=audio.name)
        options = JobOptions(model=str(args.model_path.resolve()), backend='faster-whisper',
                             device='cpu', compute_type='int8', summary=False, send=False)
        start = time.monotonic()
        process_job(job, audio, args.output / name, options)
        payload = json.loads((args.output / name / f'{name}.json').read_text())
        segments = payload['segments']
        if not segments or not any(s['text'].strip() for s in segments):
            raise RuntimeError(f'{name}: no speech recognized')
        if name == 'long' and not any(s['start'] >= 3600 for s in segments):
            raise RuntimeError('Long-file timestamp restoration failed')
        row = {'case': name, 'seconds': round(time.monotonic() - start, 2),
               'segments': len(segments), 'duration': payload['duration'],
               'last_segment_start': segments[-1]['start'], 'passed': True}
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
