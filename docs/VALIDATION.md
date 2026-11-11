# Validation

## Automated checks

Run `uv sync --extra test` and `uv run --no-sync python -m pytest -q`.
The suite covers the queue, uploads, persistence, exporters, secret redaction,
model selection and LLM failure handling. ML engines are mocked in unit tests.

GitHub Actions runs the lightweight suite and a separate locked installation
on Linux, macOS and Windows. Installation jobs import the actual ML stacks, including WhisperX alignment, and
exercise the CLI. Windows additionally verifies the CUDA 12.8 wheel, runs the
CPU setup script and checks that CPU torch/torchaudio/torchvision import.
A hosted runner without an NVIDIA GPU cannot verify actual CUDA inference.

## Real offline ASR

Supply your own normalized 16 kHz mono PCM16 WAV and a downloaded
faster-whisper model directory containing `model.bin`:

```bash
uv run --no-sync python scripts/smoke_real.py \
  --audio /path/to/local.wav \
  --model-path /path/to/downloaded/faster-whisper-small \
  --include-long
```

The check writes private fixtures and outputs to ignored `data/real-smoke/`.
It runs real CPU/int8 inference without downloading models or sending webhooks.
It requires at least one nonempty segment in each output and speech timestamps
past 3600 seconds in the long output. The long fixture repeats 30 seconds of
speech at the beginning and after 61 minutes, with silence between; this checks
VAD timestamp restoration and long media handling, not recognition quality on
an hour of continuous speech. Use a real long recording for that assessment.

## Local results (2026-11-11)

- 74 unit/regression tests passed.
- Real faster-whisper `small`, CPU/int8: 30-second clip, 12 segments, 11.14 s.
- Long fixture: duration 3690 s, 16 segments, last segment starts at 3683.29 s;
  inference completed in 23.34 s.
- Real CPU diarization: 7 segments, one speaker, 60.94 s including initial setup.
- Full WhisperX + Russian word alignment + diarization passed offline after
  caching: 7 segments, all 7 with speaker labels, 48.43 s;
  the process exited normally.
- Unreachable LLM: fallback checked available RAM and refused startup rather than
  overcommitting memory; shared-server resources were released.
- Clean macOS ARM/Python 3.13 installation from `uv.lock` passed imports and tests.
- Wheel and source archive contained no `.env`, runtime/data directories or key
  files. The wheel included templates/static assets and served the UI correctly
  when installed and launched outside the source checkout.

Private source recordings, transcripts, model weights and detailed local logs
are intentionally excluded from version control. These checks do not establish
speaker-count accuracy on labelled multi-speaker material or CUDA inference
correctness on an NVIDIA machine.

## Dependency regression caught by the real run

The initial full WhisperX run failed in NLTK 3.10.1 when importing alignment
from an in-project virtual environment. The dependency floor is now NLTK
3.10.4, and the lock update changes only NLTK. This uses the upstream fix; no
import-security checks are disabled in the application. See
[NLTK's upstream correction](https://github.com/nltk/nltk/pull/3732).

The Windows CPU installation check caught an additional packaging issue: the
constraint `torch==2.8.0` also accepts an installed `2.8.0+cu128` wheel. The CPU
script now requests the explicit `+cpu` variants and ignores the project's
CUDA package sources when installing them.

Transformers' optional remote safetensors conversion is disabled by default
(`DISABLE_SAFETENSORS_CONVERSION=true`): its background thread held the first
full smoke process open after inference had completed. Loading existing
safetensors files and downloading model weights remain supported.
