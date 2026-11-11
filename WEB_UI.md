
# AI Transcriber Web UI

The Web UI is integrated into `ai_transcriber.web`.

## Install

Install the project, including its web dependencies:

```bash
uv sync --extra test
```

## Start

```bash
ai-transcriber web
```

Default:

```text
http://127.0.0.1:8000
```

For LAN access:

```bash
ai-transcriber web --host 0.0.0.0 --port 8000
```

## Pipeline

The UI intentionally exposes only the meaningful high-level actions:

- Transcribe
- Generate summary
- Send result

Alignment is always part of WhisperX transcription.

Speaker diarization is an optional sub-option of transcription.

### Audio

```text
Audio
  -> WhisperX ASR
  -> Alignment
  -> optional Diarization
  -> Transcript
  -> optional Summary
  -> optional Send
```

### Existing TXT

```text
TXT
  -> optional Summary
  -> optional Send
```

## Send endpoint

The send action posts to a Bitrix webhook (see `exporters/webhook_exporter.py`):
it creates a smart-process item with the summary and attaches the transcript file
as a timeline comment.

The webhook URL is either picked from the predefined list or entered manually
in the UI.

Predefined webhooks are configured through environment variables:

```bash
WEBHOOK_IT_TASK=https://example.bitrix24.ru/rest/1/TOKEN/
WEBHOOK_HR=https://example.bitrix24.ru/rest/2/TOKEN/
WEBHOOK_CALL_TRACK=https://example.bitrix24.ru/rest/3/TOKEN/
```

Sending requires the summary step to be enabled (the summary file is attached
to the created item).

## Important

Jobs are serialized and the active queue is bounded to 20. Preset URLs remain server-side. See the README for security, cleanup and portal field mapping configuration.
