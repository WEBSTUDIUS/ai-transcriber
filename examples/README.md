# Manual examples

Install the project first (`uv sync`), then supply your own transcript:

```bash
uv run python examples/summarize.py /path/to/transcript.txt --prompt call
uv run python examples/summarize.py /path/to/transcript.txt --prompt meeting --llm model-id
```

These examples share the CLI's LLM selection, fallback and error redaction.
They never send a webhook. Generated summaries are written next to your input.
The automated tests are under `tests/`.

The Windows-only codec diagnostic is `scripts/check_codec.py`.
