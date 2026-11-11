from __future__ import annotations

import os
from pathlib import Path
import time
import typer
import tempfile
from typing import Any, Callable

from ai_transcriber.exporters.json_exporter import JsonExporter
from ai_transcriber.exporters.txt_exporter import TxtExporter
from ai_transcriber.exporters.webhook_exporter import create_smart_process_item
from ai_transcriber.utils.runtime import cleanup_memory, normalize_compute_type, resolve_runtime

if os.name == "nt":
    ffmpeg_path = os.getenv("FFMPEG_DLL_PATH", r"C:\ffmpeg7\bin")
    if Path(ffmpeg_path).exists():
        os.add_dll_directory(ffmpeg_path)

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ai_transcriber.config import DEFAULT_MODEL, DEFAULT_PROMPT, HF_TOKEN, DEFAULT_LMSTUDIO_BASE_URL
from ai_transcriber import __version__
from ai_transcriber.doctor.service import run_checks
from ai_transcriber.doctor.renderer import render
from ai_transcriber.llm.client import LMStudioClient
from ai_transcriber.llm.local_server import LocalLLMServer
from ai_transcriber.transcriber.models import AVAILABLE_MODELS, AVAILABLE_PROMPTS, Transcript, AVAILABLE_SEND_POINTS


from ai_transcriber.utils.time import format_time

from ai_transcriber.audio.processor import normalize_audio, get_audio_duration

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    AI Transcriber CLI.
    """
    if ctx.invoked_subcommand is None:
        console.print("[bold]AI Transcriber[/bold]")
        console.print("Use --help for available commands.")


def _noop_cleanup() -> None:
    return None


def choose_llm_interactively(console: Console, models: list[str]) -> str:
    """
    Show the available LLM models and let the user pick one from the keyboard.
    Enter (or an invalid value) falls back to the first model.
    """
    if len(models) == 1:
        console.print(f"[green]Using LLM model:[/green] {models[0]}")
        return models[0]

    table = Table(title="Available LLM models")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Model")
    for i, model in enumerate(models, start=1):
        table.add_row(str(i), model)
    console.print(table)

    answer = Prompt.ask(
        "[bold]Select LLM model number[/bold] (Enter = first)",
        default="1",
        console=console,
    )

    try:
        index = int(answer.strip())
    except ValueError:
        console.print("[yellow]Invalid selection, using the first model.[/yellow]")
        return models[0]

    if index < 1 or index > len(models):
        console.print("[yellow]Selection out of range, using the first model.[/yellow]")
        return models[0]

    return models[index - 1]


def resolve_llm(
    console: Console,
    pinned: str | None = None,
) -> tuple[LMStudioClient, Callable[[], None]]:
    """
    Return an (LLM client, cleanup) pair for summarization.

    Priority:
      1. `pinned` model id (no prompts).
      2. Interactive selection from the configured LLM endpoint (LM Studio).
      3. If the endpoint is unreachable: download llama.cpp + the configured GGUF model
         and start a local OpenAI-compatible server (configured context size).
    """
    client = LMStudioClient(base_url=DEFAULT_LMSTUDIO_BASE_URL)

    if pinned:
        client.model = pinned
        return client, _noop_cleanup

    try:
        models = client.list_models()
    except Exception:
        models = []

    if models:
        client.model = choose_llm_interactively(console, models)
        return client, _noop_cleanup

    console.print(
        "[yellow]LM Studio is not reachable - "
        "selecting a local model for available RAM...[/yellow]"
    )
    server = LocalLLMServer(console=console)
    try:
        server.prepare_and_start()
    except Exception:
        server.stop()
        raise

    fallback = LMStudioClient(
        base_url=server.base_url,
        model=server.model_name,
        api_key="lm-studio",
        context_tokens=server.ctx_size,
    )
    return fallback, server.stop



@app.command()
def doctor() -> None:
    """Run environment checks."""
    try:
        results = run_checks()
        render(results)
    except Exception as e:
        console.print(f"[red]Error running checks: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def transcribe(
        file: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Audio file to transcribe",
        ),
        model: str = typer.Option(
            DEFAULT_MODEL,
            "--model",
            "-m",
            help=f"Whisper model. Available: {', '.join(AVAILABLE_MODELS)}",
        ),
        prompt: str = typer.Option(
            DEFAULT_PROMPT,
            "--prompt",
            "-p",
            help=f"Prompts. Available: {', '.join(AVAILABLE_PROMPTS)}",
        ),
        language: str | None = typer.Option(
            None,
            "--language",
            "-l",
            help="Language code for transcription (auto-detect if omitted).",
        ),
        output: Path | None = typer.Option(
            None,
            "--output",
            "-o",
            help="Output text file",
        ),
        normalize: bool = typer.Option(
            False,
            "--normalize",
            "-n",
            help="Normalize audio volume before transcription.",
        ),
        compute_type: str = typer.Option(
            "auto",
            "--compute-type",
            "-ct",
            help="Compute type (auto, int8, int8_float32, float16, float32). Default: auto",
        ),
        summary: bool = typer.Option(
            False,
            "--summary",
            "-sum",
            help="Summary of transcription.",
        ),
        llm: str = typer.Option(
            None,
            "--llm",
            help="LLM model id to use for the summary "
                 "(skips interactive selection).",
        ),
        gpu: bool = typer.Option(
            False,
            "--gpu",
            "-g",
            help="Use GPU.",
        ),
        send: str = typer.Option(
            None,
            "--send",
            "-send",
            help=f"Sent to. Available: {', '.join(AVAILABLE_SEND_POINTS)}",
        ),
        unload_model: bool = typer.Option(
            True,
            "--unload-model/--no-unload-model",
            help="Unload the bundled LLM model after completion to free memory.",
        ),
) -> None:
    """Transcribe audio file."""
    if model not in AVAILABLE_MODELS:
        raise typer.BadParameter(f"Unknown model: {model}")

    if prompt not in AVAILABLE_PROMPTS:
        raise typer.BadParameter(f"Unknown prompt: {prompt}")
    if send and send not in AVAILABLE_SEND_POINTS:
        raise typer.BadParameter(f"Unknown send preset: {send}")
    if send and not summary:
        raise typer.BadParameter("--send requires --summary")
    if send and not os.getenv(f"WEBHOOK_{send.upper()}"):
        raise typer.BadParameter(f"Set WEBHOOK_{send.upper()} before sending")

    console.print(f"[cyan]Loading Whisper model:[/cyan] {model} (compute_type: {compute_type})")

    # --output задаёт базовый путь для .txt/.json (без него — рядом с исходником).
    base_output = Path(output) if output is not None else Path(file).with_suffix("")

    try:
        duration = get_audio_duration(str(file))
    except Exception:
        duration = None
    service = get_transcriber_service(
        model, language, gpu, compute_type, duration_seconds=duration,
    )

    try:
        start_time = time.perf_counter()
        transcript = transcribe_audio(file, normalize, service)
        if unload_model:
            del service
            from ai_transcriber.utils.runtime import clear_asr_models
            clear_asr_models()
        json_output = str(base_output.with_suffix(".json"))
        txt_output = str(base_output.with_suffix(".txt"))

        JsonExporter().export(transcript, json_output)
        path = TxtExporter().export(transcript, txt_output)

        elapsed = time.perf_counter() - start_time

        console.print(f"Language: [green]{transcript.language}[/green]")
        console.print(f"Duration: [green]{format_time(transcript.duration, include_ms=True)}[/green]")
        console.print(f"Transcription time: [green]{elapsed:.2f} seconds[/green]")
        console.print()

        if path:
            console.print(f"[green]File created:[/green] {path}")

            if summary:
                summaryRes = _run_summarize(path, prompt, None, llm, unload_model)
                if summaryRes is None:
                    console.print("[red]Summary failed, skipping send.[/red]")
                elif send in AVAILABLE_SEND_POINTS:
                    filepath = Path(path)
                    filename = filepath.stem

                    result = create_smart_process_item(
                        webhook_url=os.getenv(f"WEBHOOK_{send.upper()}") or None,
                        title=filename,
                        summary_file=(str(summaryRes)),
                        file=(str(path)),
                        ids=None
                    )
                    console.print("[yellow]File was sent successful[/yellow]")

        else:
            console.print("[red]ERROR: File NOT created[/red]")
            raise typer.Exit(1)

        # MEMORY CLEANUP
        cleanup_memory()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error during transcription: {e}[/red]")
        raise typer.Exit(1)


def get_transcriber_service(
    model: str,
    language: str | None,
    gpu: bool,
    compute_type: str,
    duration_seconds: float | None = None,
) -> Transcriber | WhisperXTranscriber:
    """Get the appropriate transcriber service based on GPU usage."""
    # --gpu устарел: устройство определяется автоматически.
    if HF_TOKEN:
        asr_device, torch_device, runtime_compute = resolve_runtime(
            "cuda" if gpu else "auto", duration_seconds,
        )
        resolved_compute = normalize_compute_type(
            compute_type, runtime_compute, asr_device,
        )
        from ai_transcriber.transcriber.whisperx_service import WhisperXTranscriber
        return WhisperXTranscriber(
            model_name=model,
            asr_device=asr_device,
            torch_device=torch_device,
            compute_type=resolved_compute,
            language=language,
            hf_token=HF_TOKEN,
        )
    else:
        asr_device, _torch_device, runtime_compute = resolve_runtime(
            "cuda" if gpu else "auto", duration_seconds,
        )
        resolved_compute = normalize_compute_type(
            compute_type, runtime_compute, asr_device,
        )
        from ai_transcriber.transcriber.service import Transcriber
        return Transcriber(
            model_name=model,
            device=asr_device,
            language=language,
            compute_type=resolved_compute,
        )


def transcribe_audio(file: Path, normalize: bool, service: Transcriber | WhisperXTranscriber) -> Transcript:
    """Transcribe audio file with normalization option."""
    if normalize:
        with tempfile.TemporaryDirectory() as temp_dir:
            normalized_file = Path(temp_dir) / "normalized.wav"
            console.print("[cyan]Normalizing audio...[/cyan]")
            normalize_audio(str(file), str(normalized_file))
            console.print("[green]✓ Audio normalized[/green]")
            return service.transcribe(str(normalized_file))
    else:
        return service.transcribe(str(file))


def _run_summarize(
    file: Path,
    prompt: str,
    send: str | None,
    llm: str | None,
    unload_model: bool = True,
) -> Path | None:
    """Core summarize logic shared by the `summarize` command and `transcribe`."""
    console.print(f"[cyan]Reading transcript:[/cyan] {file}")
    text = file.read_text(encoding="utf-8")

    safe_stem = "".join(
        c if (c.isalnum() or c in ("-", "_", " ")) else "_" for c in file.stem
    ).strip()[:80] or "transcript"
    summary_output = (
        file.parent / f"SUMMARY_OF_{safe_stem}_{int(time.time())}.md"
    )

    console.print("[cyan]Generating summary...[/cyan]")
    client, cleanup = resolve_llm(console, pinned=llm)

    def _cli_progress(part: int, total: int, stage: str) -> None:
        if total <= 1:
            return
        if stage == "map":
            console.print(f"[cyan]Summarizing part {part}/{total}...[/cyan]")
        elif stage == "final":
            console.print("[cyan]Building the final summary...[/cyan]")

    try:
        summary = client.summarize_long(
            text,
            prompt_type=prompt,
            on_progress=_cli_progress,
        )
    finally:
        if unload_model:
            cleanup()

    summary_output.write_text(summary, encoding="utf-8")
    console.print(f"[green]Summary created:[/green] {summary_output}")

    if send in AVAILABLE_SEND_POINTS:
        filepath = Path(summary_output)
        filename = filepath.stem

        result = create_smart_process_item(
            webhook_url=os.getenv(f"WEBHOOK_{send.upper()}") or None,
            title=filename,
            summary_file=(str(summary_output)),
            file=(str(file)),
            ids=None
        )
        console.print("[yellow]File was sent successful[/yellow]")


    return summary_output


@app.command()
def summarize(
        file: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Transcript text file to summarize",
        ),
        prompt: str = typer.Option(
            DEFAULT_PROMPT,
            "--prompt",
            "-p",
            help=f"Prompt. Available: {', '.join(AVAILABLE_PROMPTS)}",
        ),
        send: str = typer.Option(
            None,
            "--send",
            "-send",
            help=f"Sent to. Available: {', '.join(AVAILABLE_SEND_POINTS)}",
        ),
        llm: str = typer.Option(
            None,
            "--llm",
            help="LLM model id to use (skips interactive selection).",
        ),
        unload_model: bool = typer.Option(
            True,
            "--unload-model/--no-unload-model",
            help="Unload the bundled LLM model after completion to free memory.",
        ),
) -> Path | None:
    """Summarize a transcript."""
    if prompt not in AVAILABLE_PROMPTS:
        raise typer.BadParameter(f"Unknown prompt: {prompt}")

    try:
        return _run_summarize(file, prompt, send, llm, unload_model)
    except Exception as e:
        console.print(f"[red]Error during summarization: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the local AI Transcriber Web UI."""
    import uvicorn

    uvicorn.run(
        "ai_transcriber.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


@app.command()
def version() -> None:
    """Display application version."""
    typer.echo(f"ai-transcriber {__version__}")


@app.command(name="get_token")
def get_token() -> None:
    """Display hugging face token."""
    if HF_TOKEN:
        typer.echo("HF_TOKEN is set.")
    else:
        typer.echo("HF_TOKEN is not set. Set the HF_TOKEN environment variable for speaker diarization.")


if __name__ == "__main__":
    app()
