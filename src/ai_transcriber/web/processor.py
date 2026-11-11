
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ai_transcriber.audio.processor import normalize_audio, get_audio_duration
from ai_transcriber.config import HF_TOKEN, LLAMA_CTX
from ai_transcriber.exporters.json_exporter import JsonExporter
from ai_transcriber.exporters.txt_exporter import TxtExporter, transcript_to_text
from ai_transcriber.exporters.webhook_exporter import create_smart_process_item
from ai_transcriber.llm.client import LMStudioClient
from ai_transcriber.llm.local_server import get_shared_server, stop_shared_server
from ai_transcriber.transcriber.models import Transcript
from ai_transcriber.utils.runtime import (
    cleanup_memory,
    clear_asr_models,
    normalize_compute_type,
    resolve_runtime,
)

from .models import Job, JobOptions


def _emit(job: Job, message: str) -> None:
    job.log(message)


def _set(job: Job, step: str, progress: int) -> None:
    job.current_step = step
    job.progress = progress


def _transcript_to_text(transcript: Transcript) -> str:
    return transcript_to_text(transcript)


def _whisperx_service(
    job: Job,
    options: JobOptions,
    duration_seconds: float | None = None,
) -> WhisperXTranscriber:
    if options.diarization and not HF_TOKEN:
        raise ValueError("WhisperX requires HF_TOKEN to be set.")
    asr_device, torch_device, runtime_compute = resolve_runtime(
        options.device,
        duration_seconds,
    )
    compute_type = normalize_compute_type(
        options.compute_type, runtime_compute, asr_device,
    )
    _emit(
        job,
        f"WhisperX runtime: ASR={asr_device}, "
        f"Torch={torch_device}, compute={compute_type}",
    )
    from ai_transcriber.transcriber.whisperx_service import WhisperXTranscriber
    return WhisperXTranscriber(
        model_name=options.model,
        asr_device=asr_device,
        torch_device=torch_device,
        compute_type=compute_type,
        language=options.language,
        hf_token=HF_TOKEN,
        enable_diarization=options.diarization,
    )


def _faster_whisper_service(
    job: Job,
    options: JobOptions,
    duration_seconds: float | None = None,
) -> Transcriber:
    _emit(job, "Backend: faster-whisper (without speaker diarization).")
    if options.diarization:
        _emit(
            job,
            "Warning: diarization is not supported "
            "by faster-whisper and will be skipped.",
        )
    asr_device, _, runtime_compute = resolve_runtime(
        options.device,
        duration_seconds,
    )
    compute_type = normalize_compute_type(
        options.compute_type, runtime_compute, asr_device,
    )
    _emit(
        job,
        f"faster-whisper device: {asr_device}, compute: {compute_type}",
    )
    from ai_transcriber.transcriber.service import Transcriber
    return Transcriber(
        model_name=options.model,
        device=asr_device,
        language=options.language,
        compute_type=compute_type,
    )


def _redact_url(url: str | None) -> str:
    if not url:
        return ""
    # Не светим токен вебхука в логах: показываем хост + префикс.
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or ""
        return f"{parsed.scheme}://{host}/…"
    except Exception:
        return "<webhook>"


def _get_service(
    job: Job,
    options: JobOptions,
    audio_file: str | None = None,
    duration_seconds: float | None = None,
):
    backend = (options.backend or "auto").strip().lower()

    #
    # Длительность уже посчитана вызывающим кодом (один ffprobe на job),
    # сюда приходит готовое значение чтобы не спавнить ffprobe дважды.
    #
    if duration_seconds is None and (
        (options.device or "auto").strip().lower() in ("", "auto")
        and audio_file
    ):
        try:
            duration_seconds = get_audio_duration(audio_file)
        except Exception:
            duration_seconds = None

    if backend == "faster-whisper":
        return _faster_whisper_service(job, options, duration_seconds)

    if backend == "whisperx":
        return _whisperx_service(job, options, duration_seconds)

    # Auto: WhisperX when HF_TOKEN is configured, otherwise faster-whisper.
    if HF_TOKEN:
        return _whisperx_service(job, options, duration_seconds)

    _emit(
        job,
        "HF_TOKEN is not set; using faster-whisper.",
    )
    return _faster_whisper_service(job, options, duration_seconds)

def _summarize_with_llm(
    job: Job,
    options: JobOptions,
    transcript_text: str,
) -> str:
    """
    Generate a summary using the configured LLM endpoint.

    If the endpoint is unreachable, falls back to the bundled llama.cpp
    server with the configured GGUF model, started automatically.
    """
    summarize_kwargs = {
        "prompt_type": options.summary_prompt_type,
        "custom_prompt": options.custom_prompt,
    }

    client = LMStudioClient(
        base_url=options.llm_endpoint,
        model=options.llm_model,
    )

    def _report_parts(part: int, total: int, stage: str) -> None:
        if total <= 1:
            return
        if stage == "map":
            _emit(job, f"Summarizing part {part}/{total}...")
        elif stage == "reduce":
            _emit(job, f"Merging summaries {part}/{total}...")
        elif stage == "final":
            _emit(job, "Building the final summary...")

    try:
        summary = client.summarize_long(
            transcript_text,
            ctx_tokens=LLAMA_CTX,
            on_progress=_report_parts,
            **summarize_kwargs,
        )
        _emit(job, f"Summary generated with {_redact_url(options.llm_endpoint)}")
        return summary
    except Exception as exc:
        _emit(job, f"LLM endpoint {_redact_url(options.llm_endpoint)} failed: {type(exc).__name__}")
        _emit(job, "Selecting local summary model for available RAM...")
        _set(job, "Preparing bundled LLM (download if needed)", 86)
        _emit(
            job,
            "Preparing local llama.cpp model "
            "(first run downloads the model)...",
        )

        last_logged = {"decile": -1}

        def _llm_progress(percent: float) -> None:
            _set(job, "Preparing bundled LLM (download if needed)", 86 + int(percent * 0.10))
            decile = int(percent // 5)
            if decile != last_logged["decile"]:
                last_logged["decile"] = decile
                _emit(job, f"Local LLM preparation: {percent:.0f}%")

        server = get_shared_server(log=job.log, on_progress=_llm_progress)
        _set(job, "Generating summary", 96)
        fallback = LMStudioClient(
            base_url=server.base_url,
            model=server.model_name,
            api_key="lm-studio",
        )
        summary = fallback.summarize_long(
            transcript_text,
            ctx_tokens=server.ctx_size,
            on_progress=_report_parts,
            **summarize_kwargs,
        )
        _emit(job, f"Summary generated with bundled {server.model_name}")
        return summary


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_job(
    job: Job,
    input_file: Path,
    output_dir: Path,
    options: JobOptions,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _process_job_inner(job, input_file, output_dir, options)
    finally:
        # "Unload model after completion" (default on): stop the shared
        # llama.cpp server to free ~6 GB of memory as soon as the job ends,
        # even when it fails. The GGUF stays on disk, so the next job only
        # reloads it into RAM (no re-download).
        try:
            if options.unload_model and stop_shared_server():
                _emit(job, "Bundled LLM unloaded (memory freed).")
        finally:
            # Всегда возвращаем память ASR/LLM аллокаторов в долгоживущем web.
            if options.unload_model:
                clear_asr_models()
            cleanup_memory()
            # Не держим полные тексты в RAM вечно: файлы уже на диске,
            # snapshot() их всё равно выкидывает.
            job.transcript_text = None
            job.summary_text = None


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "transcript"
    safe = "".join(
        c if (c.isalnum() or c in ("-", "_", " ", ".")) else "_" for c in stem
    ).strip()
    return safe[:100] or "transcript"


def _process_job_inner(
    job: Job,
    input_file: Path,
    output_dir: Path,
    options: JobOptions,
) -> None:
    source = input_file
    transcript_text = ""
    transcript_path: Path | None = None
    # Имя файла от клиента нельзя использовать напрямую для путей:
    # только basename + санитизированный stem.
    source_name = Path(job.filename or input_file.name).name
    source_stem = _safe_stem(source_name)

    if options.transcribe:
        _set(job, "Loading transcription model", 5)
        _emit(job, f"Input: {input_file.name}")

        # Один ffprobe на job: длительность нужна и для лога,
        # и для resolve_runtime (MPS fallback).
        duration_seconds: float | None = None
        try:
            duration_seconds = get_audio_duration(str(source))
            _emit(job, f"Duration: {format_duration(duration_seconds)}")
        except Exception as exc:
            _emit(job, f"Could not read duration, continuing: {exc}")

        if options.normalize:
            normalized = output_dir / f"{source_stem}.normalized.wav"
            _set(job, "Normalizing audio", 10)
            normalize_audio(str(input_file), str(normalized))
            source = normalized

        _set(job, "Transcribing", 20)
        _emit(job, "Running transcription + alignment...")
        service = _get_service(
            job, options, audio_file=str(source), duration_seconds=duration_seconds,
        )
        transcript = service.transcribe(str(source))
        # On 8 GB machines ASR must release memory before loading the LLM.
        del service
        if options.unload_model:
            clear_asr_models()

        _set(job, "Writing transcript", 70)
        transcript_text = _transcript_to_text(transcript)

        transcript_path = output_dir / f"{source_stem}.txt"
        json_path = output_dir / f"{source_stem}.json"
        JsonExporter().export(transcript, str(json_path))
        TxtExporter().export(transcript, str(transcript_path))

        _emit(job, f"Transcript created: {transcript_path.name}")
        job.transcript_file = str(transcript_path)
        job.transcript_text = transcript_text

    else:
        if input_file.suffix.lower() not in {".txt", ".md", ".log", ".text"}:
            raise ValueError(
                "Transcription is disabled, so the input file must be a text transcript (.txt, .md, .log)."
            )

        _set(job, "Reading transcript", 20)
        transcript_text = input_file.read_text(encoding="utf-8")
        # Для текстового входа тоже используем безопасное имя.
        transcript_path = output_dir / f"{source_stem}.txt"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        job.transcript_file = str(transcript_path)
        job.transcript_text = transcript_text
        _emit(job, f"Transcript copied: {transcript_path.name}")

    if options.summary:
        _set(job, "Generating summary", 85)
        _emit(job, "Generating summary with LLM...")

        summary = _summarize_with_llm(job, options, transcript_text)

        summary_path = output_dir / f"SUMMARY_OF_{source_stem}.md"
        summary_header = (
            f"# Summary: {source_name}\n\n"
            f"**Source file:** `{source_name}`\n\n"
            "---\n\n"
        )
        summary_path.write_text(
            summary_header + summary.lstrip(),
            encoding="utf-8",
        )

        job.summary_file = str(summary_path)
        job.summary_text = summary

        _emit(job, f"Summary created: {summary_path.name}")

    if options.send:
        if not options.send_endpoint:
            raise ValueError("Send is enabled but send endpoint is empty")

        if not job.summary_file:
            raise ValueError(
                "Sending to webhook requires a summary "
                "(enable 'Generate summary')."
            )

        _set(job, "Sending result", 97)
        _emit(job, f"Sending result to {_redact_url(options.send_endpoint)}")

        result = create_smart_process_item(
            webhook_url=options.send_endpoint,
            title=source_stem,
            summary_file=job.summary_file,
            file=str(transcript_path),
        )

        job.send_result = result
        _emit(job, "Send completed.")

    _set(job, "Ready", 100)
    _emit(job, "Processing completed successfully.")
