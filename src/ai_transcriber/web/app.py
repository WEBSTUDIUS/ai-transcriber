
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import httpx
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ai_transcriber.config import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_MODEL,
)
from ai_transcriber.transcriber.models import AVAILABLE_MODELS, AVAILABLE_PROMPTS

from ai_transcriber.llm.local_server import local_model_status

from .jobs import JobManager
from .models import JobOptions
from .processor import process_job


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AI_TRANSCRIBER_DATA_DIR", str(BASE_DIR.parent.parent.parent / "data"))).expanduser()
UPLOAD_DIR = DATA_DIR / "web_uploads"
OUTPUT_DIR = DATA_DIR / "web_outputs"
CONFIG_FILE = DATA_DIR / "web_config.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Transcriber Web")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
jobs = JobManager()
_config_lock = threading.RLock()


@app.middleware("http")
async def protect_browser_requests(request: Request, call_next):
    # Local APIs can start downloads and send transcripts: reject foreign pages.
    from urllib.parse import urlsplit

    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "Cross-site requests are forbidden"}, status_code=403)
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return JSONResponse({"detail": "Invalid origin"}, status_code=403)
        if parsed.netloc != request.headers.get("host") or parsed.scheme != request.url.scheme:
            return JSONResponse({"detail": "Cross-origin requests are forbidden"}, status_code=403)
    return await call_next(request)



DEFAULT_CONFIG = {
    "model": DEFAULT_MODEL,
    "backend": "auto",
    "device": "auto",
    "language": "",
    "compute_type": "auto",
    "summary_prompt_type": "meeting",
    "custom_prompt": "",
    "llm_endpoint": DEFAULT_LMSTUDIO_BASE_URL,
    "llm_model": "",
    "llm_model_custom": "",
    "send_endpoint": "",
    "transcribe": True,
    "diarization": False,
    "summary": True,
    "send": False,
    "normalize": False,
    "unload_model": True,
}

CONFIG_KEYS = set(DEFAULT_CONFIG.keys())

# Защита от SSRF: блокируем облачные метаданные и локальные трюки.
BLOCKED_LLM_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.google",
    "0.0.0.0",
}

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 ГБ
ALLOWED_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".mkv",
    ".flac", ".ogg", ".opus", ".aac", ".webm", ".wma",
    ".txt", ".md", ".log", ".text",
}


def _validate_llm_base_url(base_url: str) -> str:
    from urllib.parse import urlparse

    raw = (base_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="LLM endpoint is empty")
    if len(raw) > 500:
        raise HTTPException(status_code=400, detail="LLM endpoint too long")
    try:
        parsed = urlparse(raw)
        parsed.port
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid LLM endpoint URL")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="LLM endpoint must be http(s)")
    host = (parsed.hostname or "").lower()
    if not host or host in BLOCKED_LLM_HOSTS:
        raise HTTPException(status_code=400, detail="LLM endpoint host is blocked")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HTTPException(status_code=400, detail="Endpoint must not contain credentials, query or fragment")
    return raw.rstrip("/")


def _sanitize_config_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid config payload")
    # Mass-assignment fix: только известные ключи, без мусора.
    clean = {k: payload[k] for k in CONFIG_KEYS if k in payload}
    for key, value in clean.items():
        if isinstance(DEFAULT_CONFIG[key], str) and (not isinstance(value, str) or len(value) > 20000):
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}")
    # Лёгкая валидация типов для булевых флагов.
    for key in ("transcribe", "diarization", "summary", "send", "normalize", "unload_model"):
        if key in clean and not isinstance(clean[key], bool):
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}")
    return clean


WEBHOOK_ENV_VARS = {
    "it_task": "WEBHOOK_IT_TASK",
    "hr": "WEBHOOK_HR",
    "call_track": "WEBHOOK_CALL_TRACK",
    "custom_webhook": "WEBHOOK_CUSTOM",
}


def available_send_endpoints() -> list[dict]:
    """
    Named Bitrix webhooks, configured through environment variables.

    Example (.env):
        WEBHOOK_IT_TASK=https://your-portal.example/rest/USER/TOKEN/
    """
    endpoints = []
    for name, env_var in WEBHOOK_ENV_VARS.items():
        url = os.getenv(env_var, "").strip()
        if url:
            endpoints.append({"name": name, "url": name})
    return endpoints


def available_devices() -> list[dict]:
    """
    Devices selectable in the UI. Only options available on the
    current machine are exposed (e.g. MPS only on macOS).
    """
    devices = [{"value": "auto", "label": "Auto"}]
    devices.append({"value": "cpu", "label": "CPU — надёжный режим"})
    if torch.backends.mps.is_available():
        devices.append({"value": "mps", "label": "CPU + MPS — Apple Silicon"})
    if torch.cuda.is_available():
        devices.append({"value": "cuda", "label": "CUDA — NVIDIA GPU"})
    return devices


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        config = {**DEFAULT_CONFIG, **_sanitize_config_payload(data)}
        config["send_endpoint"] = _persisted_send_endpoint(config["send_endpoint"])
        return config
    except Exception:
        return DEFAULT_CONFIG.copy()


def _persisted_send_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if endpoint in WEBHOOK_ENV_VARS:
        return endpoint
    for name, env_var in WEBHOOK_ENV_VARS.items():
        if endpoint and endpoint == os.getenv(env_var, "").strip():
            return name
    # Manual URLs are valid for the current job, but never saved to preferences.
    return ""


def save_config(data: dict) -> None:
    data = {**data, "send_endpoint": _persisted_send_endpoint(data.get("send_endpoint", ""))}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Unique temporary files avoid collisions between concurrent saves.
    with _config_lock:
        fd, filename = tempfile.mkstemp(dir=DATA_DIR, prefix=".web-config-", suffix=".tmp")
        temporary = Path(filename)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            temporary.replace(CONFIG_FILE)
        finally:
            temporary.unlink(missing_ok=True)


def _cleanup_old_outputs(keep_id: str, keep_last: int = 5) -> None:
    """
    Удаляет старые загрузки/результаты, оставляя текущий job
    и последние N завершённых. Не трогает in-flight файлы:
    имена в UPLOAD_DIR уникальны по job.id.
    """
    if UPLOAD_DIR.exists():
        uploads = sorted(
            (p for p in UPLOAD_DIR.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )
        # Держим файлы текущего job + последние несколько.
        keep_names = {keep_id} | jobs.active_ids()
        for p in uploads[-keep_last:]:
            # job.id — префикс имени файла "<jobid><suffix>"
            keep_names.add(p.name[:32])
        for path in uploads:
            if not any(path.name.startswith(k) for k in keep_names):
                path.unlink(missing_ok=True)

    if OUTPUT_DIR.exists():
        dirs = sorted(
            (p for p in OUTPUT_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )
        keep_dirs = {keep_id} | jobs.active_ids() | {p.name for p in dirs[-keep_last:]}
        for path in dirs:
            if path.name not in keep_dirs:
                shutil.rmtree(path, ignore_errors=True)


def cleanup_previous_data(current_job_id: str) -> None:
    """Backward-compat: делегирует безопасной очистке."""
    _cleanup_old_outputs(keep_id=current_job_id)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    from ai_transcriber.transcriber.models import MODEL_INFO

    cfg = load_config()
    # Старые сохранённые конфиги могли хранить int8 как дефолт —
    # мигрируем на auto, чтобы CUDA получал float16.
    if cfg.get("compute_type") not in ("auto", "int8", "int8_float32", "float16", "float32"):
        cfg["compute_type"] = "auto"
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": AVAILABLE_MODELS,
            "model_info": MODEL_INFO,
            "prompts": AVAILABLE_PROMPTS,
            "config": cfg,
            "send_endpoints": available_send_endpoints(),
            "devices": available_devices(),
            "local_model": local_model_status(),
        },
    )


@app.get("/api/llm/models")
async def llm_models(base_url: str = DEFAULT_LMSTUDIO_BASE_URL):
    """List models served by the LLM endpoint (OpenAI-compatible /models)."""
    safe_base = _validate_llm_base_url(base_url)
    url = f"{safe_base}/models"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(url, headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'lm-studio')}"
            })
            response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        models = sorted({
            item.get("id")
            for item in data
            if isinstance(item, dict) and item.get("id")
        })
        return {"models": models}
    except HTTPException:
        raise
    except Exception as exc:
        # Не светим внутренности: короткое сообщение без URL/трейса.
        local = local_model_status()
        return {"models": [], "error": f"LLM endpoint unreachable: {type(exc).__name__}",
                "fallback_model": local["model"], "fallback_ready": local["fits"],
                "fallback_reason": local["reason"]}


@app.get("/api/local-model")
async def get_local_model():
    return local_model_status()


@app.get("/api/config")
async def get_config():
    return load_config()


@app.post("/api/config")
async def set_config(payload: dict):
    clean = _sanitize_config_payload(payload)
    with _config_lock:
        current = load_config()
        current.update(clean)
        save_config(current)
    return current


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    snapshot = jobs.snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return snapshot


@app.get("/api/jobs/{job_id}/file")
async def get_job_file(job_id: str, kind: str = "transcript"):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if kind not in ("transcript", "summary"):
        raise HTTPException(status_code=400, detail="Unknown file kind")
    path = job.transcript_file if kind == "transcript" else job.summary_file
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path,
        filename=Path(path).name,
        media_type="text/markdown" if Path(path).suffix.lower() == ".md" else "text/plain",
    )


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcribe: bool = Form(True),
    diarization: bool = Form(False),
    summary: bool = Form(True),
    send: bool = Form(False),
    normalize: bool = Form(False),
    model: str = Form(DEFAULT_MODEL),
    backend: str = Form("auto"),
    device: str = Form("auto"),
    language: str = Form(""),
    compute_type: str = Form("auto"),
    summary_prompt_type: str = Form("meeting"),
    custom_prompt: str = Form(""),
    llm_endpoint: str = Form(DEFAULT_LMSTUDIO_BASE_URL),
    llm_model: str = Form(""),
    llm_model_custom: str = Form(""),
    send_endpoint: str = Form(""),
    unload_model: bool = Form(True),
):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    if backend not in ("auto", "whisperx", "faster-whisper"):
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")
    if device not in ("auto", "cpu", "mps", "cuda"):
        raise HTTPException(status_code=400, detail=f"Unknown device: {device}")
    if compute_type not in ("auto", "int8", "int8_float32", "float16", "float32"):
        raise HTTPException(status_code=400, detail=f"Unknown compute_type: {compute_type}")
    if summary_prompt_type not in (*AVAILABLE_PROMPTS, "custom"):
        raise HTTPException(status_code=400, detail="Unknown prompt type")

    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is empty")

    # Только basename: защита от path traversal через имя файла.
    safe_filename = Path(file.filename).name[:200] or "upload"
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix or 'no extension'}",
        )
    text_extensions = {".txt", ".md", ".log", ".text"}
    if suffix in text_extensions:
        # Text files are already transcripts: transcription/audio-only options are invalid.
        transcribe = False
        diarization = False
        normalize = False
    elif not transcribe:
        raise HTTPException(
            status_code=400,
            detail="Transcription can only be disabled for text transcript files (.txt, .md, .log).",
        )

    if not transcribe and not summary and not send:
        raise HTTPException(
            status_code=400,
            detail="Enable at least one action",
        )

    # llm_model может прийти из селектора или из ручного поля.
    resolved_llm_model = (llm_model or "").strip() or (llm_model_custom or "").strip() or None
    if resolved_llm_model and len(resolved_llm_model) > 200:
        raise HTTPException(status_code=400, detail="LLM model id too long")

    if send and (not summary or not send_endpoint.strip()):
        raise HTTPException(status_code=400, detail="Sending requires a summary and webhook endpoint")
    if summary:
        llm_endpoint = _validate_llm_base_url(llm_endpoint)
        if summary_prompt_type == "custom" and not custom_prompt.strip():
            raise HTTPException(status_code=400, detail="Custom prompt is empty")

    resolved_send_endpoint = send_endpoint.strip()
    if send:
        if resolved_send_endpoint in WEBHOOK_ENV_VARS:
            resolved_send_endpoint = os.getenv(WEBHOOK_ENV_VARS[resolved_send_endpoint], "").strip()
        if not resolved_send_endpoint:
            raise HTTPException(status_code=400, detail="Webhook preset is not configured")
        from ai_transcriber.exporters.webhook_exporter import validate_webhook_url
        try:
            resolved_send_endpoint = validate_webhook_url(resolved_send_endpoint)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    try:
        job = jobs.create(safe_filename)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    safe_name = f"{job.id}{suffix}"
    input_path = UPLOAD_DIR / safe_name

    # Не блокируем event loop: читаем upload чанками в треде.
    # Лимит размера — ранний выход до записи на диск.
    import anyio as _anyio

    def _save_upload() -> None:
        size = 0
        with input_path.open("wb") as output:
            while True:
                chunk = file.file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    try:
                        output.close()
                        input_path.unlink(missing_ok=True)
                    finally:
                        raise HTTPException(
                            status_code=413,
                            detail="File too large (max 2 GB)",
                        )
                output.write(chunk)
            if size == 0:
                input_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="File is empty")

    try:
        await _anyio.to_thread.run_sync(_save_upload)
    except HTTPException:
        jobs.discard(job.id)
        raise
    except Exception as exc:
        jobs.discard(job.id)
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Failed to save upload")

    # cleanup_previous_data больше не трогает in-flight задачи:
    # чистим только старые job-директории, текущий job не удаляем.
    # (Раньше удаление было здесь и гонялось с параллельным аплоадом.)
    try:
        _cleanup_old_outputs(keep_id=job.id)
    except Exception:
        pass
    job.log("Cleaned up old job data.")

    options = JobOptions(
        transcribe=transcribe,
        diarization=diarization,
        summary=summary,
        send=send,
        normalize=normalize,
        model=model,
        backend=backend,
        device=device,
        language=(language or "").strip() or None,
        compute_type=compute_type,
        summary_prompt_type=summary_prompt_type,
        custom_prompt=(custom_prompt or "").strip() or None,
        llm_endpoint=(llm_endpoint or "").strip() or DEFAULT_LMSTUDIO_BASE_URL,
        llm_model=resolved_llm_model,
        unload_model=unload_model,
        send_endpoint=resolved_send_endpoint or None,
    )

    config = {
        "model": model,
        "backend": backend,
        "device": device,
        "language": language,
        "compute_type": compute_type,
        "summary_prompt_type": summary_prompt_type,
        "custom_prompt": custom_prompt,
        "llm_endpoint": llm_endpoint,
        "llm_model": resolved_llm_model or "",
        "llm_model_custom": (llm_model_custom or "").strip(),
        "send_endpoint": send_endpoint,
        "transcribe": transcribe,
        "diarization": diarization,
        "summary": summary,
        "send": send,
        "normalize": normalize,
        "unload_model": unload_model,
    }
    try:
        save_config(config)
    except OSError:
        # Preferences are optional; an accepted upload must still be processed.
        job.log("Could not save UI preferences; continuing with this job.")

    output_dir = OUTPUT_DIR / job.id
    job.source_file = safe_filename
    job.output_dir = str(output_dir)

    background_tasks.add_task(
        jobs.run,
        job.id,
        lambda: process_job(
            job,
            input_path,
            output_dir,
            options,
        ),
    )

    return {"job_id": job.id}
