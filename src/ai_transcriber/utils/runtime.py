import gc
import os

import torch

#
# In "auto" mode, files longer than this (seconds) run on CPU even on
# Apple Silicon: unified memory cannot hold the full MPS pipeline
# (Whisper + alignment + diarization) for long recordings.
#
MPS_LONG_FILE_THRESHOLD_SECONDS = float(
    os.getenv("MPS_LONG_FILE_THRESHOLD_SECONDS", "600")
)


def detect_runtime() -> tuple[str, str, str]:
    """
    Returns:
        asr_device,
        torch_device,
        compute_type
    """

    if torch.cuda.is_available():
        return (
            "cuda",
            "cuda",
            "float16",
        )

    if torch.backends.mps.is_available():
        return (
            "cpu",
            "mps",
            "int8",
        )

    return (
        "cpu",
        "cpu",
        "int8",
    )


def resolve_runtime(
    device: str = "auto",
    duration_seconds: float | None = None,
) -> tuple[str, str, str]:
    """
    Resolve a user-selected device into (asr_device, torch_device, compute_type).

    Choices:
        auto  — best available (detect_runtime)
        cpu   — everything on CPU
        mps   — ASR on CPU, alignment/diarization on Apple Silicon MPS
        cuda  — everything on NVIDIA CUDA

    In "auto" mode long recordings fall back to CPU on MPS machines to
    avoid unified-memory exhaustion. An explicit "mps" choice is always
    honored, even for long files.
    """
    choice = (device or "auto").strip().lower()

    if choice in ("", "auto"):
        asr_device, torch_device, compute_type = detect_runtime()
        if (
            torch_device == "mps"
            and MPS_LONG_FILE_THRESHOLD_SECONDS > 0
            and duration_seconds is not None
            and duration_seconds > MPS_LONG_FILE_THRESHOLD_SECONDS
        ):
            return ("cpu", "cpu", "int8")
        return (asr_device, torch_device, compute_type)

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA selected but not available on this machine."
            )
        return ("cuda", "cuda", "float16")

    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS selected but not available (macOS Apple Silicon only)."
            )
        return ("cpu", "mps", "int8")

    # "cpu" and anything else
    return ("cpu", "cpu", "int8")

def cleanup_memory():

    gc.collect()

    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    gc.collect()


VALID_COMPUTE_TYPES = {"int8", "int8_float32", "float16", "float32"}


def normalize_compute_type(
    requested: str | None,
    runtime_compute: str,
    asr_device: str,
) -> str:
    """
    Resolve user-requested compute type.

    - "" / None / "auto" -> runtime default
    - unknown values -> runtime default
    - int8/int8_float32 on CUDA are not supported by faster-whisper/
      ctranslate2 -> fall back to float16
    """
    choice = (requested or "").strip().lower()
    if choice in ("", "auto"):
        return runtime_compute
    if choice not in VALID_COMPUTE_TYPES:
        return runtime_compute
    if asr_device == "cpu" and choice == "float16":
        return "float32"
    if asr_device == "cuda" and choice in ("int8", "int8_float32"):
        return "float16"
    return choice

def clear_asr_models() -> None:
    """Release loaded engines without importing unused optional ML stacks."""
    import sys
    for name, method in (
        ("ai_transcriber.transcriber.service", "clear_model_cache"),
        ("ai_transcriber.transcriber.whisperx_service", "clear_whisperx_caches"),
    ):
        module = sys.modules.get(name)
        if module is not None:
            getattr(module, method)()
    cleanup_memory()
