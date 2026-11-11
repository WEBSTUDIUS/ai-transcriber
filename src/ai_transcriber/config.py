from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path)

# Loading an existing .bin alignment model must not start a non-daemon
# Transformers conversion job against an external Hugging Face service.
# Existing safetensors files remain preferred; model downloads still work.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

if os.name == "nt":
    ffmpeg_path = os.getenv("FFMPEG_DLL_PATH", r"C:\ffmpeg7\bin")
    if Path(ffmpeg_path).exists():
        os.add_dll_directory(ffmpeg_path)

import torch

HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_MODEL = "large-v3-turbo"

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
    DEFAULT_TORCH_DEVICE = "cuda"
    DEFAULT_COMPUTE_TYPE = "float16"
    # Ограничение VRAM только по явному запросу: импорт не должен
    # резать память для любого `cli --help` / `doctor`.
    _frac = os.getenv("CUDA_MEM_FRACTION", "").strip()
    if _frac:
        try:
            torch.cuda.set_per_process_memory_fraction(float(_frac))
        except Exception:
            pass
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "cpu"
    DEFAULT_TORCH_DEVICE = "mps"
    DEFAULT_COMPUTE_TYPE = "int8"
else:
    DEFAULT_DEVICE = "cpu"
    DEFAULT_TORCH_DEVICE = "cpu"
    DEFAULT_COMPUTE_TYPE = "int8"

DEFAULT_LANGUAGE: str | None = None
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LMSTUDIO_TIMEOUT_SECONDS = 3
DEFAULT_PROMPT = "meeting"

#
# Bundled local LLM fallback (llama.cpp + GGUF).
# Used by the CLI when LM Studio is not reachable.
#
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "1234"))
LLAMA_CTX = int(os.getenv("LLAMA_CTX", "8192"))
LLAMA_MODEL_REPO = os.getenv(
    "LLAMA_MODEL_REPO",
    "unsloth/Qwen3-4B-Instruct-2507-GGUF",
)
LLAMA_MODEL_FILE = os.getenv(
    "LLAMA_MODEL_FILE",
    "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
)
LLAMA_CACHE_DIR = Path(
    os.getenv(
        "LLAMA_CACHE_DIR",
        str(Path.home() / ".cache" / "ai-transcriber"),
    )
)

_default_ngpu = "0"
if torch.cuda.is_available() or torch.backends.mps.is_available():
    _default_ngpu = "99"
LLAMA_GPU_LAYERS = int(os.getenv("LLAMA_NGPU", _default_ngpu))

# Quantized KV cache keeps the 4B model practical on 8 GB machines.
LLAMA_KV_CACHE_TYPE = os.getenv("LLAMA_KV_CACHE_TYPE", "q8_0")
