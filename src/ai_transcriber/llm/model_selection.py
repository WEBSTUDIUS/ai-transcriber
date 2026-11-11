"""Conservative RAM budgets for local GGUF inference (q8 KV, <=8K context)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import psutil

GIB = 1024 ** 3


@dataclass(frozen=True)
class Memory:
    total: int
    available: int


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    repo: str
    filename: str
    required_gib: float
    context: int = 8192


# Runtime estimates include weights, q8 KV and inference workspace, not just disk size.
# 2B is intentionally excluded: it failed our Russian task-retention checks.
PROFILES = (
    Profile('4b', 'Qwen3 4B Instruct', 'unsloth/Qwen3-4B-Instruct-2507-GGUF',
            'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', 3.5),
    Profile('8b', 'Qwen3 8B', 'unsloth/Qwen3-8B-GGUF', 'Qwen3-8B-Q4_K_M.gguf', 7.0),
    Profile('14b', 'Qwen3 14B', 'unsloth/Qwen3-14B-GGUF', 'Qwen3-14B-Q4_K_M.gguf', 11.0),
)


@dataclass(frozen=True)
class Selection:
    profile: Profile
    memory: Memory | None
    budget: int
    mode: str
    fits: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            'mode': self.mode, 'profile': self.profile.id, 'label': self.profile.label,
            'model': self.profile.filename, 'context': self.profile.context,
            'total_gib': round(self.memory.total / GIB, 2) if self.memory else None,
            'available_gib': round(self.memory.available / GIB, 2) if self.memory else None,
            'budget_gib': round(self.budget / GIB, 2),
            'required_gib': self.profile.required_gib, 'fits': self.fits, 'reason': self.reason,
        }


def detect_memory() -> Memory | None:
    try:
        value = psutil.virtual_memory()
        if value.total <= 0 or value.available < 0:
            return None
        return Memory(value.total, min(value.available, value.total))
    except (OSError, RuntimeError, AttributeError):
        return None


def memory_budget(memory: Memory | None) -> int:
    if memory is None:
        return 0
    # Keep >=2 GiB / 35% of total outside the model and 0.5 GiB of available
    # memory for new allocations. Available includes reclaimable OS cache.
    return max(0, min(memory.total - max(2 * GIB, int(memory.total * .35)),
                      memory.available - GIB // 2))


def select_model(memory: Memory | None, mode: str = 'auto',
                 repo: str = '', filename: str = '', context: int = 8192,
                 cache_type: str = 'q8_0') -> Selection:
    if context < 4096:
        raise ValueError('LLAMA_CTX must be at least 4096')
    if cache_type not in ('q8_0', 'f16'):
        raise ValueError('LLAMA_KV_CACHE_TYPE must be q8_0 or f16 for RAM budgeting')
    budget = memory_budget(memory)
    if bool(repo) != bool(filename):
        raise ValueError('Set both LLAMA_MODEL_REPO and LLAMA_MODEL_FILE')
    if mode == 'manual':
        if not repo:
            raise ValueError('Manual mode requires LLAMA_MODEL_REPO and LLAMA_MODEL_FILE')
        if filename != os.path.basename(filename) or '/' in filename or '\\' in filename:
            raise ValueError('LLAMA_MODEL_FILE must be a filename, not a path')
        profile = Profile('manual', 'Своя модель', repo, filename, 0, context)
        return Selection(profile, memory, budget, mode, True,
                         'Ручная модель: оценка расхода памяти неизвестна')
    if mode != 'auto' and mode not in {p.id for p in PROFILES}:
        raise ValueError('LLAMA_MODEL_PROFILE must be auto, 4b, 8b, 14b or manual')
    profiles = []
    for profile in PROFILES:
        # KV estimate scales with context and cache precision. Never shrink the
        # user's context silently: the client and server must use the same size.
        cache_factor = 2 if cache_type == 'f16' else 1
        extra = max(0, context / 8192 * cache_factor - 1) * (1 if profile.id == '4b' else 2)
        profiles.append(Profile(profile.id, profile.label, profile.repo, profile.filename,
                                profile.required_gib + extra, context))
    candidates = [p for p in profiles if p.required_gib * GIB <= budget]
    chosen = (candidates[-1] if candidates else profiles[0]) if mode == 'auto' else next(p for p in profiles if p.id == mode)
    fits = chosen.required_gib * GIB <= budget
    if memory is None:
        reason = 'Не удалось определить доступную RAM. Автозапуск приостановлен.'
    elif not fits:
        reason = f'Нужно освободить память: для модели требуется {chosen.required_gib:.1f} ГиБ, бюджет {budget / GIB:.1f} ГиБ.'
    else:
        reason = 'Самая крупная модель, которая помещается в доступный бюджет RAM.' if mode == 'auto' else 'Профиль выбран вручную; проверка памяти пройдена.'
    return Selection(chosen, memory, budget, mode, fits, reason)


def configured_selection(context: int | None = None) -> Selection:
    repo = os.getenv('LLAMA_MODEL_REPO', '').strip()
    filename = os.getenv('LLAMA_MODEL_FILE', '').strip()
    mode = os.getenv('LLAMA_MODEL_PROFILE', '').strip().lower() or ('manual' if repo or filename else 'auto')
    # Explicit auto/profile takes priority over legacy pinned repo/file values.
    return select_model(detect_memory(), mode,
                        repo=repo if mode == 'manual' else '',
                        filename=filename if mode == 'manual' else '',
                        context=context if context is not None else int(os.getenv('LLAMA_CTX', '8192')),
                        cache_type=os.getenv('LLAMA_KV_CACHE_TYPE', 'q8_0'))
