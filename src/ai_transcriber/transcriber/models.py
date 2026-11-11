from dataclasses import dataclass, field

AVAILABLE_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
]

# Подсказки для селектора: память и назначение.
# RAM — CPU RAM без диаризации (int8), VRAM — полный WhisperX-пайплайн.
MODEL_INFO = {
    "tiny": {"params": "39M", "ram": "~1 ГБ", "vram": "~1 ГБ", "hint": "Смоки, слабый сервер"},
    "base": {"params": "74M", "ram": "~1-2 ГБ", "vram": "~1 ГБ", "hint": "Черновики, короткие клипы"},
    "small": {"params": "244M", "ram": "~2-3 ГБ", "vram": "~1.5 ГБ", "hint": "Баланс, сервер 4 ГБ"},
    "medium": {"params": "769M", "ram": "~4-6 ГБ", "vram": "~3 ГБ", "hint": "Точнее, нужно 8+ ГБ"},
    "large-v3": {"params": "1550M", "ram": "~10-16 ГБ", "vram": "~5 ГБ", "hint": "Макс. точность, 16+ ГБ"},
    "large-v3-turbo": {"params": "809M", "ram": "~5-8 ГБ", "vram": "~3 ГБ", "hint": "Дефолт: почти как large-v3, в 8× быстрее"},
}

AVAILABLE_PROMPTS = [
    "call",
    "meeting",
]

AVAILABLE_SEND_POINTS = [
    'it_task',
    'hr',
    'call_track'
]

@dataclass
class TranscriptSegment:

    start: float
    end: float
    text: str

    words: list[dict] = field(
        default_factory=list
    )

    speaker: str | None = None

@dataclass(slots=True)
class Transcript:
    language: str
    duration: float
    segments: list[TranscriptSegment]
