from faster_whisper import WhisperModel

from ai_transcriber.config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL
from ai_transcriber.transcriber.models import Transcript, TranscriptSegment

_model_cache: dict[str, WhisperModel] = {}

MAX_CACHED_MODELS = 2


def clear_model_cache() -> None:
    """Evict cached faster-whisper models (used by long-lived web process)."""
    global _model_cache
    for model in list(_model_cache.values()):
        try:
            del model
        except Exception:
            pass
    _model_cache = {}


class Transcriber:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:

        cache_key = f"{model_name}:{device}:{compute_type}"
        if cache_key in _model_cache:
            self._model = _model_cache[cache_key]
        else:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            # Bound the cache: evict oldest entry instead of growing forever.
            if len(_model_cache) >= MAX_CACHED_MODELS:
                oldest = next(iter(_model_cache))
                try:
                    del _model_cache[oldest]
                except Exception:
                    pass
            _model_cache[cache_key] = self._model

        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter


    def transcribe(self, audio_file: str) -> Transcript:
        options = {
            "beam_size": self._beam_size,
            "vad_filter": self._vad_filter,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "condition_on_previous_text": False,
        }

        if self._language:
            options["language"] = self._language

        segments, info = self._model.transcribe(audio_file, **options)

        transcript_segments = [
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
            for segment in segments
        ]

        return Transcript(
            language=info.language,
            duration=info.duration,
            segments=transcript_segments,
        )
