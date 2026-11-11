from __future__ import annotations

import gc
from typing import Any

import torch
import whisperx

from ai_transcriber.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_TORCH_DEVICE,
    HF_TOKEN,
)

from ai_transcriber.transcriber.models import (
    Transcript,
    TranscriptSegment,
)

from ai_transcriber.transcriber.diarizer import Diarizer


_whisperx_model_cache: dict[str, Any] = {}
_align_model_cache: dict[str, tuple[Any, Any]] = {}

MAX_CACHED_WHISPERX_MODELS = 2
MAX_CACHED_ALIGN_MODELS = 3


def _evict_oldest(cache: dict, max_size: int) -> None:
    while len(cache) >= max_size:
        try:
            cache.pop(next(iter(cache)))
        except (StopIteration, KeyError):
            break


def clear_whisperx_caches() -> None:
    """Evict cached WhisperX ASR/alignment models."""
    global _whisperx_model_cache, _align_model_cache
    for model in list(_whisperx_model_cache.values()):
        try:
            del model
        except Exception:
            pass
    for pair in list(_align_model_cache.values()):
        try:
            del pair
        except Exception:
            pass
    _whisperx_model_cache = {}
    _align_model_cache = {}


class WhisperXTranscriber:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        asr_device: str = DEFAULT_DEVICE,
        torch_device: str = DEFAULT_TORCH_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str | None = DEFAULT_LANGUAGE,
        hf_token: str | None = None,
        enable_diarization: bool = True,
    ) -> None:

        self._model_name = model_name
        self._asr_device = asr_device
        self._torch_device = torch_device
        self._compute_type = compute_type
        self._language = language
        self._enable_diarization = enable_diarization
        self._hf_token = hf_token or HF_TOKEN

        self._align_model = None
        self._align_metadata = None
        self._align_language = None
        self._diarizer = None

        #
        # MPS (Apple Silicon) работает на unified memory —
        # она ограничена, поэтому на MPS держим только одну
        # модель за раз: не кешируем и освобождаем модель
        # сразу после её этапа.
        #
        self._mps_mode = torch_device == "mps"

        #
        # 1. WhisperX ASR model
        #
        cache_key = (
            f"{model_name}:"
            f"{asr_device}:"
            f"{compute_type}:{language or 'auto'}"
        )

        #
        # CPU / CUDA: используем глобальный cache.
        # MPS: не кешируем, чтобы не удерживать память.
        #
        if (
            not self._mps_mode
            and cache_key in _whisperx_model_cache
        ):

            self._model = _whisperx_model_cache[cache_key]

            print(
                f"Using cached WhisperX model: "
                f"{model_name}"
            )

        else:

            print(
                f"Loading WhisperX model: "
                f"{model_name} "
                f"on {asr_device} "
                f"({compute_type})"
            )

            self._model = whisperx.load_model(
                model_name,
                device=asr_device,
                compute_type=compute_type,
                language=language,
            )

            if not self._mps_mode:
                _evict_oldest(_whisperx_model_cache, MAX_CACHED_WHISPERX_MODELS)
                _whisperx_model_cache[cache_key] = self._model

        #
        # 2. Speaker diarization
        #
        # Загружается лениво — только когда реально нужна,
        # и освобождается сразу после использования (MPS).
        # Diarization model самая тяжёлая, поэтому её нельзя
        # держать в памяти вместе с остальными моделями.
        #

    def _get_align_model(
        self,
        language: str,
    ):
        """
        Load alignment model.

        MPS:
            Не кешируем.

        CPU / CUDA:
            Кешируем глобально.
        """

        cache_key = (
            f"{language}:"
            f"{self._torch_device}"
        )

        #
        # CPU / CUDA cache
        #
        if self._torch_device != "mps":

            if cache_key in _align_model_cache:

                (
                    self._align_model,
                    self._align_metadata,
                ) = _align_model_cache[cache_key]

                self._align_language = language

                return (
                    self._align_model,
                    self._align_metadata,
                )

        print(
            f"Loading alignment model: "
            f"{language} "
            f"on {self._torch_device}"
        )

        (
            self._align_model,
            self._align_metadata,
        ) = whisperx.load_align_model(
            language_code=language,
            device=self._torch_device,
        )

        self._align_language = language

        #
        # Не кешируем MPS alignment model.
        #
        if self._torch_device != "mps":

            _evict_oldest(_align_model_cache, MAX_CACHED_ALIGN_MODELS)
            _align_model_cache[cache_key] = (
                self._align_model,
                self._align_metadata,
            )

        return (
            self._align_model,
            self._align_metadata,
        )

    def _get_diarizer(self):
        """
        Лениво загружает diarization model.

        Diarization — самый тяжёлый этап, поэтому модель
        создаётся только перед использованием и сразу
        освобождается после (на MPS).
        """

        if (
            self._diarizer is None
            and self._hf_token
            and self._enable_diarization
        ):

            self._diarizer = Diarizer(
                device=self._torch_device,
                token=self._hf_token,
            )

        return self._diarizer

    def _release(self, attr: str) -> None:
        """
        Сбрасывает ссылку на модель и возвращает память.

        gc.collect() + empty_cache обязательны:
        иначе аллокатор не вернёт кэшированные блоки.
        """

        setattr(self, attr, None)

        gc.collect()

        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def transcribe(
        self,
        audio_file: str,
    ) -> Transcript:

        audio = whisperx.load_audio(
            audio_file
        )

        duration = len(audio) / 16000

        try:

            #
            # 1. Whisper transcription
            #
            result = self._model.transcribe(
                audio
            )

            language = result["language"]

            #
            # ASR-модель больше не нужна — освобождаем её
            # до загрузки alignment/diarization (MPS).
            #
            if self._mps_mode:
                self._release("_model")

            #
            # 2. Word-level alignment
            #
            model_a, metadata = self._get_align_model(
                language
            )

            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self._torch_device,
            )

            #
            # Alignment-модель больше не нужна (MPS).
            #
            if self._mps_mode:
                del model_a
                self._release("_align_model")
                self._align_metadata = None
                self._align_language = None

            #
            # 3. Speaker diarization
            #
            diarizer = self._get_diarizer()

            if diarizer is not None:

                print(
                    f"Running diarization "
                    f"on {self._torch_device}..."
                )

                diarize_segments = diarizer.diarize(
                    audio
                )

                result = whisperx.assign_word_speakers(
                    diarize_segments,
                    result,
                )

                #
                # Diarization-модель больше не нужна для этого файла:
                # освобождаем сразу, чтобы отдать память следующим этапам
                # (alignment уже выгружен на MPS, дальше — LLM/export).
                # Новый job создаст её лениво заново при необходимости.
                #
                del diarizer
                self._release("_diarizer")

            #
            # 4. Convert to domain model
            #
            segments = []

            for segment in result.get(
                "segments",
                [],
            ):

                segments.append(
                    TranscriptSegment(
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"].strip(),
                        speaker=segment.get(
                            "speaker"
                        ),
                        words=segment.get(
                            "words",
                            [],
                        ),
                    )
                )

            return Transcript(
                language=language,
                duration=duration,
                segments=segments,
            )

        finally:

            #
            # Удаляем большие временные объекты.
            #
            del audio

            if "result" in locals():
                del result

            if "model_a" in locals():
                del model_a

            if "metadata" in locals():
                del metadata

            if "diarize_segments" in locals():
                del diarize_segments

            gc.collect()

            #
            # Для MPS очищаем allocator cache.
            #
            if (
                self._torch_device == "mps"
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()
