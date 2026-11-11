import subprocess
from pathlib import Path


def get_audio_duration(
    input_file: str,
    timeout: float = 30,
) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_file,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        # PyAV is already a faster-whisper dependency. It can probe media
        # even when the standalone ffprobe executable is not installed.
        import av
        with av.open(input_file) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
        raise ValueError(f"No duration metadata for: {input_file}")

    raw = (result.stdout or "").strip()
    if not raw:
        raise ValueError(f"ffprobe returned no duration for: {input_file}")

    return float(raw)

def normalize_audio(
    input_file: str,
    output_file: str,
    timeout: float = 600,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-af",
            "loudnorm",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_file,
        ],
        check=True,
        capture_output=True,
        timeout=timeout,
    )
