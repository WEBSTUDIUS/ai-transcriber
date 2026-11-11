from .ffmpeg import check_ffmpeg
from .lmstudio import check_lmstudio
from .python import check_python
from .whisper import check_whisper
from .base import CheckResult


def run_checks() -> list[CheckResult]:
    checks = [
        check_python,
        check_ffmpeg,
        check_whisper,
        check_lmstudio
    ]

    results: list[CheckResult] = []

    for check in checks:
        results.append(check())

    return results
