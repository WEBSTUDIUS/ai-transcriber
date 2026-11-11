from importlib.metadata import PackageNotFoundError, version

from ai_transcriber.doctor.base import CheckResult


def check_whisper() -> CheckResult:
    try:
        package_version = version("faster-whisper")

        return CheckResult(
            name="Faster-Whisper",
            ok=True,
            message=package_version,
        )

    except PackageNotFoundError:
        return CheckResult(
            name="Faster-Whisper",
            ok=False,
            message="Package is not installed",
        )
