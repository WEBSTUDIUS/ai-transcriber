import subprocess

from ai_transcriber.doctor.base import CheckResult


def check_ffmpeg() -> CheckResult:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )

        first_line = result.stdout.splitlines()[0]
        version = first_line.split()[2]

        return CheckResult(
            name="FFmpeg",
            ok=True,
            message=version,
        )

    except FileNotFoundError:
        return CheckResult(
            name="FFmpeg",
            ok=False,
            message="Not found in PATH",
        )

    except subprocess.CalledProcessError:
        return CheckResult(
            name="FFmpeg",
            ok=False,
            message="Failed to execute",
        )
