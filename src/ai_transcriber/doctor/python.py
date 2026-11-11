import sys

from .base import CheckResult


def check_python() -> CheckResult:
    version = sys.version.split()[0]

    return CheckResult(
        name="Python",
        ok=True,
        message=version,
    )
