from dataclasses import dataclass


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    details: str | None = None
