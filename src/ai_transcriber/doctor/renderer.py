from .base import CheckResult


def render(results: list[CheckResult]) -> None:
    print()
    print("AI Transcriber Doctor")
    print()

    for result in results:
        icon = "✓" if result.ok else "✗"

        print(
            f"{icon} {result.name}: {result.message}"
        )
