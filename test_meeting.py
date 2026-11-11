from pathlib import Path

from ai_transcriber.llm.client import (
    LMStudioClient,
)


INPUT = Path(
    "res_large.txt"
)

OUTPUT = Path(
    "examples/meeting_summary.md"
)


def main():

    text = INPUT.read_text(
        encoding="utf-8"
    )


    client = LMStudioClient()


    summary = client.summarize(
        text,
        prompt_type="meeting"
    )


    OUTPUT.write_text(
        summary,
        encoding="utf-8"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
