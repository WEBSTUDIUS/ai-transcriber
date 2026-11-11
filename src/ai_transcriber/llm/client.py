import os
from typing import Callable

import httpx
from openai import OpenAI

from ai_transcriber.config import DEFAULT_LMSTUDIO_BASE_URL, LLAMA_CTX
from ai_transcriber.llm.prompts import (
    CALL_SUMMARY_PROMPT,
    CHUNK_SUMMARY_PROMPT,
    MEETING_SUMMARY_PROMPT,
)

# Conservative budget for long transcripts: keep every prompt comfortably
# inside the context window so the generated summary is never cut off.
CHARS_PER_TOKEN = 2.5
RESERVED_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2000"))
CHUNK_OUTPUT_TOKENS = 1000
TEMPLATE_OVERHEAD_TOKENS = 300


def split_transcript(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars``, preferring line boundaries."""
    if max_chars <= 0:
        raise ValueError("Chunk size must be positive")
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.split("\n"):
        while len(line) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]

        if current and current_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


class LMStudioClient:

    def __init__(
            self,
            base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
            model: str | None = None,
            api_key: str | None = None,
            context_tokens: int = LLAMA_CTX,
    ):

        self.context_tokens = context_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "lm-studio")
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )

        self.model = model

    def list_models(self, timeout: float = 3) -> list[str]:
        """
        List model ids served by the OpenAI-compatible endpoint.

        Raises on connection / HTTP errors so callers can detect
        that the LLM server is not reachable.
        """
        url = f"{str(self.client.base_url).rstrip('/')}/models"
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
        payload = response.json()
        return [
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    def summarize(
            self,
            text: str,
            prompt_type: str,
            custom_prompt: str | None = None,
    ) -> str:

        if self.model is None:
            models = self.client.models.list()
            if not getattr(models, "data", None):
                raise ValueError("No models available from LM Studio")
            self.model = models.data[0].id

        if custom_prompt:
            prompt = custom_prompt.replace("{text}", text)
        elif prompt_type == "chunk":
            prompt = CHUNK_SUMMARY_PROMPT.format(text=text)
        elif prompt_type == "call":
            prompt = CALL_SUMMARY_PROMPT.format(text=text)
        else:
            prompt = MEETING_SUMMARY_PROMPT.format(text=text)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=CHUNK_OUTPUT_TOKENS if prompt_type == "chunk" else RESERVED_OUTPUT_TOKENS,
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты анализируешь рабочие встречи."
                    )
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("Summary reached the output token limit; use a shorter transcript or larger output budget")
        content = choice.message.content
        if not content or not content.strip():
            raise ValueError("LLM returned an empty summary")
        return content

    def summarize_long(
        self,
        text: str,
        prompt_type: str,
        custom_prompt: str | None = None,
        ctx_tokens: int | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> str:
        """
        Summarize a transcript of any length.

        Short texts are summarized in a single call. Long texts are split
        into chunks that fit the context window, each chunk is condensed,
        then the condensations are merged recursively and finally turned
        into the requested report.
        """
        ctx = ctx_tokens or self.context_tokens
        budget_tokens = ctx - RESERVED_OUTPUT_TOKENS - TEMPLATE_OVERHEAD_TOKENS
        budget_chars = int(max(budget_tokens, 1000) * CHARS_PER_TOKEN)

        if len(text) <= budget_chars:
            return self.summarize(
                text,
                prompt_type=prompt_type,
                custom_prompt=custom_prompt,
            )

        chunks = split_transcript(text, budget_chars)
        total = len(chunks)
        if on_progress:
            on_progress(0, total, "map")

        summaries: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress(index, total, "map")
            summaries.append(self.summarize(chunk, prompt_type="chunk"))

        est_summary_chars = int(CHUNK_OUTPUT_TOKENS * CHARS_PER_TOKEN)
        # group_size минимум 2: при group_size=1 reduce не уменьшает
        # число саммари и while len>1 крутится вечно.
        group_size = max(2, budget_chars // (2 * est_summary_chars))

        while len(summaries) > 1:
            groups = [
                summaries[i:i + group_size]
                for i in range(0, len(summaries), group_size)
            ]
            # Если группировка не уменьшает число элементов — выходим,
            # иначе бесконечный цикл на маленьком ctx.
            if len(groups) >= len(summaries):
                break
            merged: list[str] = []
            for index, group in enumerate(groups, start=1):
                if on_progress:
                    on_progress(index, len(groups), "reduce")
                combined = "\n\n".join(group)
                merged.append(self.summarize(combined, prompt_type="chunk"))
            # Страховка: если merge не уменьшил число — выходим.
            if len(merged) >= len(summaries):
                summaries = merged
                break
            summaries = merged

        if on_progress:
            on_progress(1, 1, "final")
        return self.summarize(
            "\n\n".join(summaries),
            prompt_type=prompt_type,
            custom_prompt=custom_prompt,
        )
