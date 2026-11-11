import json
import base64
import os
import re
from pathlib import Path

import requests

# Webhook по умолчанию берётся из окружения.
# Хардкодить продовый токен в коде нельзя — иначе любой вызов
# без явного URL будет слать данные в прод.
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "") or os.getenv("WEBHOOK_IT_TASK", "")

# Portal-specific schema is private configuration, never source code.
ENTITY_TYPE_ID = int(os.getenv("BITRIX_ENTITY_TYPE_ID", "0"))
CATEGORY_ID = int(os.getenv("BITRIX_CATEGORY_ID", "0"))
STAGE_ID = os.getenv("BITRIX_STAGE_ID", "")



def validate_webhook_url(url: str) -> str:
    from urllib.parse import urlsplit
    try:
        parsed = urlsplit(url)
        parsed.port
    except ValueError:
        raise ValueError("Invalid webhook URL") from None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Webhook URL must be http(s), without credentials, query or fragment")
    return url.rstrip("/") + "/"


def markdown_to_plain_text(markdown_text: str) -> str:
    text = markdown_text

    # Заголовки
    text = re.sub(
        r"^#{1,6}\s*(.+)$",
        r"\n\1\n",
        text,
        flags=re.MULTILINE,
    )

    # Жирный текст
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"\1",
        text,
    )

    # Курсив
    text = re.sub(
        r"\*(.+?)\*",
        r"\1",
        text,
    )

    # Ненумерованные списки
    text = re.sub(
        r"^\s*[-*]\s+",
        "• ",
        text,
        flags=re.MULTILINE,
    )

    # Нумерованные списки оставляем как есть

    # Убираем лишние пустые строки
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def create_smart_process_item(
        webhook_url: str | None = None,
        title: str = "",
        summary_file: str = "",
        file: str = "",
        ids: list[int] | None = None,
        executor: int = 1
):
    # Resolve at call time: CLI imports this module before loading .env.
    webhook_url = webhook_url or os.getenv("WEBHOOK_URL") or os.getenv("WEBHOOK_IT_TASK") or WEBHOOK_URL
    entity_type_id = int(os.getenv("BITRIX_ENTITY_TYPE_ID", str(ENTITY_TYPE_ID)))
    category_id = int(os.getenv("BITRIX_CATEGORY_ID", str(CATEGORY_ID)))
    stage_id = os.getenv("BITRIX_STAGE_ID", STAGE_ID)
    if not webhook_url:
        raise ValueError(
            "Bitrix webhook URL is not set. "
            "Pass webhook_url explicitly or set WEBHOOK_URL / WEBHOOK_IT_TASK in .env."
        )
    if ids is None:
        ids = json.loads(os.getenv("BITRIX_ASSIGNEE_IDS", "[]"))
    if not entity_type_id:
        raise ValueError("Set BITRIX_ENTITY_TYPE_ID and portal field mapping in .env")
    webhook_url = validate_webhook_url(webhook_url)
    # 1. Читаем Markdown-файл с саммари
    summary_markdown = Path(summary_file).read_text(
        encoding="utf-8"
    )

    # 2. Конвертируем Markdown -> в читаемый ))
    summary = markdown_to_plain_text(
        summary_markdown
    )

    # Validate the attachment before creating an external item.
    attachment = Path(file)
    if not attachment.is_file():
        raise FileNotFoundError("Transcript file not found")
    if attachment.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Transcript file exceeds 50 MB")

    # 3. Создаём элемент смарт-процесса
    item_response = requests.post(
        f"{webhook_url}crm.item.add.json",
        json={
            "entityTypeId": entity_type_id,
            "useOriginalUfNames": "Y",
            "fields": {
                "title": title,
                "categoryId": category_id,
                **({"stageId": stage_id} if stage_id else {}),
                **{
                    key: ({"summary": summary, "assignees": ids, "executor": executor}.get(value, value)
                          if isinstance(value, str) else value)
                    for key, value in json.loads(os.getenv("BITRIX_FIELDS_JSON", "{}")).items()
                },
            },
        },
        timeout=30,
        allow_redirects=False,
    )

    item_response.raise_for_status()
    item_result = item_response.json()

    if "error" in item_result:
        raise RuntimeError(
            f"Ошибка создания элемента: {item_result.get('error', 'unknown')}"
        )

    item_id = item_result["result"]["item"]["id"]

    print(f"Элемент создан: ID = {item_id}")

    # 4. Подготавливаем файл для комментария
    def prepare_file(file_path: str):
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        # Ограничение 50 МБ: Bitrix timeline не переваривает огромные файлы,
        # а base64 раздувает память в ~1.37 раза.
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(
                f"Transcript file too large for webhook: {path.name} "
                f"({path.stat().st_size / 1024 / 1024:.1f} MB > 50 MB)"
            )

        with path.open("rb") as file:
            content = base64.b64encode(
                file.read()
            ).decode("utf-8")

        return [
            path.name,
            content,
        ]

    files = [
        prepare_file(file),
    ]

    # 5. Добавляем комментарий с файлом
    comment_response = requests.post(
        f"{webhook_url}crm.timeline.comment.add.json",
        json={
            "fields": {
                "ENTITY_ID": item_id,
                "ENTITY_TYPE_ID": entity_type_id,
                "COMMENT": "Прикрепленный файл расшифровки",
                "FILES": files,
            },
        },
        timeout=60,
        allow_redirects=False,
    )

    print("HTTP status:", comment_response.status_code)

    comment_response.raise_for_status()
    comment_result = comment_response.json()

    if "error" in comment_result:
        raise RuntimeError(
            "Элемент создан, но не удалось добавить файл: "
            f"{comment_result.get('error', 'unknown')}"
        )

    print("Файл успешно прикреплен")

    return {
        "item_id": item_id,
        "comment": comment_result,
    }


if __name__ == "__main__":
    import os as _os

    _example_webhook = _os.getenv("WEBHOOK_URL", "")
    if not _example_webhook:
        print("Set WEBHOOK_URL env var to run the example.")
    else:
        result = create_smart_process_item(
            webhook_url=_example_webhook,
            title="Example summary",
            summary_file="SUMMARY_EXAMPLE.md",
            file="transcript.txt",
            ids=None
        )

        print(result)
