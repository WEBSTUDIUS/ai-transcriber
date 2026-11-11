import httpx

from ai_transcriber.config import DEFAULT_LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_TIMEOUT_SECONDS
from ai_transcriber.doctor.base import CheckResult

CHECK_NAME = "LM Studio"
TIMEOUT = DEFAULT_LMSTUDIO_TIMEOUT_SECONDS


def _endpoint() -> str:
    return f"{DEFAULT_LMSTUDIO_BASE_URL.rstrip('/')}/models"


def _success(message: str) -> CheckResult:
    return CheckResult(
        name=CHECK_NAME,
        ok=True,
        message=message,
    )


def _fail(message: str) -> CheckResult:
    return CheckResult(
        name=CHECK_NAME,
        ok=False,
        message=message,
    )


def check_lmstudio() -> CheckResult:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(_endpoint())
            response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            return _fail("Invalid API response")

        models = payload.get("data")
        if not isinstance(models, list) or not models:
            return _fail("No model loaded")

        model_name = models[0].get("id")
        if not isinstance(model_name, str):
            return _fail("Invalid model information")

        return _success(model_name)

    except httpx.ConnectError:
        return _fail("LM Studio is not running")

    except httpx.TimeoutException:
        return _fail("Connection timeout")

    except httpx.HTTPStatusError as e:
        return _fail(f"HTTP {e.response.status_code}")

    except (KeyError, TypeError, ValueError):
        return _fail("Invalid API response")
