"""Conditions in which a separate local LLM is an appropriate fallback."""
import httpx
from openai import APIConnectionError, APIStatusError


def endpoint_unavailable(error: Exception) -> bool:
    # Authentication, model IDs, invalid requests and bad completions need a
    # correction, not a multi-GB download. Gateway/service outages may fall back.
    if isinstance(error, (APIConnectionError, httpx.TransportError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (502, 503, 504)
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in (502, 503, 504)
    return False
