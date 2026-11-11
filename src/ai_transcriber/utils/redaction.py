"""Sanitize user-visible diagnostics without exposing credentials or full URLs."""
from __future__ import annotations

import os
import re


def redact_diagnostic(value: object) -> str:
    text = str(value)
    # Also handle errors which print a credential without its URL/prefix.
    secrets = [secret for name, secret in os.environ.items()
               if secret and len(secret) >= 4 and
               (name.startswith('WEBHOOK_') or name in ('HF_TOKEN', 'OPENAI_API_KEY'))]
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, '[secret redacted]')
    text = re.sub(r'https?://[^\s<>"\']+', '[endpoint redacted]', text, flags=re.IGNORECASE)
    text = re.sub(r'/rest/\d+/[^/\s"\']+', '/rest/[redacted]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:hf_|sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}', '[secret redacted]', text)
    text = re.sub(r'(?i)\bBearer\s+[^\s,"\']+', 'Bearer [redacted]', text)
    return text
