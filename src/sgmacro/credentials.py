"""API keys: read from the environment, held in memory, never written.

A key reaches the program only through an environment variable (never a command
line option, which would land in shell history and process listings, and never a
file the program reads). It is sent only in the ``Authorization`` header of an
HTTPS request to the provider it belongs to: a provider's own variable, such as
``DEEPSEEK_API_KEY``, is sent to that provider's preset address and nowhere else,
and a custom address gets only ``SGMACRO_OPENAI_API_KEY``.

Nothing the program writes may contain a key: provider errors and agent console
logs pass through :func:`redact`, saved agent sessions through :func:`redact_keys`,
and reports, manifests and response-cache entries are checked by
:func:`sgmacro.store.assert_no_credentials` before they are written.
"""

from __future__ import annotations

import os
import re

from sgmacro.sources.http import without_query

#: Every variable the program may read a key from.
KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "SGMACRO_OPENAI_API_KEY", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "OPENROUTER_API_KEY")

#: Shapes of common keys and of an echoed authorization header.
KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}|\bAIza[0-9A-Za-z_\-]{30,}|Bearer\s+[A-Za-z0-9._\-]{8,}")

REDACTED = "[key redacted]"


def redact_keys(text: str) -> str:
    """Replace the value of every key in the environment, and anything shaped
    like a key, with a marker. Everything else is left as it was."""
    for name in KEY_ENV_VARS:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            text = text.replace(value, REDACTED)
    return KEY_PATTERN.sub(REDACTED, text)


def redact(text: str) -> str:
    """For error text: keys and query strings (which may carry signed URLs) removed."""
    return redact_keys(without_query(text))
