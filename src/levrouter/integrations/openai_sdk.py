"""Route OpenAI SDK calls: client.chat.completions.create / stream / parse, sync and async.

The OpenAI SDK is also the usual client for OpenAI-compatible APIs, so calls are routed for
whichever known provider the client's base_url points at (OpenAI, OpenRouter, Groq, DeepSeek,
xAI, Mistral, Gemini), always within that provider's own models.
"""

from __future__ import annotations

import logging

from ..converters.openai import extract
from ._sdk_common import SdkPatches

log = logging.getLogger("levrouter")

_patches = SdkPatches(allowed=frozenset({"openai", "openrouter", "groq", "deepseek", "xai", "mistral", "gemini"}))


def install(providers: tuple[str, ...]) -> list[str]:
    """Patch the OpenAI SDK's chat completions resources. Returns the providers routed through it."""
    wanted = [p for p in providers if p in _patches.allowed]
    if not wanted:
        return []
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        log.debug("levrouter: openai SDK not installed; skipping")
        return []
    _patches.enabled.update(wanted)
    _patches.patch((Completions, AsyncCompletions), ("create", "stream", "parse"), extract)
    return wanted


def uninstall() -> None:
    _patches.restore()
