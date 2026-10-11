"""Route Anthropic SDK calls: client.messages.create / stream / parse, sync and async."""

from __future__ import annotations

import logging

from ..converters.anthropic import extract
from ._sdk_common import SdkPatches

log = logging.getLogger("levrouter")

_patches = SdkPatches(allowed=frozenset({"anthropic"}))


def install(providers: tuple[str, ...]) -> list[str]:
    """Patch the Anthropic SDK's Messages resources. Returns the providers routed through it."""
    if "anthropic" not in providers:
        return []
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        log.debug("levrouter: anthropic SDK not installed; skipping")
        return []
    _patches.enabled.add("anthropic")
    _patches.patch((Messages, AsyncMessages), ("create", "stream", "parse"), extract)
    return ["anthropic"]


def uninstall() -> None:
    _patches.restore()
