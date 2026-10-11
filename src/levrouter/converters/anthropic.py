"""Anthropic SDK message converter - extracts RequestFeatures from dict messages."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..features import RequestFeatures


def _content(content: str | list) -> tuple[str, Counter, int]:
    """Text, attachment counts and PDF count from Anthropic message content."""
    from ..features import ATTACHMENT_KINDS, block_kind, _mime_of

    if isinstance(content, str):
        return content, Counter(), 0

    texts, kinds, pdfs = [], Counter(), 0

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                else:
                    kind = block_kind(block)
                    if kind:
                        kinds[kind] += 1
                        if kind == "file" and (_mime_of(block) or "").lower() == "application/pdf":
                            pdfs += 1

    return "\n".join(texts), kinds, pdfs


def extract(
    messages: list[dict[str, Any]], tools: list | None = None, tool_choice: object = None
) -> RequestFeatures:
    """Extract RequestFeatures from Anthropic SDK dict messages.

    Args:
        messages: List of dicts with role and content
        tools: Optional list of tools
        tool_choice: Optional tool choice specification

    Returns:
        RequestFeatures with extracted information
    """
    from ..features import is_forced_tool_choice

    system, last_user, user_texts = [], "", []
    attachments: Counter = Counter()
    turns = tool_results = total_chars = pdfs = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        text, kinds, n_pdf = _content(content)
        total_chars += len(text)
        attachments += kinds
        pdfs += n_pdf

        if role == "system":
            system.append(text)
        elif role == "user":
            last_user = text
            user_texts.append(text)
            turns += 1
        elif role == "assistant":
            turns += 1
        elif role == "tool":
            tool_results += 1

    # The human turn being answered is not a "prior" turn.
    return RequestFeatures(
        system="\n".join(system),
        last_user=last_user,
        turns=max(turns - 1, 0),
        tool_count=len(tools or []),
        tool_results=tool_results,
        attachments=dict(attachments),
        total_chars=total_chars,
        pdfs=pdfs,
        forced_tool=bool(tools) and is_forced_tool_choice(tool_choice),
        earlier_user=tuple(t for t in user_texts[:-1] if t.strip())[-10:],
    )
