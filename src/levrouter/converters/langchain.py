"""LangChain message converter - extracts RequestFeatures from BaseMessage list."""

from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..features import RequestFeatures


def _content(content: object) -> tuple[str, Counter, int]:
    """Text, attachment counts and PDF count of one message's content."""
    from ..features import ATTACHMENT_KINDS, block_kind, _mime_of

    if isinstance(content, str):
        return content, Counter(), 0
    texts, kinds, pdfs = [], Counter(), 0
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
                    continue
                kind = block_kind(block)
                if kind:
                    kinds[kind] += 1
                    if kind == "file" and (_mime_of(block) or "").lower() == "application/pdf":
                        pdfs += 1
    return "\n".join(texts), kinds, pdfs


def extract(messages: list[BaseMessage], tools: list | None = None, tool_choice: object = None) -> RequestFeatures:
    """Extract RequestFeatures from LangChain messages.

    Args:
        messages: List of langchain_core.messages.BaseMessage
        tools: Optional list of tools
        tool_choice: Optional tool choice specification

    Returns:
        RequestFeatures with extracted information
    """
    from ..features import is_forced_tool_choice

    system, last_user, user_texts = [], "", []
    attachments: Counter = Counter()
    turns = tool_results = total_chars = pdfs = 0

    for m in messages:
        text, kinds, n_pdf = _content(m.content)
        total_chars += len(text)
        attachments += kinds
        pdfs += n_pdf

        if isinstance(m, SystemMessage):
            system.append(text)
        elif isinstance(m, HumanMessage):
            last_user = text
            user_texts.append(text)
            turns += 1
        elif isinstance(m, ToolMessage):
            tool_results += 1
        elif isinstance(m, AIMessage):
            turns += 1

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
