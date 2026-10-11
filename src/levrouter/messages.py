"""Message abstraction layer for framework-agnostic feature extraction.

This module provides converters to transform messages from different frameworks
(LangChain, Anthropic SDK, OpenAI SDK) into a normalized format that core
feature extraction can work with.
"""

from __future__ import annotations

from typing import Any, Protocol


class MessageLike(Protocol):
    """Protocol for anything that looks like a message to levrouter."""

    role: str  # "user", "assistant", "system", "tool"
    content: str | list  # text or list of content blocks
    tool_calls: list[Any] | None


def detect_attachments(content: str | list) -> dict[str, int]:
    """Detect attachment types and counts from any content structure.

    Returns: {"image": n, "file": n, "audio": n, "video": n}
    """
    from .features import ATTACHMENT_KINDS, block_kind

    attachments = {kind: 0 for kind in ATTACHMENT_KINDS}

    if isinstance(content, str):
        return attachments

    if not isinstance(content, list):
        return attachments

    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block_kind(block)
        if kind:
            attachments[kind] += 1

    return attachments


def messages_from_langchain(messages: list[Any]) -> list[dict]:
    """Convert LangChain BaseMessage list to normalized dicts.

    Args:
        messages: List of langchain_core.messages.BaseMessage

    Returns:
        List of normalized dicts with role, content, tool_calls
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    normalized = []

    for msg in messages:
        entry: dict[str, Any] = {"role": "user", "content": ""}

        if isinstance(msg, SystemMessage):
            entry["role"] = "system"
        elif isinstance(msg, HumanMessage):
            entry["role"] = "user"
        elif isinstance(msg, AIMessage):
            entry["role"] = "assistant"
        elif isinstance(msg, ToolMessage):
            entry["role"] = "tool"

        entry["content"] = msg.content

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls

        normalized.append(entry)

    return normalized


def messages_from_anthropic(messages: list[dict]) -> list[dict]:
    """Normalize Anthropic SDK messages (mostly pass-through).

    Args:
        messages: List of dicts with role and content

    Returns:
        Normalized list (identity function for Anthropic format)
    """
    return messages


def messages_from_openai(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI SDK messages.

    Args:
        messages: List of OpenAI ChatCompletionMessageParam dicts

    Returns:
        Normalized list with consistent structure
    """
    normalized = []

    for msg in messages:
        entry = {
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        }

        if "tool_calls" in msg:
            entry["tool_calls"] = msg["tool_calls"]

        normalized.append(entry)

    return normalized


def get_converter(framework: str) -> callable:
    """Get message converter for a framework.

    Args:
        framework: "langchain", "anthropic", "openai", etc.

    Returns:
        Callable that converts messages to normalized format
    """
    converters = {
        "langchain": messages_from_langchain,
        "anthropic": messages_from_anthropic,
        "openai": messages_from_openai,
    }

    converter = converters.get(framework.lower())
    if not converter:
        raise ValueError(f"Unknown framework: {framework}. Supported: {list(converters.keys())}")

    return converter
