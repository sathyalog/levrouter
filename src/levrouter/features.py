"""Turn a message list into the compact "state" the classifier reads, plus the
request properties the capability guard checks (attachments, tools, size).

Supports multiple frameworks via converters (LangChain, Anthropic SDK, OpenAI SDK).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

ATTACHMENT_KINDS = ("image", "file", "audio", "video")

# Content-block "type" values that carry media, across LangChain's standard blocks and each
# provider's native format. Value None: decide from the block's MIME type.
_BLOCK_KINDS: dict[str, str | None] = {
    "image": "image",          # LangChain standard, Anthropic
    "image_url": "image",      # OpenAI
    "audio": "audio",          # LangChain standard
    "input_audio": "audio",    # OpenAI
    "video": "video",          # LangChain standard
    "file": None,              # LangChain standard / OpenAI: PDF, image, audio... by MIME
    "document": "file",        # Anthropic (PDF, text documents)
    "media": None,             # Google Gemini
}


def _kind_from_mime(mime: str | None) -> str:
    mime = (mime or "").lower()
    for prefix in ("image", "audio", "video"):
        if mime.startswith(prefix + "/"):
            return prefix
    return "file"


def _mime_of(block: dict) -> str | None:
    for key in ("mime_type", "media_type"):
        if block.get(key):
            return block[key]
    for nested in ("source", "file"):
        inner = block.get(nested)
        if isinstance(inner, dict):
            for key in ("mime_type", "media_type"):
                if inner.get(key):
                    return inner[key]
            name = inner.get("filename") or ""
            if name.lower().endswith(".pdf"):
                return "application/pdf"
    return None


def block_kind(block: dict) -> str | None:
    """'image' | 'file' | 'audio' | 'video' for a media block, None for anything else."""
    btype = block.get("type")
    if btype not in _BLOCK_KINDS:
        return None
    kind = _BLOCK_KINDS[btype]
    return kind if kind is not None else _kind_from_mime(_mime_of(block))


@dataclass(frozen=True)
class RequestFeatures:
    system: str
    last_user: str
    turns: int
    tool_count: int
    tool_results: int
    # Media attached anywhere in the conversation, e.g. {"image": 2, "file": 1}.
    attachments: dict[str, int] = field(default_factory=dict)
    # All message text, for the context-window guard.
    total_chars: int = 0
    # Attachments that are PDFs, for a readable state line ("1 PDF").
    pdfs: int = 0
    # The request forces a tool call (with_structured_output, tool_choice="any"/named tool).
    forced_tool: bool = False
    # Earlier user messages in this request (newest last, up to 10), to notice a question asked again.
    earlier_user: tuple[str, ...] = ()

    @property
    def has_images(self) -> bool:
        return self.attachments.get("image", 0) > 0

    def attachment_summary(self) -> str:
        parts = []
        for kind in ATTACHMENT_KINDS:
            n = self.attachments.get(kind, 0)
            if not n:
                continue
            if kind == "file" and self.pdfs:
                parts.append(f"{self.pdfs} PDF" + ("s" if self.pdfs > 1 else ""))
                n -= self.pdfs
                if not n:
                    continue
            label = {"image": "image", "file": "file", "audio": "audio clip", "video": "video"}[kind]
            parts.append(f"{n} {label}" + ("s" if n > 1 else ""))
        return ", ".join(parts)

    def state(self, max_chars: int = 2000) -> str:
        user = self.last_user
        if len(user) > max_chars:
            # Keep both ends: the ask is usually at the start, the constraint at the end.
            half = max_chars // 2
            user = user[:half] + "\n...\n" + user[-half:]
        parts = []
        if self.system:
            parts.append(f"System instructions: {self.system[:400]}")
        parts.append(f"User request: {user}")
        attached = self.attachment_summary()
        if attached:
            parts.append(f"Attachments: {attached}")
        parts.append(
            f"Context: {self.turns} prior turns, {self.tool_count} tools available, "
            f"{self.tool_results} tool results so far"
        )
        return "\n".join(parts)


def is_forced_tool_choice(tool_choice: object) -> bool:
    """True for tool_choice values that force a tool call, in Anthropic or OpenAI form."""
    if tool_choice in (None, "auto", "none", False):
        return False
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") in ("any", "tool", "function")
    return True  # "any", "required", a tool name, True


def extract_generic(
    messages: list[dict],
    tools: list | None = None,
    tool_choice: object = None
) -> RequestFeatures:
    """Generic extraction: works with any normalized message dicts.

    Each dict should have: {"role": "...", "content": "...", "tool_calls": ...?}
    """
    system, last_user, user_texts = [], "", []
    attachments: Counter = Counter()
    turns = tool_results = total_chars = pdfs = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Parse content (same logic for all frameworks)
        if isinstance(content, str):
            text, kinds, n_pdf = content, Counter(), 0
        else:
            text, kinds, n_pdf = _parse_content_blocks(content)

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


def _parse_content_blocks(content: object) -> tuple[str, Counter, int]:
    """Extract text and attachment counts from content blocks."""
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


def extract(messages: list, tools: list | None = None, tool_choice: object = None) -> RequestFeatures:
    """LangChain-specific wrapper. Converts BaseMessage list to generic format."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

    # Convert LangChain messages to normalized dicts
    normalized = []
    for m in messages:
        entry: dict = {"role": "user", "content": ""}
        if isinstance(m, SystemMessage):
            entry["role"] = "system"
        elif isinstance(m, HumanMessage):
            entry["role"] = "user"
        elif isinstance(m, AIMessage):
            entry["role"] = "assistant"
        elif isinstance(m, ToolMessage):
            entry["role"] = "tool"
        entry["content"] = m.content
        if hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        normalized.append(entry)

    return extract_generic(normalized, tools, tool_choice)
