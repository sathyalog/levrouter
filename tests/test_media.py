"""Attachments in every provider's block format, and routing that respects each model's input types."""

import pytest
from conftest import SIMPLE, FixedClassifier
from langchain_core.messages import HumanMessage

from levrouter import Router, RouterConfig
from levrouter.features import block_kind, extract

B64 = "iVBORw0KGgo="


def msg(*blocks, text="please look at this"):
    return HumanMessage(content=[{"type": "text", "text": text}, *blocks])


@pytest.mark.parametrize(
    "block, kind",
    [
        ({"type": "image", "base64": B64, "mime_type": "image/png"}, "image"),                       # LangChain standard
        ({"type": "image_url", "image_url": {"url": "https://x/y.png"}}, "image"),                    # OpenAI
        ({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64}}, "image"),  # Anthropic
        ({"type": "audio", "base64": B64, "mime_type": "audio/wav"}, "audio"),                       # LangChain standard
        ({"type": "input_audio", "input_audio": {"data": B64, "format": "wav"}}, "audio"),            # OpenAI
        ({"type": "video", "url": "https://x/clip.mp4", "mime_type": "video/mp4"}, "video"),          # LangChain standard
        ({"type": "file", "base64": B64, "mime_type": "application/pdf"}, "file"),                   # LangChain standard
        ({"type": "file", "file": {"filename": "report.pdf", "file_data": B64}}, "file"),            # OpenAI
        ({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": B64}}, "file"),  # Anthropic
        ({"type": "media", "mime_type": "audio/mp3", "data": B64}, "audio"),                         # Gemini
        ({"type": "media", "mime_type": "video/mp4", "file_uri": "gs://x"}, "video"),                # Gemini
        ({"type": "file", "base64": B64, "mime_type": "image/jpeg"}, "image"),                       # file carrying an image
        ({"type": "text", "text": "hi"}, None),
        ({"type": "tool_use", "id": "1"}, None),
    ],
)
def test_block_kinds(block, kind):
    assert block_kind(block) == kind


def test_attachments_counted_and_described():
    f = extract([msg(
        {"type": "file", "base64": B64, "mime_type": "application/pdf"},
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
        {"type": "image_url", "image_url": {"url": "https://x/b.png"}},
        {"type": "input_audio", "input_audio": {"data": B64, "format": "wav"}},
    )])
    assert f.attachments == {"file": 1, "image": 2, "audio": 1}
    assert f.has_images and f.pdfs == 1
    assert "Attachments: 2 images, 1 PDF, 1 audio clip" in f.state()


def _route(provider, configured, *blocks):
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))
    return router.decide(provider, configured, extract([msg(*blocks)]))


AUDIO = {"type": "audio", "base64": B64, "mime_type": "audio/wav"}
VIDEO = {"type": "video", "url": "https://x/clip.mp4", "mime_type": "video/mp4"}
PDF = {"type": "file", "base64": B64, "mime_type": "application/pdf"}
IMAGE = {"type": "image_url", "image_url": {"url": "https://x/a.png"}}


def test_audio_only_routes_where_supported():
    # Gemini accepts audio at every tier: routing proceeds normally.
    assert _route("gemini", "gemini-3.1-pro-preview", AUDIO).model == "gemini-3.1-flash-lite"
    # No Claude tier accepts audio: keep what the developer configured, and say why.
    d = _route("anthropic", "claude-opus-5-5", AUDIO)
    assert d.model == "claude-opus-5-5" and "no audio input" in d.reason


def test_pdf_and_image_capabilities():
    assert _route("anthropic", "claude-opus-5-5", PDF).model == "claude-haiku-4-5"      # Claude reads PDFs
    assert _route("openai", "gpt-6-astra", PDF).model == "gpt-6-luna"
    d = _route("groq", "openai/gpt-oss-120b", PDF)                                      # gpt-oss is text-only
    assert d.model == "openai/gpt-oss-120b" and "kept configured" in d.reason
    # DeepSeek: flash reads images, v4-pro does not -> stays on flash tiers
    assert _route("deepseek", "deepseek-v4-pro", IMAGE).model == "deepseek-flash"


def test_video_moves_up_to_a_capable_tier():
    # Mistral small takes images but not video; no Mistral tier takes video.
    d = _route("mistral", "mistral-medium-latest", VIDEO)
    assert d.model == "mistral-medium-latest" and "no video input" in d.reason
    # Hugging Face Qwen 9B and 35B take video; the 2.4T complex model is text-only.
    assert _route("huggingface", "Qwen/Qwen3.8-2.4T-A95B", VIDEO).model == "Qwen/Qwen3.5-9B"


def test_forced_tool_calls_avoid_models_that_reject_them():
    from levrouter.features import is_forced_tool_choice
    from conftest import COMPLEX

    assert is_forced_tool_choice({"type": "tool", "name": "Profile"}) and is_forced_tool_choice("any")
    assert is_forced_tool_choice({"type": "function", "function": {"name": "x"}}) and is_forced_tool_choice("required")
    assert not is_forced_tool_choice("auto") and not is_forced_tool_choice(None)

    tools = [{"name": "Profile"}]
    forced = extract([HumanMessage("Design a data platform and extract the fields")], tools, {"type": "tool", "name": "Profile"})
    router = Router(RouterConfig(), FixedClassifier(COMPLEX))
    d = router.decide("anthropic", "claude-sonnet-5", forced)
    assert d.model == "claude-sonnet-5" and "rejects forced tool calls" in d.reason     # complex tier (Opus 5.5) skipped
    d = router.decide("openrouter", "anthropic/claude-sonnet-5", forced)
    assert d.model == "anthropic/claude-sonnet-5"
    free_choice = extract([HumanMessage("Design a data platform")], tools, "auto")
    assert router.decide("anthropic", "claude-sonnet-5", free_choice).model == "claude-opus-5-5"
