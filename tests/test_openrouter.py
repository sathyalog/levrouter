"""OpenRouter (family by default, ladder opt-in), its guardrails, and dated model ids."""

import logging

import pytest
from conftest import COMPLEX, SIMPLE, FixedClassifier
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

import levrouter
from levrouter import Router, RouterConfig
from levrouter.features import extract
from levrouter.providers import canonical_id, price_of, spec_for
from levrouter.telemetry import cost_usd

KEY = "sk-or-v1-TESTKEY000000000000000000"


def _fake(self, messages, stop=None, run_manager=None, **kw):
    msg = AIMessage(content=f"model={self.model_name}", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def fake_network(monkeypatch):
    for cls in (ChatOpenRouter, ChatOpenAI):
        monkeypatch.setattr(cls, "_generate", _fake)


def _enable(sig=SIMPLE, **cfg):
    levrouter.enable(router=Router(RouterConfig(log_level=None, summary_at_exit=False, **cfg), FixedClassifier(sig)))


def _decide(model, sig=SIMPLE, text="hi", blocks=(), **cfg):
    router = Router(RouterConfig(**cfg), FixedClassifier(sig))
    content = [{"type": "text", "text": text}, *blocks] if blocks else text
    return router.decide("openrouter", model, extract([HumanMessage(content=content)]))


# -- family mode (default) -------------------------------------------------------


@pytest.mark.parametrize("configured, simple, complex_", [
    ("anthropic/claude-opus-5.5", "anthropic/claude-haiku-4.5", "anthropic/claude-opus-5.5"),
    ("openai/gpt-6-astra", "openai/gpt-6-luna", "openai/gpt-6-astra"),
    ("google/gemini-3.1-pro-preview", "google/gemini-3.1-flash-lite", "google/gemini-3.1-pro-preview"),
    ("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro"),
])
def test_family_stays_with_vendor(configured, simple, complex_):
    assert _decide(configured, SIMPLE).model == simple
    assert _decide(configured, COMPLEX).model == complex_


def test_chatopenrouter_is_routed_and_keeps_key(fake_network):
    _enable(SIMPLE)
    llm = ChatOpenRouter(model="anthropic/claude-opus-5.5", api_key=KEY)
    out = llm.invoke("What is 2+2?")
    assert out.content == "model=anthropic/claude-haiku-4.5"
    assert out.response_metadata["levrouter"]["provider"] == "openrouter"
    assert llm.model_name == "anthropic/claude-opus-5.5"


def test_chatopenai_on_openrouter_endpoint(fake_network):
    _enable(SIMPLE)
    llm = ChatOpenAI(model="openai/gpt-6-astra", api_key=KEY, base_url="https://openrouter.ai/api/v1")
    assert llm.invoke("hi").content == "model=openai/gpt-6-luna"


@pytest.mark.parametrize("model", ["openrouter/free", "openrouter/auto", "meta-llama/llama-3.3-70b-instruct:free",
                                   "~anthropic/claude-sonnet-latest", "anthropic/claude-opus-5.5:batch"])
def test_free_meta_and_alias_models_never_routed(fake_network, caplog, model):
    _enable(SIMPLE)
    with caplog.at_level(logging.INFO, logger="levrouter"):
        assert ChatOpenRouter(model=model, api_key=KEY).invoke("hi").content == f"model={model}"
    assert any("never_route" in r.getMessage() for r in caplog.records)


def test_unknown_vendor_is_skipped(fake_network, caplog):
    _enable(SIMPLE)
    with caplog.at_level(logging.INFO, logger="levrouter"):
        out = ChatOpenRouter(model="cohere/command-a", api_key=KEY).invoke("hi")
    assert out.content == "model=cohere/command-a"
    assert any("no OpenRouter family for vendor 'cohere'" in r.getMessage() for r in caplog.records)


def test_family_capability_uses_catalog_modalities():
    audio = {"type": "audio", "base64": "AA==", "mime_type": "audio/wav"}
    d = _decide("anthropic/claude-opus-5.5", SIMPLE, blocks=[audio])
    assert d.model == "anthropic/claude-opus-5.5" and "no audio input" in d.reason   # Claude takes no audio
    assert _decide("google/gemini-3.1-pro-preview", SIMPLE, blocks=[audio]).model == "google/gemini-3.1-flash-lite"


# -- ladder mode (opt-in) --------------------------------------------------------


def test_ladder_picks_cheapest_allowed_vendor():
    d = _decide("anthropic/claude-opus-5.5", SIMPLE, openrouter_mode="ladder",
                openrouter_vendors=("anthropic", "openai", "google"))
    assert d.model == "openai/gpt-6-luna"          # $0.10/$0.50 beats haiku and flash-lite
    assert "ladder: anthropic -> openai" in d.reason


def test_ladder_respects_allowlist_and_capabilities():
    d = _decide("anthropic/claude-opus-5.5", SIMPLE, openrouter_mode="ladder", openrouter_vendors=("anthropic",))
    assert d.model == "anthropic/claude-haiku-4.5"
    video = {"type": "video", "url": "https://x/v.mp4", "mime_type": "video/mp4"}
    d = _decide("anthropic/claude-opus-5.5", SIMPLE, blocks=[video], openrouter_mode="ladder",
                openrouter_vendors=("anthropic", "openai", "google"))
    assert d.model == "google/gemini-3.1-flash-lite"   # only Gemini takes video


def test_dated_ids_resolve_prices_and_capabilities():
    assert canonical_id("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert canonical_id("claude-opus-4-5@20251101") == "claude-opus-4-5"
    assert spec_for("anthropic", "claude-haiku-4-5-20251001").context == 200_000
    assert price_of("claude-sonnet-4-5-20250929") == (3.0, 15.0)
    assert cost_usd("anthropic/claude-opus-5.5", 1_000_000, 0) == 4.0


def test_pinned_dated_id_is_kept_when_tier_is_the_same_model():
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))
    d = router.decide("anthropic", "claude-haiku-4-5-20251001", extract([HumanMessage("hi")]))
    assert d.model == "claude-haiku-4-5-20251001"


def test_bare_openrouter_names_resolve_to_their_vendor():
    from levrouter.providers import openrouter_full_id, openrouter_vendor
    assert openrouter_full_id("gpt-4o-mini") == "openai/gpt-4o-mini" and openrouter_vendor("gpt-4o-mini") == "openai"
    assert openrouter_vendor("no-such-model") == "no-such-model"
    assert _decide("gpt-4o-mini", SIMPLE).model == "openai/gpt-6-luna"
    # with allow_upgrade=False the price never rises above the configured model: gpt-6-sol/astra cost
    # more than gpt-4o-mini, so the cap steps down to gpt-6-luna, which is cheaper still
    d = _decide("gpt-4o-mini", COMPLEX, allow_upgrade=False)
    assert d.model == "openai/gpt-6-luna" and "capped" in d.reason
    assert _decide("gpt-4o-mini", COMPLEX).model == "openai/gpt-6-astra"   # default: upgrades allowed


def test_price_cap_keeps_configured_when_nothing_is_cheaper():
    d = _decide("openai/gpt-6-luna", COMPLEX, allow_upgrade=False)
    assert d.model == "openai/gpt-6-luna"


def test_chatanthropic_on_openrouter_anthropic_api(fake_network, monkeypatch):
    from langchain_anthropic import ChatAnthropic
    from levrouter.providers import openrouter_full_id

    assert openrouter_full_id("claude-haiku-4-5-20251001") == "anthropic/claude-haiku-4.5"
    assert openrouter_full_id("claude-opus-5-5") == "anthropic/claude-opus-5.5"

    def fake(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"model={self.model}"))])

    monkeypatch.setattr(ChatAnthropic, "_generate", fake)
    _enable(COMPLEX)
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=KEY, base_url="https://openrouter.ai/api")
    out = llm.invoke("Design a distributed cache")
    assert out.content == "model=anthropic/claude-opus-5.5"
    assert out.response_metadata["levrouter"]["provider"] == "openrouter"


def test_same_model_under_two_names_is_not_routed_or_evaluated():
    from levrouter.providers import same_model
    assert same_model("openrouter", "claude-sonnet-5", "anthropic/claude-sonnet-5")
    assert same_model("openrouter", "claude-haiku-4-5-20251001", "anthropic/claude-haiku-4.5")
    assert not same_model("openrouter", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4.5")
    router = Router(RouterConfig(), FixedClassifier(STANDARD := __import__("conftest").STANDARD))
    d = router.decide("openrouter", "claude-sonnet-5", extract([HumanMessage("Summarise this")]))
    assert d.model == "claude-sonnet-5"   # the app's own id is kept, so eval mode sees "same model" and skips
