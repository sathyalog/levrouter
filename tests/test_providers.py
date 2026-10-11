"""Every supported provider class, endpoint detection, and the guardrails around swapping models."""

import logging

import pytest
from conftest import COMPLEX, SIMPLE, FixedClassifier
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI

import levrouter
from levrouter import Router, RouterConfig
from levrouter.features import extract

SECRET = "sk-test-SUPERSECRETKEY1234567890"


def _model_of(llm):
    for f in ("model_name", "model", "model_id"):
        v = getattr(llm, f, None)
        if isinstance(v, str):
            return v


def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
    msg = AIMessage(content=f"model={_model_of(self)}", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    return ChatResult(generations=[ChatGeneration(message=msg)])


CLASSES = [ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI, ChatGroq, ChatDeepSeek, ChatMistralAI, ChatXAI, ChatHuggingFace]


@pytest.fixture
def fake_network(monkeypatch):
    for cls in CLASSES:
        monkeypatch.setattr(cls, "_generate", _fake_generate)


def _enable(sig=SIMPLE, **cfg):
    levrouter.enable(router=Router(RouterConfig(log_level=None, summary_at_exit=False, **cfg), FixedClassifier(sig)))


def _hf(repo="Qwen/Qwen3.8-2.4T-A95B"):
    return ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id=repo, huggingfacehub_api_token="hf_test", provider="auto"))


@pytest.mark.parametrize(
    "make, simple_model, complex_model",
    [
        (lambda: ChatAnthropic(model="claude-sonnet-5", api_key=SECRET), "claude-haiku-4-5", "claude-opus-5-5"),
        (lambda: ChatOpenAI(model="gpt-6-sol", api_key=SECRET), "gpt-6-luna", "gpt-6-astra"),
        (lambda: ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=SECRET), "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"),
        (lambda: ChatGroq(model="openai/gpt-oss-120b", api_key=SECRET), "openai/gpt-oss-20b", "openai/gpt-oss-120b"),
        (lambda: ChatDeepSeek(model="deepseek-flash", api_key=SECRET), "deepseek-flash", "deepseek-v4-pro"),
        (lambda: ChatMistralAI(model="mistral-large-latest", api_key=SECRET), "mistral-small-latest", "mistral-medium-latest"),
        (lambda: ChatXAI(model="grok-4.7", api_key=SECRET), "grok-4.7", "grok-4.7"),
        (_hf, "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-2.4T-A95B"),
    ],
    ids=["anthropic", "openai", "gemini", "groq", "deepseek", "mistral", "xai", "huggingface"],
)
def test_each_provider_routes_within_itself(fake_network, make, simple_model, complex_model):
    _enable(SIMPLE)
    assert make().invoke("hi").content == f"model={simple_model}"
    levrouter.disable()
    _enable(COMPLEX)
    assert make().invoke("hi").content == f"model={complex_model}"


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
        ("https://api.deepseek.com", "deepseek-flash"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.1-flash-lite"),
        ("https://router.huggingface.co/v1", "Qwen/Qwen3.5-9B"),
        ("https://api.x.ai/v1", "grok-4.7"),
    ],
)
def test_chatopenai_on_compatible_endpoints_uses_that_providers_models(fake_network, base_url, expected):
    _enable(SIMPLE)
    llm = ChatOpenAI(model="whatever-they-configured", api_key=SECRET, base_url=base_url)
    out = llm.invoke("hi")
    assert out.content == f"model={expected}"
    assert out.response_metadata["levrouter"]["provider"] != "openai"


def test_chatanthropic_on_deepseek_anthropic_endpoint(fake_network):
    _enable(SIMPLE)
    llm = ChatAnthropic(model="deepseek-v4-pro", api_key=SECRET, base_url="https://api.deepseek.com/anthropic")
    assert llm.invoke("hi").content == "model=deepseek-flash"


def test_unknown_endpoint_is_never_routed(fake_network, caplog):
    _enable(SIMPLE)
    llm = ChatOpenAI(model="my-model", api_key=SECRET, base_url="https://llm-proxy.example.net/v1")
    with caplog.at_level(logging.INFO, logger="levrouter"):
        assert llm.invoke("hi").content == "model=my-model"
        llm.invoke("again")
    skips = [r.getMessage() for r in caplog.records if r.getMessage().startswith("skip")]
    assert skips == ["skip  ChatOpenAI calls are not routed: unknown endpoint host llm-proxy.example.net"]


def test_gateway_host_can_be_mapped(fake_network):
    _enable(SIMPLE, provider_hosts={"llm-gateway.corp.example": "anthropic"})
    llm = ChatAnthropic(model="claude-opus-5-5", api_key=SECRET, base_url="https://llm-gateway.corp.example")
    assert llm.invoke("hi").content == "model=claude-haiku-4-5"


def test_disabled_provider_is_not_routed(fake_network):
    levrouter.enable(
        router=Router(RouterConfig(log_level=None, summary_at_exit=False), FixedClassifier(SIMPLE)),
        providers=("anthropic", "openai"),
    )
    llm = ChatOpenAI(model="openai/gpt-oss-120b", api_key=SECRET, base_url="https://api.groq.com/openai/v1")
    assert llm.invoke("hi").content == "model=openai/gpt-oss-120b"


def test_fine_tuned_models_are_never_swapped(fake_network):
    _enable(SIMPLE)
    llm = ChatOpenAI(model="ft:gpt-6-sol:acme:support:abc123", api_key=SECRET)
    assert llm.invoke("hi").content == "model=ft:gpt-6-sol:acme:support:abc123"


def test_dedicated_hf_endpoint_is_not_routed(fake_network):
    _enable(SIMPLE)
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(endpoint_url="https://xyz.endpoints.huggingface.cloud", huggingfacehub_api_token="hf_test"),
        model_id="acme/private-model",
    )
    # ChatHuggingFace reports the endpoint URL as model_id; the point is it is unchanged.
    assert llm.invoke("hi").content == f"model={llm.model_id}"


def test_api_key_stays_on_the_copy_and_out_of_logs(fake_network, tmp_path, capsys):
    path = tmp_path / "calls.jsonl"
    levrouter.enable(router=Router(RouterConfig(log_path=str(path), summary_at_exit=False), FixedClassifier(SIMPLE)))
    seen = {}

    def spy(self, messages, stop=None, run_manager=None, **kw):
        seen["key"] = self.anthropic_api_key.get_secret_value()
        seen["url"] = self.anthropic_api_url
        return _fake_generate(self, messages, stop, run_manager, **kw)

    levrouter.disable()
    ChatAnthropic._generate = spy
    levrouter.enable(router=Router(RouterConfig(log_path=str(path), summary_at_exit=False), FixedClassifier(SIMPLE)))
    ChatAnthropic(model="claude-opus-5-5", api_key=SECRET).invoke(f"my key is {SECRET}, email a@b.com")
    assert seen == {"key": SECRET, "url": "https://api.anthropic.com"}
    written = capsys.readouterr().err + path.read_text()
    assert SECRET not in written and "a@b.com" not in written
    assert "[SECRET]" in written and "[EMAIL]" in written
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_capability_guard_context_and_vision():
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))
    big = extract([HumanMessage("x" * 1_000_000)])  # ~285k tokens: over Haiku's 200k
    d = router.decide("anthropic", "claude-opus-5-5", big)
    assert d.model == "claude-sonnet-5" and "context" in d.reason

    image = extract([HumanMessage(content=[{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:..."}}])])
    d = router.decide("groq", "meta-llama/llama-4-scout-17b-16e-instruct", image)
    assert d.model == "meta-llama/llama-4-scout-17b-16e-instruct" and "kept configured" in d.reason


def test_min_tier_floor():
    d = Router(RouterConfig(min_tier="standard"), FixedClassifier(SIMPLE)).decide("openai", "gpt-6-astra", extract([HumanMessage("hi")]))
    assert d.model == "gpt-6-sol" and "min_tier" in d.reason


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        RouterConfig(min_tier="tiny")
    with pytest.raises(ValueError):
        RouterConfig(tiers={"cohere": {"simple": "command-a"}})  # new provider without all three tiers


def test_tier_overrides_merge_with_defaults():
    cfg = RouterConfig(tiers={"openai": {"simple": "gpt-4o-mini"}})
    assert cfg.tiers["openai"] == {"simple": "gpt-4o-mini", "standard": "gpt-6-sol", "complex": "gpt-6-astra"}
    assert cfg.tiers["anthropic"]["simple"] == "claude-haiku-4-5"  # other providers untouched
