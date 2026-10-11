"""Provider registry: which LangChain class talks to which provider, which host it must
point at, and the cheapest -> most capable model ladder inside that provider.

Model ids and prices were checked against each provider's docs in September 2026.
They change often: override any tier with `models:` in levrouter.yaml or RouterConfig(tiers=...).
Prices are USD per 1M tokens (input, output) and only feed the savings estimate.
"""

from __future__ import annotations

import dataclasses

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse


# Input types a model accepts, as listed in the OpenRouter model catalog (September 2026).
TEXT = frozenset({"text"})
TEXT_IMAGE = frozenset({"text", "image"})
TEXT_IMAGE_FILE = frozenset({"text", "image", "file"})
TEXT_IMAGE_VIDEO = frozenset({"text", "image", "video"})
ALL_INPUTS = frozenset({"text", "image", "file", "audio", "video"})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    input_price: float
    output_price: float
    context: int
    max_output: int | None = None
    inputs: frozenset[str] = TEXT_IMAGE_FILE
    tools: bool = True
    # Accepts a forced tool call (tool_choice "any"/named tool), which with_structured_output uses.
    forced_tools: bool = True

    @property
    def vision(self) -> bool:
        return "image" in self.inputs


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    # simple / standard / complex -> model id
    tiers: dict[str, str]
    models: tuple[ModelSpec, ...]
    # Hosts this provider's API is served from. A chat model pointed anywhere else
    # (a proxy, another vendor's OpenAI-compatible endpoint) is not routed.
    hosts: tuple[str, ...]
    note: str = ""

    def spec(self, model_id: str) -> ModelSpec | None:
        return next((m for m in self.models if m.id == model_id), None)


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        "anthropic",
        {"simple": "claude-haiku-4-5", "standard": "claude-sonnet-5", "complex": "claude-opus-5-5"},
        (
            ModelSpec("claude-haiku-4-5", 1.00, 5.00, 200_000, 64_000),
            ModelSpec("claude-sonnet-5", 2.00, 10.00, 1_000_000, 128_000),
            # Opus 5.5 and Fable 5.1 reject forced tool_choice (400), so they can't serve with_structured_output's
            # default tool-calling mode.
            ModelSpec("claude-opus-5-5", 4.00, 20.00, 1_000_000, 128_000, forced_tools=False),
            ModelSpec("claude-opus-5", 5.00, 25.00, 1_000_000, 128_000),
            ModelSpec("claude-fable-5-1", 10.00, 50.00, 1_000_000, 128_000, forced_tools=False),
            ModelSpec("claude-sonnet-4-6", 3.00, 15.00, 1_000_000, 128_000),
            ModelSpec("claude-sonnet-4-5", 3.00, 15.00, 1_000_000, 64_000),
        ),
        ("api.anthropic.com",),
    ),
    "openai": ProviderSpec(
        "openai",
        {"simple": "gpt-6-luna", "standard": "gpt-6-sol", "complex": "gpt-6-astra"},
        (
            ModelSpec("gpt-6-luna", 0.10, 0.50, 1_050_000),
            ModelSpec("gpt-6-sol", 2.00, 10.00, 1_050_000),
            ModelSpec("gpt-6-astra", 10.00, 50.00, 1_050_000),
            ModelSpec("gpt-4o-mini", 0.15, 0.60, 128_000, 16_384),
            ModelSpec("gpt-4o", 2.50, 10.00, 128_000, 16_384),
        ),
        ("api.openai.com",),
    ),
    "gemini": ProviderSpec(
        "gemini",
        {"simple": "gemini-3.1-flash-lite", "standard": "gemini-3.8-flash", "complex": "gemini-3.1-pro-preview"},
        (
            ModelSpec("gemini-3.1-flash-lite", 0.25, 1.50, 1_000_000, inputs=ALL_INPUTS),
            # Introductory price through 2026-12-31; $1.50 / $7.50 after.
            ModelSpec("gemini-3.8-flash", 0.75, 3.75, 1_000_000, inputs=ALL_INPUTS),
            ModelSpec("gemini-3.1-pro-preview", 2.00, 12.00, 1_000_000, inputs=ALL_INPUTS),
            ModelSpec("gemini-2.5-flash-lite", 0.10, 0.40, 1_000_000, inputs=ALL_INPUTS),
            ModelSpec("gemini-2.5-pro", 1.25, 10.00, 1_000_000, inputs=ALL_INPUTS),
        ),
        ("generativelanguage.googleapis.com", "aiplatform.googleapis.com"),
    ),
    "groq": ProviderSpec(
        "groq",
        # Groq serves open-weight models only; its strongest self-serve production model
        # is gpt-oss-120b, so standard and complex share it.
        {"simple": "openai/gpt-oss-20b", "standard": "openai/gpt-oss-120b", "complex": "openai/gpt-oss-120b"},
        (
            ModelSpec("openai/gpt-oss-20b", 0.075, 0.30, 131_072, 65_536, inputs=TEXT),
            ModelSpec("openai/gpt-oss-120b", 0.15, 0.60, 131_072, 65_536, inputs=TEXT),
            ModelSpec("qwen/qwen3.8-27b", 0.80, 4.00, 131_072, 16_384, inputs=TEXT_IMAGE_VIDEO),
        ),
        ("api.groq.com",),
    ),
    "deepseek": ProviderSpec(
        "deepseek",
        {"simple": "deepseek-flash", "standard": "deepseek-flash", "complex": "deepseek-v4-pro"},
        (
            # Peak-hour prices; off-peak is half.
            ModelSpec("deepseek-flash", 0.30, 1.20, 1_000_000, 384_000, inputs=TEXT_IMAGE),
            ModelSpec("deepseek-v4-pro", 1.32, 3.96, 1_000_000, 384_000, inputs=TEXT),
        ),
        ("api.deepseek.com",),
    ),
    "mistral": ProviderSpec(
        "mistral",
        # Mistral Large 3 is priced below Medium 3.5, which is Mistral's frontier model.
        {"simple": "mistral-small-latest", "standard": "mistral-large-latest", "complex": "mistral-medium-latest"},
        (
            ModelSpec("mistral-small-latest", 0.15, 0.60, 128_000, inputs=TEXT_IMAGE),
            ModelSpec("mistral-large-latest", 0.50, 1.50, 128_000),
            ModelSpec("mistral-medium-latest", 1.50, 7.50, 128_000),
        ),
        ("api.mistral.ai",),
    ),
    "xai": ProviderSpec(
        "xai",
        # xAI lists a single current text model; routing only changes if you configure cheaper ones.
        {"simple": "grok-4.7", "standard": "grok-4.7", "complex": "grok-4.7"},
        (ModelSpec("grok-4.7", 2.00, 6.00, 500_000),),
        ("api.x.ai",),
    ),
    "huggingface": ProviderSpec(
        "huggingface",
        # Inference Providers router; prices are the cheapest listed provider and vary by provider.
        {"simple": "Qwen/Qwen3.5-9B", "standard": "Qwen/Qwen3.6-35B-A3B", "complex": "Qwen/Qwen3.8-2.4T-A95B"},
        (
            ModelSpec("Qwen/Qwen3.5-9B", 0.10, 0.15, 262_144, inputs=TEXT_IMAGE_VIDEO),
            ModelSpec("Qwen/Qwen3.6-35B-A3B", 0.10, 0.95, 262_144, inputs=TEXT_IMAGE_VIDEO),
            ModelSpec("Qwen/Qwen3.8-2.4T-A95B", 2.00, 6.00, 262_144, inputs=TEXT),
            ModelSpec("meta-llama/Llama-3.1-8B-Instruct", 0.02, 0.05, 131_072, inputs=TEXT, tools=False),
        ),
        ("router.huggingface.co", "api-inference.huggingface.co"),
    ),
}


def all_models() -> dict[str, ModelSpec]:
    return {m.id: m for p in PROVIDERS.values() for m in p.models}


_DATED = re.compile(r"(?:-|@)(?:20\d{6})$")


def canonical_id(model: str | None) -> str:
    """Model id without a dated snapshot suffix: claude-haiku-4-5-20251001 -> claude-haiku-4-5."""
    return _DATED.sub("", model or "")


# -- OpenRouter ----------------------------------------------------------------

OPENROUTER_HOSTS = ("openrouter.ai",)

# Vendor prefix -> tier models. Default "family" mode stays inside the vendor your code chose.
OPENROUTER_FAMILIES: dict[str, dict[str, str]] = {
    "anthropic": {"simple": "anthropic/claude-haiku-4.5", "standard": "anthropic/claude-sonnet-5",
                  "complex": "anthropic/claude-opus-5.5"},
    "openai": {"simple": "openai/gpt-6-luna", "standard": "openai/gpt-6-sol", "complex": "openai/gpt-6-astra"},
    "google": {"simple": "google/gemini-3.1-flash-lite", "standard": "google/gemini-3.8-flash",
               "complex": "google/gemini-3.1-pro-preview"},
    "deepseek": {"simple": "deepseek/deepseek-v4.1-flash", "standard": "deepseek/deepseek-v4.1-flash",
                 "complex": "deepseek/deepseek-v4-pro"},
    "x-ai": {"simple": "x-ai/grok-4.3", "standard": "x-ai/grok-4.7", "complex": "x-ai/grok-4.7"},
    "mistralai": {"simple": "mistralai/mistral-small-2603", "standard": "mistralai/mistral-large-2512",
                  "complex": "mistralai/mistral-medium-3-5"},
    "qwen": {"simple": "qwen/qwen3.5-9b", "standard": "qwen/qwen3.6-35b-a3b", "complex": "qwen/qwen3.8-max-0902"},
}


def openrouter_catalog_path() -> Path:
    """User copy written by `levrouter openrouter refresh`, if present."""
    home = os.environ.get("LEVROUTER_HOME") or str(Path.home() / ".levrouter")
    return Path(home).expanduser() / "openrouter_models.json"


def available_models_path() -> Path:
    """Model ids each provider's API reported for your key, written by `levrouter models refresh`."""
    return openrouter_catalog_path().with_name("available_models.json")


@lru_cache(maxsize=1)
def openrouter_catalog() -> dict[str, dict]:
    """OpenRouter models: the refreshed user copy if present, else the snapshot shipped with the package."""
    user = openrouter_catalog_path()
    try:
        raw = user.read_text() if user.is_file() else None
    except OSError:
        raw = None
    if raw is None:
        raw = resources.files("levrouter").joinpath("data", "openrouter_models.json").read_text()
    return json.loads(raw)["models"]


CATALOG_VENDORS = frozenset({"anthropic", "openai", "google", "deepseek", "x-ai", "mistralai", "qwen", "meta-llama"})


def build_openrouter_catalog(api_models: list[dict]) -> dict:
    """Trim OpenRouter's /api/v1/models response to what routing needs (prices per 1M tokens)."""
    out = {}
    for m in api_models:
        mid = m.get("id", "")
        if mid.split("/")[0] not in CATALOG_VENDORS or mid.endswith(":batch"):
            continue
        pricing = m.get("pricing") or {}
        try:
            pin, pout = float(pricing.get("prompt", 0)) * 1e6, float(pricing.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            continue
        if pin < 0 or pout < 0:  # router pseudo-models carry negative prices
            continue
        arch = m.get("architecture") or {}
        out[mid] = {
            "input_price": round(pin, 6), "output_price": round(pout, 6),
            "context": m.get("context_length") or 0,
            "max_output": (m.get("top_provider") or {}).get("max_completion_tokens"),
            "inputs": sorted(set(arch.get("input_modalities") or ["text"])),
            "tools": "tools" in (m.get("supported_parameters") or []),
        }
    return {"source": "https://openrouter.ai/api/v1/models", "models": out}


def openrouter_full_id(model: str) -> str:
    """OpenRouter also accepts bare names ("gpt-4o-mini"); resolve them to "openai/gpt-4o-mini" via the catalog."""
    model = model or ""
    if "/" in model:
        return model
    base = model.split(":", 1)[0]
    canon = canonical_id(base)
    # Provider-native ids: claude-haiku-4-5-20251001 -> claude-haiku-4.5 (OpenRouter writes versions with a dot)
    candidates = [base, canon, re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", canon)]
    catalog = openrouter_catalog()
    for name in candidates:
        matches = [mid for mid in catalog if mid.split("/", 1)[1] == name]
        if len(matches) == 1:
            return matches[0] + model[len(base):]
    return model


def same_model(provider: str, a: str, b: str) -> bool:
    """Two ids for one model: dated snapshots, and on OpenRouter bare vs vendor-prefixed names."""
    if provider == "openrouter":
        return canonical_id(openrouter_full_id(a).split(":", 1)[0]) == canonical_id(openrouter_full_id(b).split(":", 1)[0]) \
            and a.endswith(":free") == b.endswith(":free")
    return canonical_id(a) == canonical_id(b)


def openrouter_vendor(model: str) -> str:
    return openrouter_full_id(model).split("/", 1)[0]


# Models that reject a forced tool call; the OpenRouter catalog does not publish this.
NO_FORCED_TOOLS = frozenset({"claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1",
                             "anthropic/claude-opus-5.5", "anthropic/claude-fable-5.1", "anthropic/claude-mythos-5.1"})


def _catalog_spec(model: str) -> ModelSpec | None:
    full = openrouter_full_id(model)
    entry = openrouter_catalog().get(full)
    if entry is None:
        return None
    return ModelSpec(model, entry["input_price"], entry["output_price"], entry["context"] or 0,
                     entry.get("max_output"), frozenset(entry["inputs"]), bool(entry.get("tools", True)),
                     forced_tools=full.split(":", 1)[0] not in NO_FORCED_TOOLS)


# Direct provider -> its vendor prefix in the OpenRouter catalogue, used for models newer than the built-in list.
CATALOG_VENDOR_OF = {"anthropic": "anthropic", "openai": "openai", "gemini": "google", "xai": "x-ai",
                     "mistral": "mistralai", "deepseek": "deepseek"}


def spec_for(provider: str, model: str) -> ModelSpec | None:
    """Capabilities and prices of a model, understanding dated ids and OpenRouter slugs.

    Direct providers use the built-in list first, then the OpenRouter catalogue (refreshed by
    `levrouter models refresh`), so new models still get prices and guardrail checks.
    """
    if provider == "openrouter":
        return _catalog_spec(model)
    spec = PROVIDERS.get(provider)
    found = spec.spec(canonical_id(model)) if spec else None
    vendor = CATALOG_VENDOR_OF.get(provider)
    if found is None and vendor and model:
        full = openrouter_full_id(canonical_id(model))
        if full.startswith(vendor + "/"):
            entry = _catalog_spec(full)
            found = dataclasses.replace(entry, id=model) if entry else None
    return found


def price_of(model: str) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens for any known model id, dated or OpenRouter."""
    spec = all_models().get(canonical_id(model)) or _catalog_spec(model)
    return (spec.input_price, spec.output_price) if spec else None


# -- LangChain classes -------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One LangChain chat model class and how to read / change its model and endpoint."""

    module: str
    class_name: str
    model_field: str
    # Provider used when the endpoint is the class default.
    default_provider: str
    # Fields holding the endpoint URL, checked in order.
    url_fields: tuple[str, ...] = ()
    # Env vars the SDK reads for the endpoint when the field is empty.
    url_env: tuple[str, ...] = ()
    max_tokens_fields: tuple[str, ...] = ("max_tokens",)
    extra: dict = field(default_factory=dict)


TARGETS: dict[str, Target] = {
    "ChatAnthropic": Target("langchain_anthropic", "ChatAnthropic", "model", "anthropic",
                            ("anthropic_api_url",), ("ANTHROPIC_BASE_URL",)),
    "ChatOpenAI": Target("langchain_openai", "ChatOpenAI", "model_name", "openai",
                         ("openai_api_base",), ("OPENAI_BASE_URL", "OPENAI_API_BASE")),
    "ChatGoogleGenerativeAI": Target("langchain_google_genai", "ChatGoogleGenerativeAI", "model", "gemini",
                                     ("base_url",), (), ("max_output_tokens",)),
    "ChatGroq": Target("langchain_groq", "ChatGroq", "model_name", "groq", ("groq_api_base",), ("GROQ_API_BASE",)),
    "ChatDeepSeek": Target("langchain_deepseek", "ChatDeepSeek", "model_name", "deepseek", ("api_base",)),
    "ChatMistralAI": Target("langchain_mistralai", "ChatMistralAI", "model", "mistral", ("endpoint",)),
    "ChatXAI": Target("langchain_xai", "ChatXAI", "model_name", "xai", ("xai_api_base",)),
    "ChatHuggingFace": Target("langchain_huggingface", "ChatHuggingFace", "model_id", "huggingface"),
    "ChatOpenRouter": Target("langchain_openrouter", "ChatOpenRouter", "model_name", "openrouter",
                             ("openrouter_api_base",), ("OPENROUTER_BASE_URL",)),
}

# Which classes to patch for each provider name passed to enable(providers=...).
# ChatOpenAI and ChatAnthropic appear under several providers because DeepSeek, Groq,
# Gemini, xAI, Mistral and Hugging Face also serve OpenAI- (or Anthropic-) compatible APIs.
PROVIDER_CLASSES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ChatAnthropic",),
    "openai": ("ChatOpenAI",),
    "gemini": ("ChatGoogleGenerativeAI", "ChatOpenAI"),
    "groq": ("ChatGroq", "ChatOpenAI"),
    "deepseek": ("ChatDeepSeek", "ChatOpenAI", "ChatAnthropic"),
    "mistral": ("ChatMistralAI", "ChatOpenAI"),
    "xai": ("ChatXAI", "ChatOpenAI"),
    "huggingface": ("ChatHuggingFace", "ChatOpenAI"),
    # OpenRouter also serves an Anthropic-compatible API (ChatAnthropic with base_url=https://openrouter.ai/api).
    "openrouter": ("ChatOpenRouter", "ChatOpenAI", "ChatAnthropic"),
}

ALL_PROVIDERS: tuple[str, ...] = (*PROVIDERS, "openrouter")


def host_of(url: str | None) -> str | None:
    if not url:
        return None
    return (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower() or None


def resolve_provider(llm, target: Target, extra_hosts: dict[str, str] | None = None) -> tuple[str | None, str]:
    """Which provider this instance actually talks to, judged by its endpoint.

    Returns (provider, why). provider is None when the endpoint is not a known
    provider host: then the call is not routed, because a model id from the class's
    usual provider would be wrong (or leaked) at that endpoint.
    """
    if target.class_name == "ChatHuggingFace":
        inner = getattr(llm, "llm", None)
        # Only the serverless Inference Providers path takes a per-request model;
        # dedicated endpoints and local pipelines serve one fixed model.
        if type(inner).__name__ == "HuggingFaceEndpoint" and getattr(inner, "repo_id", None) and not getattr(inner, "endpoint_url", None):
            return "huggingface", "serverless endpoint"
        return None, "dedicated Hugging Face endpoint or local pipeline serves one fixed model"

    url = next((getattr(llm, f, None) for f in target.url_fields if getattr(llm, f, None)), None)
    if url is None:
        url = next((os.environ[e] for e in target.url_env if os.environ.get(e)), None)
    host = host_of(str(url)) if url else None
    if host is None:
        return target.default_provider, "default endpoint"

    for provider_host, provider in (extra_hosts or {}).items():
        if host == provider_host.lower():
            return provider, f"configured host {host}"
    if any(host == h or host.endswith("." + h) for h in OPENROUTER_HOSTS):
        if target.class_name in PROVIDER_CLASSES["openrouter"]:
            return "openrouter", f"host {host}"
    for name, spec in PROVIDERS.items():
        if any(host == h or host.endswith("." + h) for h in spec.hosts):
            if name not in PROVIDER_CLASSES or target.class_name in PROVIDER_CLASSES[name]:
                return name, f"host {host}"
    return None, f"unknown endpoint host {host}"
