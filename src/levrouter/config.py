"""Router configuration: tier maps per provider, thresholds, guardrails, observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .providers import OPENROUTER_FAMILIES, PROVIDERS, all_models

Tier = Literal["simple", "standard", "complex"]
TIERS: tuple[Tier, ...] = ("simple", "standard", "complex")

Backend = Literal["lev-local", "lev-http", "heuristic"]
Mode = Literal["route", "eval"]

DEFAULT_TIERS: dict[str, dict[str, str]] = {name: dict(p.tiers) for name, p in PROVIDERS.items()}

# USD per million tokens (input, output). Used only for the savings estimate in telemetry.
PRICES: dict[str, tuple[float, float]] = {m.id: (m.input_price, m.output_price) for m in all_models().values()}

# Output-token ceilings, so a routed call does not inherit a max_tokens the smaller model rejects.
MAX_OUTPUT_TOKENS: dict[str, int] = {m.id: m.max_output for m in all_models().values() if m.max_output}

DEFAULT_NEVER_ROUTE = (
    "ft:*", "*tunedModels/*", "*:ft-*", "accounts/*",   # fine-tunes and custom deployments
    "openrouter/*",                                      # OpenRouter's own routers (auto, free)
    "*:free",                                            # never turn a free call into a paid one
    "~*",                                                # "latest" aliases that float between models
    "*:batch",
)


@dataclass
class RouterConfig:
    backend: Backend = "lev-local"
    lev_checkpoint: str = "interfaze-ai/lev"
    lev_url: str = "http://localhost:8000"
    # Bearer token for a protected `lev serve`. Read from LEVROUTER_LEV_API_KEY; never logged.
    lev_api_key: str | None = field(default=None, repr=False)
    # lev-http only: extra request headers for a protected server, as {header name: ENV VAR NAME}.
    # The file holds only variable names; the values are read from the environment, so no secret is committed.
    # Example (Modal proxy auth): {"Modal-Key": "MODAL_KEY", "Modal-Secret": "MODAL_SECRET"}.
    lev_headers: dict[str, str] = field(default_factory=dict)
    # lev-http only: send one background request at load_config() so a scale-to-zero server starts
    # before the first real call. Off by default because it is a billable request.
    warm_on_load: bool = False
    tiers: dict[str, dict[str, str]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_TIERS.items()})
    # A request is "complex" when P(complex) + 0.25*P(reasoning) + 0.15*P(code) reaches this.
    complex_threshold: float = 0.5
    # A request is "simple" when P(simple) reaches this and reasoning is unlikely.
    simple_threshold: float = 0.55
    # Below this top-choice probability the decision is bumped one tier up (safe side).
    min_confidence: float = 0.45
    # Seconds to wait for one lev decision before falling back to the heuristic.
    timeout_s: float = 1.5
    # lev-local: if False, the first call starts loading lev in the background and
    # routes with the heuristic until it is ready, instead of blocking the app.
    block_on_load: bool = False
    cache_size: int = 1024
    # Keep one conversation on one model so the provider's prompt cache keeps working: a thread
    # (LangGraph thread_id, or metadata={"levrouter_thread": ...}) can move up a tier, never down.
    sticky_threads: bool = True
    # Idle seconds before a thread may change model again: Anthropic's default prompt-cache lifetime
    # (set 3600 when using the 1-hour cache).
    thread_ttl_s: int = 300
    # Asked again this many times (rephrased, or "that's wrong, try again"): move up a tier; 0 = off.
    escalate_after_repeats: int = 2
    max_state_chars: int = 2000

    # -- guardrails -------------------------------------------------------------
    # Let routing pick a model more capable (pricier) than the one the developer configured.
    allow_upgrade: bool = True
    # Quality floor: never route below this tier (e.g. "standard" for customer-facing answers).
    min_tier: Tier | None = None
    # Current-model patterns that are never swapped (fine-tunes, tuned or custom deployments).
    never_route: tuple[str, ...] = DEFAULT_NEVER_ROUTE
    # Extra endpoint hosts to treat as a provider, e.g. {"llm-gateway.corp.com": "anthropic"}.
    # Unknown hosts are never routed.
    provider_hosts: dict[str, str] = field(default_factory=dict)
    # Keep routed models within context window / input types (image, PDF, audio, video) / tool support;
    # bump the tier otherwise.
    capability_guard: bool = True
    # Redact PII and secrets from everything levrouter writes (logs, JSONL, spans).
    redact_pii: bool = True
    # lev-http only: allow a lev server that is not on this machine / private network.
    # Prompts are sent to it, so it is off by default; when on, https is required.
    allow_remote_lev: bool = False
    # Redact PII from the state sent to a lev-http server (lev-local never leaves the process).
    redact_classifier_input: bool = True

    # -- OpenRouter -------------------------------------------------------------
    # "family": stay with the vendor your code chose (anthropic/opus -> anthropic/haiku).
    # "ladder": pick the cheapest capable model across openrouter_vendors (opt-in).
    openrouter_mode: Literal["family", "ladder"] = "family"
    # Vendors the ladder may use; None = every vendor in openrouter_families.
    openrouter_vendors: tuple[str, ...] | None = None
    # Per-vendor tier overrides, merged over the defaults.
    openrouter_families: dict[str, dict[str, str]] = field(default_factory=dict)

    # -- evaluation (see levrouter/evaluate.py) -------------------------------
    # "eval": for a sample of calls, also run the other model and have a judge compare the answers.
    mode: Mode = "route"
    eval_sample_rate: float = 0.2
    # Which answer the app receives in eval mode: "baseline" (configured model, zero risk) or "routed".
    eval_serve: Literal["baseline", "routed"] = "baseline"
    # Cap on the extra spend of evaluation (second call + judge) per 24 hours.
    eval_budget_usd: float = 1.0
    # "top" = the same provider's complex tier (OpenRouter: the vendor's top model), or a model id served
    # by the same endpoint and key as the app's chat model (the judge is a copy of it with another model).
    eval_judge: str = "top"
    # Opt-in JSONL of (redacted) prompts and both answers, for reviewing regressions.
    eval_log_path: str | None = None

    # -- observability ----------------------------------------------------------
    # Level for the "levrouter" logger, printed to stderr. None: configure nothing and
    # let records propagate to the host app's own logging setup.
    log_level: str | None = "INFO"
    log_format: Literal["text", "json"] = "text"
    # Characters of the (redacted) user prompt shown in each log line; 0 hides prompt text.
    log_prompt_chars: int = 60
    # Append one JSON line per finished call to this file (created with owner-only permissions).
    log_path: str | None = None
    # Log a per-model / per-task cost summary when the process exits.
    summary_at_exit: bool = True
    # Emit an OpenTelemetry span per call when opentelemetry-api is installed.
    otel: bool = True

    # -- usage (last 24h, see levrouter/usage_store.py) ---------------------------------
    # Rolling hourly files under <usage_home>/usage/, read by `levrouter usage`.
    usage_file: bool = True
    # In-memory window answered over a local socket, read by `levrouter usage --live`.
    usage_live: bool = True
    usage_retention_hours: int = 24
    usage_memory_max: int = 100_000
    # Default: $LEVROUTER_HOME, else ~/.levrouter
    usage_home: str | None = None

    def __post_init__(self) -> None:
        # Merge overrides over the defaults: tiers={"openai": {"simple": "gpt-4o-mini"}} changes that one
        # tier and keeps every other provider and tier.
        merged = {k: dict(v) for k, v in DEFAULT_TIERS.items()}
        for provider, mapping in self.tiers.items():
            merged.setdefault(provider, {}).update(mapping)
        self.tiers = merged
        families = {k: dict(v) for k, v in OPENROUTER_FAMILIES.items()}
        for vendor, mapping in self.openrouter_families.items():
            families.setdefault(vendor, {}).update(mapping)
        self.openrouter_families = families
        self.never_route = tuple(self.never_route)
        if isinstance(self.openrouter_vendors, str):  # a comma-separated string: "anthropic,google"
            self.openrouter_vendors = tuple(v.strip() for v in self.openrouter_vendors.split(",") if v.strip())
        elif self.openrouter_vendors is not None:
            self.openrouter_vendors = tuple(self.openrouter_vendors)
        if not 0.0 <= self.eval_sample_rate <= 1.0:
            raise ValueError("eval_sample_rate must be between 0 and 1")
        if self.min_tier is not None and self.min_tier not in TIERS:
            raise ValueError(f"min_tier must be one of {TIERS}, got {self.min_tier!r}")
        for provider, mapping in self.tiers.items():
            missing = [t for t in TIERS if not mapping.get(t)]
            if missing:
                raise ValueError(f"tiers[{provider!r}] is missing {missing}")
