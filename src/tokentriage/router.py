"""Classify a request, map the signals to a tier, resolve the tier to a model id."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from .classifiers.base import Classifier, Signals
from .classifiers.heuristic import HeuristicClassifier
from .config import TIERS, RouterConfig, Tier
from .features import RequestFeatures
from .providers import canonical_id, openrouter_vendor, price_of, same_model, spec_for
from .repeats import repeat_count

log = logging.getLogger("tokentriage")


@dataclass(frozen=True)
class Decision:
    provider: str
    tier: Tier
    model: str
    original_model: str
    reason: str
    signals: Signals | None = None
    cached: bool = False

    def as_dict(self) -> dict:
        d = {
            "provider": self.provider,
            "tier": self.tier,
            "model": self.model,
            "original_model": self.original_model,
            "reason": self.reason,
            "cached": self.cached,
        }
        if self.signals:
            d["source"] = self.signals.source
            d["p_tier"] = {k: round(v, 3) for k, v in self.signals.tier.items()}
            d["needs_reasoning"] = round(self.signals.needs_reasoning, 3)
            d["needs_code"] = round(self.signals.needs_code, 3)
            d["classify_ms"] = round(self.signals.latency_ms, 1)
        return d


def build_classifier(cfg: RouterConfig) -> Classifier:
    if cfg.backend == "lev-local":
        from .classifiers.lev_local import LevLocalClassifier
        from .setup_check import validate_setup

        validate_setup()  # once per process: shows the install guide if lev is missing

        return LevLocalClassifier(cfg.lev_checkpoint, cfg.timeout_s, cfg.block_on_load)
    if cfg.backend == "lev-http":
        from .classifiers.lev_http import LevHttpClassifier

        return LevHttpClassifier(
            cfg.lev_url, cfg.timeout_s, cfg.lev_api_key, cfg.allow_remote_lev, cfg.redact_classifier_input,
            cfg.lev_headers,
        )
    return HeuristicClassifier()


def _blended(model: str) -> float | None:
    """Price for comparing models: input + 3x output per 1M tokens (answers are usually shorter than prompts)."""
    price = price_of(model)
    return price[0] + 3 * price[1] if price else None


def tier_from_signals(s: Signals, cfg: RouterConfig) -> tuple[Tier, str]:
    p = s.tier
    complex_score = p["complex"] + 0.25 * s.needs_reasoning + 0.15 * s.needs_code
    if complex_score >= cfg.complex_threshold:
        tier: Tier = "complex"
        why = f"complex score {complex_score:.2f} >= {cfg.complex_threshold}"
    elif p["simple"] >= cfg.simple_threshold and s.needs_reasoning < 0.5:
        tier, why = "simple", f"P(simple) {p['simple']:.2f} >= {cfg.simple_threshold}"
    else:
        tier, why = "standard", f"P(simple) {p['simple']:.2f}, complex score {complex_score:.2f}"
    if tier != "complex" and max(p.values()) < cfg.min_confidence:
        tier = TIERS[TIERS.index(tier) + 1]
        why += f"; low confidence ({max(p.values()):.2f}) -> bumped to {tier}"
    return tier, why


class Router:
    def __init__(self, config: RouterConfig | None = None, classifier: Classifier | None = None):
        self.config = config or RouterConfig()
        self.classifier = classifier or build_classifier(self.config)
        self._fallback = HeuristicClassifier()
        self._cache: OrderedDict[str, Signals] = OrderedDict()
        self._lock = threading.Lock()
        self._warned: set[str] = set()
        # thread id -> (tier, expiry on time.monotonic()), newest last
        self._threads: OrderedDict[str, tuple[Tier, float]] = OrderedDict()

    # -- signals ---------------------------------------------------------------

    def _signals(self, features: RequestFeatures) -> tuple[Signals, bool, str]:
        state = features.state(self.config.max_state_chars)
        # Cache by digest so the in-memory cache never holds prompt text.
        key = hashlib.sha256(state.encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit, True, ""
        note = ""
        try:
            signals = self.classifier.classify(features, state)
        except Exception as exc:  # noqa: BLE001 - any classifier failure falls back
            kind = "timeout" if isinstance(exc, (TimeoutError, FutureTimeout)) else type(exc).__name__
            if kind not in self._warned:
                self._warned.add(kind)
                log.warning("tokentriage: %s classifier failed (%s: %s); using heuristic", self.classifier.name, kind, exc)
            signals = self._fallback.classify(features, state)
            note = f"fallback after {self.classifier.name} {kind}; "
            # Do not cache fallback answers: lev should get the next identical request.
            return signals, False, note
        with self._lock:
            self._cache[key] = signals
            if len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)
        return signals, False, note

    # -- decision --------------------------------------------------------------

    def tiers_for(self, provider: str, current_model: str, features: RequestFeatures) -> tuple[dict | None, str]:
        """Tier -> model map for this call, and a note. OpenRouter resolves it per call from the vendor."""
        cfg = self.config
        if provider != "openrouter":
            tiers = cfg.tiers.get(provider)
            return tiers, "" if tiers else "no tier map for provider"
        vendor = openrouter_vendor(current_model)
        families = cfg.openrouter_families
        if cfg.openrouter_mode == "family":
            if vendor not in families:
                return None, f"no OpenRouter family for vendor {vendor!r}"
            return families[vendor], ""
        # ladder: for each tier, the cheapest allowed vendor model that can serve this request
        allowed = [v for v in (cfg.openrouter_vendors or tuple(families)) if v in families]
        if not allowed:
            return None, "openrouter_vendors allows no known vendor"
        need_ctx = int(features.total_chars / 3.5)
        tiers = {}
        for tier in TIERS:
            best = None
            for v in allowed:
                model = families[v][tier]
                spec = spec_for("openrouter", model)
                if spec is None or self._problems(spec, features, need_ctx):
                    continue
                price = spec.input_price + spec.output_price
                if best is None or price < best[0]:
                    best = (price, model)
            fallback = families.get(vendor, families[allowed[0]])[tier]
            tiers[tier] = best[1] if best else fallback
        return tiers, ""

    def decide(
        self,
        provider: str,
        current_model: str,
        features: RequestFeatures,
        override_tier: str | None = None,
        max_output_tokens: int | None = None,
        thread: str | None = None,
    ) -> Decision:
        cfg = self.config
        sticky = bool(thread) and cfg.sticky_threads
        tiers, note = self.tiers_for(provider, current_model, features)
        if not tiers:
            return Decision(provider, "standard", current_model, current_model, note)

        if override_tier in TIERS:
            tier, why, signals, cached = override_tier, "override via metadata", None, False
        else:
            signals, cached, note = self._signals(features)
            tier, why = tier_from_signals(signals, cfg)
            why = note + why

        if cfg.min_tier and TIERS.index(tier) < TIERS.index(cfg.min_tier):
            why += f"; raised to min_tier {cfg.min_tier}"
            tier = cfg.min_tier

        if sticky and override_tier is None:
            earlier = self._thread_tier(thread)
            if earlier and TIERS.index(earlier) > TIERS.index(tier):
                why += f"; sticky thread: kept {earlier}"
                tier = earlier

        if cfg.escalate_after_repeats and override_tier is None:
            repeats = repeat_count(features)
            if repeats >= cfg.escalate_after_repeats:
                steps = repeats - cfg.escalate_after_repeats + 1
                higher = TIERS[min(TIERS.index(tier) + steps, len(TIERS) - 1)]
                if higher != tier:
                    why += f"; asked {repeats + 1} times: moved up to {higher}"
                    tier = higher

        model = tiers[tier]
        if cfg.capability_guard:
            tier, model, guard_note = self._fit(provider, tiers, tier, current_model, features, max_output_tokens)
            why += guard_note

        if not cfg.allow_upgrade and override_tier is None:
            # Never pick a model pricier than the one configured: step down the tiers, else keep it.
            ceiling = _blended(current_model)
            if ceiling is not None and (_blended(model) or 0) > ceiling:
                cheaper = [t for t in reversed(TIERS[: TIERS.index(tier)]) if (_blended(tiers[t]) or 0) <= ceiling]
                if cheaper:
                    why += f"; capped: {model} costs more than configured, using {cheaper[0]}"
                    tier, model = cheaper[0], tiers[cheaper[0]]
                else:
                    why += f"; capped: {model} costs more than configured, kept configured model"
                    model = current_model
        if same_model(provider, model, current_model):
            model = current_model  # same model: keep the exact id the app pinned (dated, or OpenRouter bare name)
        if provider == "openrouter" and openrouter_vendor(model) != openrouter_vendor(current_model):
            why += f"; ladder: {openrouter_vendor(current_model)} -> {openrouter_vendor(model)}"
        if sticky:
            self._remember_thread(thread, tier)
        return Decision(provider, tier, model, current_model, why, signals, cached)

    # -- threads ---------------------------------------------------------------

    _MAX_THREADS = 10_000

    def _thread_tier(self, thread: str) -> Tier | None:
        with self._lock:
            entry = self._threads.get(thread)
            if entry is None:
                return None
            if time.monotonic() >= entry[1]:
                del self._threads[thread]
                return None
            return entry[0]

    def forget_thread(self, thread: str) -> None:
        """The next call in this thread is decided fresh."""
        with self._lock:
            self._threads.pop(thread, None)

    def _remember_thread(self, thread: str, tier: Tier) -> None:
        with self._lock:
            self._threads[thread] = (tier, time.monotonic() + self.config.thread_ttl_s)
            self._threads.move_to_end(thread)
            while len(self._threads) > self._MAX_THREADS:
                self._threads.popitem(last=False)

    @staticmethod
    def _problems(spec, features: RequestFeatures, need_ctx: int) -> list[str]:
        problems = []
        for kind, count in features.attachments.items():
            if count and kind not in spec.inputs:
                problems.append(f"no {kind} input")
        if features.tool_count and not spec.tools:
            problems.append("no tool calling")
        if features.forced_tool and not spec.forced_tools:
            problems.append("rejects forced tool calls (structured output)")
        if spec.context and need_ctx > spec.context:
            problems.append(f"~{need_ctx:,} tokens > {spec.context:,} context")
        return problems

    def _fit(
        self, provider: str, tiers: dict, tier: Tier, current_model: str, features: RequestFeatures,
        max_output_tokens: int | None,
    ) -> tuple[Tier, str, str]:
        """Move up the ladder until the model can actually serve this request.

        Unknown models (custom tier ids) pass: there is nothing to check them against.
        """
        # ~3.5 characters per token is conservative for English and code.
        need_ctx = int(features.total_chars / 3.5) + (max_output_tokens or 0)
        start = TIERS.index(tier)
        # Prefer more capable tiers; if none fits (e.g. the top tier rejects forced tool calls), try cheaper ones.
        order = list(TIERS[start:]) + list(reversed(TIERS[:start]))
        first_problem = last = ""
        for t in order:
            spec = spec_for(provider, tiers[t])
            problems = self._problems(spec, features, need_ctx) if spec is not None else []
            if not problems:
                note = f"; {tier} model unfit ({first_problem}), moved to {t}" if t != tier else ""
                return t, tiers[t], note
            last = ", ".join(problems)
            first_problem = first_problem or last
        return tier, current_model, f"; no tier model fits ({last}), kept configured model"
