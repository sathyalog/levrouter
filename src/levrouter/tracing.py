"""What the host application sees: one log line when a call is routed, one when it finishes,
a summary at exit, and optionally an OpenTelemetry span per call.

    12:01:03 levrouter INFO  route [c7f1a2] task=summarize "Summarize this contract..." -> claude-haiku-4-5 (simple; configured claude-opus-5-5) via lev-local 41ms
    12:01:05 levrouter INFO  done  [c7f1a2] claude-haiku-4-5 in=1830 out=212 1.92s $0.0029 (saved $0.0087 vs claude-opus-5-5)
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import pii
from .config import RouterConfig
from .features import RequestFeatures
from .router import Decision
from .telemetry import Telemetry, cost_usd

log = logging.getLogger("levrouter")

_handler: logging.Handler | None = None
_atexit_registered = False
_telemetry_for_exit: Telemetry | None = None
_json = False
# Receives every finished call for the last-24h usage stores (set by levrouter.enable()).
_usage_sink = None


def set_usage_sink(sink) -> None:
    global _usage_sink
    _usage_sink = sink


def record_eval(entry: dict) -> None:
    """Store an evaluation row next to the usage records (files and live memory)."""
    if _usage_sink is not None:
        _usage_sink.add(entry)


# -- logging setup -----------------------------------------------------------


def setup_logging(cfg: RouterConfig, telemetry: Telemetry) -> None:
    """Make routing visible in the host app's output.

    With cfg.log_level=None nothing is configured and records propagate to the app's
    own logging setup (the "levrouter" logger).
    """
    global _handler, _atexit_registered, _telemetry_for_exit, _json
    _json = cfg.log_format == "json"
    _telemetry_for_exit = telemetry

    if cfg.log_level is not None:
        log.setLevel(cfg.log_level.upper())
        if _handler is None:
            _handler = logging.StreamHandler(sys.stderr)
            log.addHandler(_handler)
            # Our handler prints it; do not print it a second time via the root logger.
            log.propagate = False
        fmt = "%(message)s" if _json else "%(asctime)s levrouter %(levelname)-5s %(message)s"
        _handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    if cfg.summary_at_exit and not _atexit_registered:
        atexit.register(_log_summary)
        _atexit_registered = True


def teardown_logging() -> None:
    global _handler, _telemetry_for_exit
    if _handler is not None:
        log.removeHandler(_handler)
        log.propagate = True
        _handler = None
    _telemetry_for_exit = None


def _log_summary() -> None:
    t = _telemetry_for_exit
    if t is None or t.calls == 0:
        return
    s = t.summary()
    if _json:
        log.info(json.dumps({"event": "summary", **s}))
        return
    models = ", ".join(f"{m} x{n}" for m, n in sorted(s["by_model"].items(), key=lambda kv: -kv[1]))
    log.info(
        "summary %d calls (%s) | cost $%.4f vs $%.4f if unrouted | saved $%.4f (%.1f%%)%s",
        s["calls"], models, s["cost_usd"], s["baseline_cost_usd"], s["saved_usd"], s["saved_pct"],
        f" | {s['errors']} errors" if s["errors"] else "",
    )
    for task, per_model in s["by_task"].items():
        log.info("summary   task=%s -> %s", task, ", ".join(f"{m} x{n}" for m, n in per_model.items()))


# -- per-call trace ----------------------------------------------------------


@dataclass
class CallTrace:
    call_id: str
    decision: Decision
    task: str | None
    prompt: str
    run_id: str | None
    user: str | None
    stream: bool
    redact: bool = True
    started: float = field(default_factory=time.perf_counter)
    span: Any = None


def _task_label(metadata: dict, tags: list[str]) -> str | None:
    for key in ("levrouter_task", "langgraph_node", "task", "ls_run_name"):
        if metadata.get(key):
            return str(metadata[key])
    user_tags = [t for t in tags if not t.startswith(("seq:", "graph:", "langsmith:"))]
    return user_tags[0] if user_tags else None


def _user_label(metadata: dict) -> str | None:
    for key in ("levrouter_user", "user_id"):
        if metadata.get(key):
            return str(metadata[key])
    return None


def _preview(text: str, n: int, redact: bool) -> str:
    if n <= 0:
        return ""
    # Redact before truncating, so a secret cut at the boundary is still caught.
    text = " ".join((pii.redact(text, redact) or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def begin(decision: Decision, features: RequestFeatures, run_manager: Any, cfg: RouterConfig, stream: bool) -> CallTrace:
    metadata = dict(getattr(run_manager, "metadata", None) or {})
    tags = list(getattr(run_manager, "tags", None) or [])
    run_id = getattr(run_manager, "run_id", None)
    trace = CallTrace(
        call_id=uuid.uuid4().hex[:6],
        decision=decision,
        task=pii.redact(_task_label(metadata, tags), cfg.redact_pii),
        prompt=_preview(features.last_user, cfg.log_prompt_chars, cfg.redact_pii),
        run_id=str(run_id) if run_id else None,
        user=pii.redact(_user_label(metadata), cfg.redact_pii),
        stream=stream,
        redact=cfg.redact_pii,
    )
    trace.span = _start_span(trace, cfg)

    if log.isEnabledFor(logging.INFO):
        if _json:
            log.info(json.dumps({"event": "route", "call_id": trace.call_id, "task": trace.task, "user": trace.user,
                                 "prompt": trace.prompt, "run_id": trace.run_id, **decision.as_dict()}))
        else:
            d = decision
            parts = [f"route [{trace.call_id}]"]
            if trace.task:
                parts.append(f"task={trace.task}")
            if trace.user:
                parts.append(f"user={trace.user}")
            if trace.prompt:
                parts.append(f'"{trace.prompt}"')
            changed = f"configured {d.original_model}" if d.model != d.original_model else "unchanged"
            compared = re.match(r"eval: serving configured model; routing would use (\S+);", d.reason or "")
            if compared:
                changed = f"eval: serving configured model, comparing with {compared.group(1)}"
            src = d.signals.source if d.signals else "override"
            ms = f" {d.signals.latency_ms:.0f}ms" if d.signals and not d.cached else (" cached" if d.cached else "")
            parts.append(f"-> {d.model} ({d.tier}; {changed}) via {src}{ms}")
            log.info(" ".join(parts))
    log.debug("route [%s] reason: %s", trace.call_id, decision.reason)
    return trace


def end(trace: CallTrace, usage: dict | None, telemetry: Telemetry | None, error: BaseException | None = None) -> dict:
    d = trace.decision
    usage = usage or {}
    tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    seconds = time.perf_counter() - trace.started
    actual = cost_usd(d.model, tin, tout)
    baseline = cost_usd(d.original_model, tin, tout)

    entry = {
        "ts": time.time(),
        "call_id": trace.call_id,
        "run_id": trace.run_id,
        "task": trace.task,
        "user": trace.user,
        "prompt": trace.prompt,
        "stream": trace.stream,
        **d.as_dict(),
        "input_tokens": tin,
        "output_tokens": tout,
        "latency_s": round(seconds, 3),
        "cost_usd": round(actual, 6) if actual is not None else None,
        "baseline_cost_usd": round(baseline, 6) if baseline is not None else None,
        # Provider errors can echo request details; redact them like prompts.
        "error": pii.redact(f"{type(error).__name__}: {error}", trace.redact) if error else None,
    }
    if telemetry is not None:
        telemetry.record(entry)
    if _usage_sink is not None:
        _usage_sink.add(entry)
    _end_span(trace, entry, error)

    level = logging.WARNING if error else logging.INFO
    if log.isEnabledFor(level):
        if _json:
            log.log(level, json.dumps({"event": "error" if error else "done", **entry}, default=str))
        elif error:
            log.warning("fail  [%s] %s after %.2fs: %s", trace.call_id, d.model, seconds, entry["error"])
        else:
            money = ""
            if actual is not None:
                money = f" ${actual:.4f}"
                if baseline is not None and d.model != d.original_model:
                    diff = baseline - actual
                    word = "saved" if diff >= 0 else "extra"
                    money += f" ({word} ${abs(diff):.4f} vs {d.original_model})"
            log.info("done  [%s] %s in=%d out=%d %.2fs%s", trace.call_id, d.model, tin, tout, seconds, money)
    return entry


# -- OpenTelemetry (optional) ------------------------------------------------


def _start_span(trace: CallTrace, cfg: RouterConfig) -> Any:
    if not cfg.otel:
        return None
    try:
        from opentelemetry import trace as otel
    except ImportError:
        return None
    d = trace.decision
    attrs = {
        "levrouter.call_id": trace.call_id,
        "levrouter.provider": d.provider,
        "levrouter.tier": d.tier,
        "levrouter.model": d.model,
        "levrouter.original_model": d.original_model,
        "levrouter.reason": d.reason,
        "gen_ai.request.model": d.model,
    }
    if trace.task:
        attrs["levrouter.task"] = trace.task
    if trace.user:
        attrs["levrouter.user"] = trace.user
    if d.signals:
        attrs["levrouter.source"] = d.signals.source
        for k, v in d.signals.tier.items():
            attrs[f"levrouter.p_{k}"] = round(v, 4)
    return otel.get_tracer("levrouter").start_span("levrouter.call", attributes=attrs)


def _end_span(trace: CallTrace, entry: dict, error: BaseException | None) -> None:
    span = trace.span
    if span is None:
        return
    span.set_attribute("gen_ai.usage.input_tokens", entry["input_tokens"])
    span.set_attribute("gen_ai.usage.output_tokens", entry["output_tokens"])
    if entry["cost_usd"] is not None:
        span.set_attribute("levrouter.cost_usd", entry["cost_usd"])
    if error is not None:
        # record_exception would copy the raw message and stack; store the redacted text instead.
        span.set_attribute("levrouter.error", entry["error"])
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, entry["error"]))
    span.end()
