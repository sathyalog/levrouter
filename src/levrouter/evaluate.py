"""Evaluation mode: does the cheaper routed model answer as well as the model your code configured?

For a sample of calls (RouterConfig.mode = "eval"), after the app has its answer:
1. the other model is called with the same messages and settings (in a background thread, so the
   app waits for nothing);
2. the two answers are compared:
   - tool calls / structured output: field by field against the configured model's answer;
   - text: a judge model (by default the same provider's top tier) rates both answers, twice with
     the order swapped to cancel position bias;
3. an "eval" row is recorded next to the usage records: verdict, scores, and what the check cost.

Everything stays with the provider your code already calls, unless you explicitly configure a
judge from another provider. Evaluation never changes or delays the answer the app receives.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from . import pii
from .config import RouterConfig
from .features import extract
from .telemetry import cost_usd

log = logging.getLogger("levrouter")

DISABLE = {"metadata": {"levrouter_disable": True}, "tags": ["levrouter-eval"]}
MAX_TEXT = 6000

JUDGE_SYSTEM = (
    "You are an impartial expert evaluator. Compare two answers to the same request. Judge correctness, "
    "completeness, helpfulness and instruction-following; ignore length and style unless they matter to the "
    "request. Reply with JSON only, no prose: "
    '{"winner": "A" | "B" | "tie", "score_a": 1-5, "score_b": 1-5, "reason": "<one sentence>"}. '
    'Use "tie" when both are equally good or equally flawed.'
)


@dataclass
class EvalJob:
    call_id: str
    provider: str
    llm: Any                      # the app's chat model instance (never modified)
    model_field: str
    messages: list[BaseMessage]
    kwargs: dict                  # tools, tool_choice, response_format ... from the original call
    routed_model: str
    baseline_model: str
    served: str                   # "baseline" | "routed": which answer the app received
    served_message: AIMessage
    served_usage: dict = field(default_factory=dict)
    tier: str = ""
    task: str | None = None
    user: str | None = None
    judge_model: str | None = None
    updates: dict = field(default_factory=dict)  # settings the routed model needs (from _compat_updates)
    judge_updates: dict = field(default_factory=dict)  # settings the judge model needs


def sampled(call_id: str, rate: float) -> bool:
    """Deterministic: the same call id is always in or out of the sample."""
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    bucket = int(hashlib.sha256(call_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def _text(msg: BaseMessage | None) -> str:
    if msg is None:
        return ""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _clip(text: str, n: int = MAX_TEXT) -> str:
    return text if len(text) <= n else text[: n // 2] + "\n[...]\n" + text[-n // 2:]


def _usage(msg: AIMessage | None) -> dict:
    return dict(getattr(msg, "usage_metadata", None) or {})


def _cost(model: str, usage: dict) -> float:
    return cost_usd(model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))) or 0.0


# -- comparisons -------------------------------------------------------------------


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, list):
        out = {}
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out or {prefix: []}
    return {prefix: value}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
        return norm(a) == norm(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return a == b


def compare_tool_calls(routed: AIMessage, baseline: AIMessage) -> tuple[str, float, float, str]:
    """Structured output / tool calls: agreement with the configured model's answer, field by field."""
    rc, bc = list(routed.tool_calls or []), list(baseline.tool_calls or [])
    if [c["name"] for c in rc] != [c["name"] for c in bc]:
        return "baseline_better", 1.0, 5.0, f"different tools: {[c['name'] for c in rc]} vs {[c['name'] for c in bc]}"
    fields_b: dict[str, Any] = {}
    fields_r: dict[str, Any] = {}
    for i, (r, b) in enumerate(zip(rc, bc)):
        fields_b.update({f"{i}:{k}": v for k, v in _flatten(b.get("args") or {}).items()})
        fields_r.update({f"{i}:{k}": v for k, v in _flatten(r.get("args") or {}).items()})
    if not fields_b:
        return "equivalent", 5.0, 5.0, "same tool calls"
    agree = sum(1 for k, v in fields_b.items() if k in fields_r and _same(fields_r[k], v))
    ratio = agree / len(fields_b)
    verdict = "equivalent" if ratio >= 0.9 else "baseline_better"
    return verdict, round(1 + 4 * ratio, 2), 5.0, f"{agree}/{len(fields_b)} fields match the configured model"


def _parse_judge(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    winner = str(data.get("winner", "")).strip().upper()
    if winner not in ("A", "B", "TIE"):
        return None
    try:
        return {"winner": winner, "a": float(data.get("score_a")), "b": float(data.get("score_b")),
                "reason": str(data.get("reason", ""))[:300]}
    except (TypeError, ValueError):
        return None


def combine_judgements(first: dict, second: dict) -> tuple[str, float, float, str]:
    """first: A=routed, B=baseline. second: A=baseline, B=routed. Agreement across both orders wins."""
    w1 = {"A": "routed", "B": "baseline", "TIE": "tie"}[first["winner"]]
    w2 = {"A": "baseline", "B": "routed", "TIE": "tie"}[second["winner"]]
    routed_score = (first["a"] + second["b"]) / 2
    baseline_score = (first["b"] + second["a"]) / 2
    if w1 == w2 and w1 != "tie":
        verdict = f"{w1}_better"
    else:
        verdict = "equivalent"
    return verdict, round(routed_score, 2), round(baseline_score, 2), first["reason"] or second["reason"]


# -- the evaluator -----------------------------------------------------------------


class Evaluator:
    def __init__(self, config: RouterConfig, record, max_workers: int = 2):
        self.config = config
        self._record = record            # callable(entry: dict) -> None
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="levrouter-eval")
        self._spent: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._pending: set[Future] = set()
        self._budget_logged = False

    # budget over a rolling 24 hours
    def spent_24h(self) -> float:
        cutoff = time.time() - 86400
        with self._lock:
            while self._spent and self._spent[0][0] < cutoff:
                self._spent.popleft()
            return sum(v for _, v in self._spent)

    def over_budget(self) -> bool:
        over = self.spent_24h() >= self.config.eval_budget_usd
        if over and not self._budget_logged:
            self._budget_logged = True
            log.warning("eval  budget of $%.2f per 24h reached; evaluation paused, serving continues",
                        self.config.eval_budget_usd)
        return over

    def should_evaluate(self, call_id: str, routed: str, baseline: str) -> bool:
        return routed != baseline and sampled(call_id, self.config.eval_sample_rate) and not self.over_budget()

    def submit(self, job: EvalJob) -> None:
        future = self._pool.submit(self._run, job)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(lambda f: self._pending.discard(f))

    def flush(self, timeout: float | None = None) -> None:
        """Wait for evaluations in progress (used by batch runs and tests)."""
        for f in list(self._pending):
            f.result(timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- one evaluation --------------------------------------------------------
    def _copy(self, job: EvalJob, model: str, extra: dict | None = None):
        update = {job.model_field: model, **(extra or {})}
        return job.llm.model_copy(update=update)

    def _run(self, job: EvalJob) -> None:
        start = time.perf_counter()
        extra_cost = 0.0
        entry = {
            "ts": time.time(), "kind": "eval", "call_id": job.call_id, "provider": job.provider,
            "model": job.routed_model, "original_model": job.baseline_model, "tier": job.tier,
            "task": job.task, "user": job.user, "input_tokens": 0, "output_tokens": 0,
        }
        try:
            other_model = job.baseline_model if job.served == "routed" else job.routed_model
            updates = job.updates if other_model == job.routed_model else {}
            other = self._copy(job, other_model, updates).invoke(job.messages, config=DISABLE, **job.kwargs)
            other_cost = _cost(other_model, _usage(other))
            extra_cost += other_cost
            served_cost = _cost(job.routed_model if job.served == "routed" else job.baseline_model, job.served_usage)
            routed_msg, baseline_msg = (job.served_message, other) if job.served == "routed" else (other, job.served_message)
            entry["cost_usd"] = round(served_cost if job.served == "routed" else other_cost, 6)
            entry["baseline_cost_usd"] = round(other_cost if job.served == "routed" else served_cost, 6)

            if routed_msg.tool_calls or baseline_msg.tool_calls:
                verdict, rs, bs, reason = compare_tool_calls(routed_msg, baseline_msg)
                entry.update(eval_method="tool_calls")
            else:
                verdict, rs, bs, reason, judge_cost = self._judge(job, routed_msg, baseline_msg)
                extra_cost += judge_cost
                entry.update(eval_method="judge", judge_model=job.judge_model)
            entry.update(verdict=verdict, routed_score=rs, baseline_score=bs)
            self._write_log(job, routed_msg, baseline_msg, verdict, reason)
            log.info("eval  [%s] %s vs %s -> %s (%.1f vs %.1f) · %s · extra $%.4f", job.call_id, job.routed_model,
                     job.baseline_model, verdict.replace("_", " "), rs, bs, reason[:80], extra_cost)
        except Exception as exc:  # noqa: BLE001 - evaluation must never affect the app
            entry.update(verdict="error", eval_method=entry.get("eval_method"))
            log.warning("eval  [%s] failed: %s", job.call_id, pii.redact(f"{type(exc).__name__}: {exc}"))
        entry["extra_cost_usd"] = round(extra_cost, 6)
        entry["latency_s"] = round(time.perf_counter() - start, 3)
        with self._lock:
            self._spent.append((time.time(), extra_cost))
        self._record(entry)

    def _judge(self, job: EvalJob, routed: AIMessage, baseline: AIMessage) -> tuple[str, float, float, str, float]:
        features = extract(job.messages)
        request = features.last_user
        if features.attachment_summary():
            request += f"\n[The request also included: {features.attachment_summary()}]"
        judge = self._copy(job, job.judge_model, {**job.judge_updates, **self._judge_settings(job)})
        cost = 0.0
        results = []
        for a, b in ((routed, baseline), (baseline, routed)):
            prompt = (f"REQUEST:\n{_clip(request, 4000)}\n\n"
                      f"ANSWER A:\n{_clip(_text(a))}\n\nANSWER B:\n{_clip(_text(b))}")
            reply = judge.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)], config=DISABLE)
            cost += _cost(job.judge_model, _usage(reply))
            parsed = _parse_judge(_text(reply))
            if parsed is None:
                raise ValueError("judge reply was not valid JSON")
            results.append(parsed)
        verdict, rs, bs, reason = combine_judgements(results[0], results[1])
        return verdict, rs, bs, reason, cost

    def _judge_settings(self, job: EvalJob) -> dict:
        """No tools for the judge; enough output room for a short JSON verdict."""
        settings = {}
        for name in ("max_tokens", "max_output_tokens"):
            if name in type(job.llm).model_fields:
                settings[name] = 2048
        return settings

    def _write_log(self, job: EvalJob, routed: AIMessage, baseline: AIMessage, verdict: str, reason: str) -> None:
        path = self.config.eval_log_path
        if not path:
            return
        row = {
            "ts": time.time(), "call_id": job.call_id, "task": job.task, "verdict": verdict, "reason": reason,
            "routed_model": job.routed_model, "baseline_model": job.baseline_model,
            "prompt": pii.redact(_clip(extract(job.messages).last_user, 2000)),
            "routed_answer": pii.redact(_clip(_text(routed) or json.dumps(routed.tool_calls, default=str), 3000)),
            "baseline_answer": pii.redact(_clip(_text(baseline) or json.dumps(baseline.tool_calls, default=str), 3000)),
        }
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(row) + "\n")


# -- batch evaluation: `levrouter eval run prompts.jsonl` ------------------------------


def load_prompts(path: str) -> list[dict]:
    """.txt: one prompt per line. .jsonl: {"prompt": str, "system"?: str, "task"?: str} per line."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if path.endswith(".jsonl"):
                data = json.loads(line)
                rows.append({"prompt": str(data["prompt"]), "system": data.get("system"), "task": data.get("task")})
            else:
                rows.append({"prompt": line, "system": None, "task": None})
    return rows


def make_chat_model(provider: str, model: str):
    """The LangChain chat model for a provider, reading its API key from the environment as usual."""
    import importlib

    from .providers import PROVIDER_CLASSES, TARGETS

    if provider == "huggingface":
        raise ValueError("batch evaluation does not support ChatHuggingFace; use --provider openrouter with a Qwen model")
    target = TARGETS[PROVIDER_CLASSES[provider][0]]
    cls = getattr(importlib.import_module(target.module), target.class_name)
    return cls(model=model)


def run_batch(llm, prompts: list[dict], evaluator: Evaluator, budget_usd: float, on_progress=None) -> dict:
    """Route every prompt (serving the routed answer), evaluate against the configured model, stop at the budget."""
    spent = 0.0
    done = skipped_same_model = 0
    stopped = None
    for i, row in enumerate(prompts, 1):
        if spent >= budget_usd:
            stopped = f"budget ${budget_usd:.2f} reached after {done} prompts"
            break
        messages = ([SystemMessage(row["system"])] if row.get("system") else []) + [HumanMessage(row["prompt"])]
        before = evaluator.spent_24h()
        reply = llm.invoke(messages, config={"metadata": {"levrouter_task": row.get("task") or f"prompt-{i}"}})
        evaluator.flush(timeout=600)
        meta = reply.response_metadata.get("levrouter", {})
        spent += _cost(meta.get("model", ""), _usage(reply)) + (evaluator.spent_24h() - before)
        if meta.get("model") == meta.get("original_model"):
            skipped_same_model += 1
        done += 1
        if on_progress:
            on_progress(i, len(prompts), meta, spent)
    return {"prompts": done, "same_model": skipped_same_model, "spent_usd": round(spent, 6), "stopped": stopped}
