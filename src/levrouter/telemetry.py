"""Running cost-vs-baseline tally and the optional JSONL call log."""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from pathlib import Path

from .providers import price_of


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_of(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


class Telemetry:
    def __init__(self, log_path: str | None = None):
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.errors = 0
        self.by_model: Counter[str] = Counter()
        self.by_tier: Counter[str] = Counter()
        self.by_task: dict[str, Counter[str]] = {}
        self.cost = 0.0
        self.baseline_cost = 0.0

    def record(self, entry: dict) -> None:
        """Add one finished call (the dict tracing.end builds) to the tally and the JSONL log."""
        with self._lock:
            self.calls += 1
            if entry.get("error"):
                self.errors += 1
            self.by_model[entry["model"]] += 1
            self.by_tier[entry["tier"]] += 1
            if entry.get("task"):
                self.by_task.setdefault(entry["task"], Counter())[entry["model"]] += 1
            if entry.get("cost_usd") is not None and entry.get("baseline_cost_usd") is not None:
                self.cost += entry["cost_usd"]
                self.baseline_cost += entry["baseline_cost_usd"]
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                # Owner read/write only: the file holds prompt previews and usage.
                fd = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")

    def summary(self) -> dict:
        with self._lock:
            saved = self.baseline_cost - self.cost
            return {
                "calls": self.calls,
                "errors": self.errors,
                "by_model": dict(self.by_model),
                "by_tier": dict(self.by_tier),
                "by_task": {task: dict(models) for task, models in self.by_task.items()},
                "cost_usd": round(self.cost, 6),
                "baseline_cost_usd": round(self.baseline_cost, 6),
                "saved_usd": round(saved, 6),
                "saved_pct": round(100 * saved / self.baseline_cost, 1) if self.baseline_cost else 0.0,
            }
