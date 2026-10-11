"""Last-24h token usage and cost: an in-memory window and rolling hourly files.

Hourly files (default ~/.levrouter/usage/, override with LEVROUTER_HOME):

    usage/2026-09-29T09.jsonl   <- oldest hour kept
    ...
    usage/2026-09-30T09.jsonl   <- current UTC hour, appended on every call

Every finished call appends one JSON line to the current hour's file. When a process starts
writing a new hour, and whenever the CLI reads, files older than the retention window are
deleted whole: data is never rewritten, so several workers can append to the same file and
a crash loses at most the line being written. Reads filter by timestamp, so "last 24h" is
exact even though the oldest file straddles the window.

Records hold no prompt text: time, ids, provider/model/tier, task and user labels (already
redacted), token counts, latency, estimated cost and the error class name.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("levrouter")

GROUP_KEYS = ("model", "provider", "tier", "task", "user")
_HOUR_FMT = "%Y-%m-%dT%H"
_HOUR_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2})\.jsonl$")


def home_dir(override: str | None = None) -> Path:
    return Path(override or os.environ.get("LEVROUTER_HOME") or Path.home() / ".levrouter").expanduser()


def _private_dir(path: Path) -> Path:
    """Create path (and a missing levrouter home above it) readable by this user only."""
    for p in (path.parent, path):
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(p, 0o700)
            except OSError:
                pass
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


# -- record ------------------------------------------------------------------


@dataclass(frozen=True)
class UsageRecord:
    ts: float
    call_id: str
    provider: str
    model: str
    original_model: str
    tier: str
    task: str | None
    user: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None
    baseline_cost_usd: float | None
    error_kind: str | None
    # "call" for a served call; "eval" for an evaluation of a call (evaluate.py). For eval rows,
    # model = routed model, original_model = baseline, cost fields = each answer's cost.
    kind: str = "call"
    verdict: str | None = None          # equivalent | routed_better | baseline_better | error
    routed_score: float | None = None   # 1-5
    baseline_score: float | None = None
    eval_method: str | None = None      # judge | tool_calls
    judge_model: str | None = None
    extra_cost_usd: float | None = None  # what the evaluation itself cost (second call + judge)

    @classmethod
    def from_entry(cls, entry: dict) -> UsageRecord:
        """Trim a tracing entry to the usage fields. Prompt previews and error text are dropped."""
        error = entry.get("error")
        return cls(
            ts=float(entry.get("ts") or time.time()),
            call_id=str(entry.get("call_id", "")),
            provider=str(entry.get("provider", "")),
            model=str(entry.get("model", "")),
            original_model=str(entry.get("original_model", "")),
            tier=str(entry.get("tier", "")),
            task=entry.get("task"),
            user=entry.get("user"),
            input_tokens=int(entry.get("input_tokens") or 0),
            output_tokens=int(entry.get("output_tokens") or 0),
            latency_s=float(entry.get("latency_s") or 0.0),
            cost_usd=entry.get("cost_usd"),
            baseline_cost_usd=entry.get("baseline_cost_usd"),
            error_kind=str(error).split(":", 1)[0] if error else None,
            kind=entry.get("kind") or "call",
            verdict=entry.get("verdict"),
            routed_score=entry.get("routed_score"),
            baseline_score=entry.get("baseline_score"),
            eval_method=entry.get("eval_method"),
            judge_model=entry.get("judge_model"),
            extra_cost_usd=entry.get("extra_cost_usd"),
        )

    @classmethod
    def from_dict(cls, data: dict) -> UsageRecord:
        fields = cls.__dataclass_fields__
        values = {k: data.get(k) for k in fields if k in data}
        values.setdefault("kind", "call")
        values["kind"] = values["kind"] or "call"
        for k, f in fields.items():  # rows written before a field existed
            if k not in values:
                values[k] = f.default if f.default is not MISSING else None
        return cls(**values)


# -- stores ------------------------------------------------------------------


class MemoryStore:
    """The last `window_h` hours of records in this process, capped at `max_records`."""

    def __init__(self, window_h: float = 24, max_records: int = 100_000, clock=time.time):
        self.window_s = window_h * 3600
        self._records: deque[UsageRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._clock = clock
        self.started_at = clock()

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._records and self._records[0].ts < cutoff:
            self._records.popleft()

    def add(self, rec: UsageRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._evict(self._clock())

    def query(self, since: float | None = None) -> list[UsageRecord]:
        with self._lock:
            now = self._clock()
            self._evict(now)
            # Filter as well as evict: records can arrive slightly out of order across threads.
            floor = max(since or 0.0, now - self.window_s)
            return [r for r in self._records if r.ts >= floor]

    def __len__(self) -> int:
        return len(self._records)


class HourlyFileStore:
    """Rolling usage files, one per UTC hour, kept for `retention_h` hours."""

    def __init__(self, directory: str | Path, retention_h: int = 24, clock=time.time):
        self.dir = Path(directory)
        self.retention_h = retention_h
        self._clock = clock
        self._current_hour: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def hour_of(ts: float) -> str:
        return datetime.fromtimestamp(ts, timezone.utc).strftime(_HOUR_FMT)

    def files(self) -> list[tuple[float, Path]]:
        """(hour start epoch, path) of every hour file, oldest first."""
        if not self.dir.is_dir():
            return []
        out = []
        for p in self.dir.iterdir():
            m = _HOUR_FILE.match(p.name)
            if m:
                start = datetime.strptime(m.group(1), _HOUR_FMT).replace(tzinfo=timezone.utc).timestamp()
                out.append((start, p))
        return sorted(out)

    def prune(self) -> int:
        """Delete hour files that end before the retention window. Returns how many."""
        cutoff = self._clock() - self.retention_h * 3600
        removed = 0
        for start, path in self.files():
            if start + 3600 <= cutoff:  # the whole hour is outside the window
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass  # another worker pruned it first
        return removed

    def add(self, rec: UsageRecord) -> None:
        if rec.ts < self._clock() - self.retention_h * 3600:
            return  # already outside the window; writing it would only recreate a pruned file
        hour = self.hour_of(rec.ts)
        with self._lock:
            if hour != self._current_hour:
                _private_dir(self.dir)
                self._current_hour = hour
                self.prune()
        line = (json.dumps(asdict(rec), separators=(",", ":")) + "\n").encode()
        fd = os.open(self.dir / f"{hour}.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line)  # one write per line: appends from several processes do not interleave
        finally:
            os.close(fd)

    def read(self, since: float | None = None) -> tuple[list[UsageRecord], int]:
        """Records at or after `since`, and the number of malformed lines skipped."""
        self.prune()
        records, bad = [], 0
        for start, path in self.files():
            if since is not None and start + 3600 <= since:
                continue
            try:
                with path.open() as f:
                    for line in f:
                        try:
                            rec = UsageRecord.from_dict(json.loads(line))
                        except (ValueError, TypeError):
                            bad += 1
                            continue
                        if since is None or rec.ts >= since:
                            records.append(rec)
            except FileNotFoundError:
                continue
        return records, bad


class UsageSink:
    """Where tracing sends each finished call: the memory window and/or the hourly files."""

    def __init__(self, memory: MemoryStore | None, files: HourlyFileStore | None):
        self.memory = memory
        self.files = files
        self._warned = False

    def add(self, entry: dict) -> None:
        try:
            rec = UsageRecord.from_entry(entry)
            if self.memory is not None:
                self.memory.add(rec)
            if self.files is not None:
                self.files.add(rec)
        except Exception:  # noqa: BLE001 - usage accounting must never break the app's call
            if not self._warned:
                self._warned = True
                log.debug("levrouter: could not record usage", exc_info=True)


# -- reporting ---------------------------------------------------------------


def parse_since(value: str | None, max_hours: float | None = None, now: float | None = None) -> tuple[float | None, str]:
    """'24h' / '90m' / '7d' / ISO timestamp -> (epoch, note). Capped at max_hours."""
    now = now if now is not None else time.time()
    if not value:
        return (now - max_hours * 3600, "") if max_hours else (None, "")
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", value)
    if m:
        seconds = float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
        since = now - seconds
    else:
        dt = datetime.fromisoformat(value)
        since = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    note = ""
    if max_hours and since < now - max_hours * 3600:
        since = now - max_hours * 3600
        note = f"only the last {max_hours:g}h are kept; showing that window"
    return since, note


def _empty() -> dict:
    return {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "baseline_cost_usd": 0.0, "latency_s": 0.0}


def _add(bucket: dict, r: UsageRecord) -> None:
    bucket["calls"] += 1
    bucket["errors"] += 1 if r.error_kind else 0
    bucket["input_tokens"] += r.input_tokens
    bucket["output_tokens"] += r.output_tokens
    bucket["latency_s"] += r.latency_s
    if r.cost_usd is not None and r.baseline_cost_usd is not None:
        bucket["cost_usd"] += r.cost_usd
        bucket["baseline_cost_usd"] += r.baseline_cost_usd


def _finish(bucket: dict) -> dict:
    saved = bucket["baseline_cost_usd"] - bucket["cost_usd"]
    out = dict(bucket)
    out["cost_usd"] = round(bucket["cost_usd"], 6)
    out["baseline_cost_usd"] = round(bucket["baseline_cost_usd"], 6)
    out["saved_usd"] = round(saved, 6)
    out["saved_pct"] = round(100 * saved / bucket["baseline_cost_usd"], 1) if bucket["baseline_cost_usd"] else 0.0
    out["avg_latency_s"] = round(bucket["latency_s"] / bucket["calls"], 3) if bucket["calls"] else 0.0
    out.pop("latency_s")
    return out


def aggregate(
    records: Iterable[UsageRecord], by: str = "model", user: str | None = None, task: str | None = None
) -> dict:
    if by not in GROUP_KEYS:
        raise ValueError(f"by must be one of {GROUP_KEYS}")
    total, groups = _empty(), {}
    first = last = None
    for r in records:
        if r.kind != "call":
            continue
        if (user is not None and r.user != user) or (task is not None and r.task != task):
            continue
        _add(total, r)
        _add(groups.setdefault(getattr(r, by) or "-", _empty()), r)
        first = r.ts if first is None else min(first, r.ts)
        last = r.ts if last is None else max(last, r.ts)
    rows = [{by: key, **_finish(b)} for key, b in groups.items()]
    rows.sort(key=lambda row: (-row["cost_usd"], -row["calls"]))
    return {"by": by, "total": _finish(total), "rows": rows, "first_ts": first, "last_ts": last}


def merge(reports: list[dict]) -> dict:
    """Combine aggregate() results (e.g. one per live process)."""
    if not reports:
        return aggregate([])
    by = reports[0]["by"]

    def raw(d: dict) -> dict:
        b = _empty()
        for k in ("calls", "errors", "input_tokens", "output_tokens", "cost_usd", "baseline_cost_usd"):
            b[k] = d[k]
        b["latency_s"] = d["avg_latency_s"] * d["calls"]
        return b

    total, groups = _empty(), {}
    for rep in reports:
        for k, v in raw(rep["total"]).items():
            total[k] += v
        for row in rep["rows"]:
            g = groups.setdefault(row[by], _empty())
            for k, v in raw(row).items():
                g[k] += v
    rows = [{by: key, **_finish(b)} for key, b in groups.items()]
    rows.sort(key=lambda row: (-row["cost_usd"], -row["calls"]))
    firsts = [r["first_ts"] for r in reports if r.get("first_ts")]
    lasts = [r["last_ts"] for r in reports if r.get("last_ts")]
    return {"by": by, "total": _finish(total), "rows": rows,
            "first_ts": min(firsts) if firsts else None, "last_ts": max(lasts) if lasts else None}


HELD = ("equivalent", "routed_better")


def aggregate_eval(
    records: Iterable[UsageRecord], by: str = "tier", user: str | None = None, task: str | None = None
) -> dict:
    """Quality of routing from eval rows: how often the routed answer held up against the baseline."""
    if by not in GROUP_KEYS:
        raise ValueError(f"by must be one of {GROUP_KEYS}")

    def empty() -> dict:
        return {"evaluated": 0, "equivalent": 0, "routed_better": 0, "baseline_better": 0, "errors": 0,
                "routed_score_sum": 0.0, "baseline_score_sum": 0.0, "scored": 0,
                "routed_cost_usd": 0.0, "baseline_cost_usd": 0.0, "extra_cost_usd": 0.0}

    def add(b: dict, r: UsageRecord) -> None:
        if r.verdict == "error":
            b["errors"] += 1
            b["extra_cost_usd"] += r.extra_cost_usd or 0.0
            return
        b["evaluated"] += 1
        if r.verdict in b:
            b[r.verdict] += 1
        if r.routed_score is not None and r.baseline_score is not None:
            b["routed_score_sum"] += r.routed_score
            b["baseline_score_sum"] += r.baseline_score
            b["scored"] += 1
        b["routed_cost_usd"] += r.cost_usd or 0.0
        b["baseline_cost_usd"] += r.baseline_cost_usd or 0.0
        b["extra_cost_usd"] += r.extra_cost_usd or 0.0

    def finish(b: dict) -> dict:
        n, scored = b["evaluated"], b["scored"]
        saved = b["baseline_cost_usd"] - b["routed_cost_usd"]
        return {
            "evaluated": n, "equivalent": b["equivalent"], "routed_better": b["routed_better"],
            "baseline_better": b["baseline_better"], "errors": b["errors"],
            "held_pct": round(100 * (b["equivalent"] + b["routed_better"]) / n, 1) if n else None,
            "avg_routed_score": round(b["routed_score_sum"] / scored, 2) if scored else None,
            "avg_baseline_score": round(b["baseline_score_sum"] / scored, 2) if scored else None,
            "routed_cost_usd": round(b["routed_cost_usd"], 6), "baseline_cost_usd": round(b["baseline_cost_usd"], 6),
            "saved_pct": round(100 * saved / b["baseline_cost_usd"], 1) if b["baseline_cost_usd"] else 0.0,
            "extra_cost_usd": round(b["extra_cost_usd"], 6),
        }

    total, groups = empty(), {}
    for r in records:
        if r.kind != "eval":
            continue
        if (user is not None and r.user != user) or (task is not None and r.task != task):
            continue
        add(total, r)
        add(groups.setdefault(getattr(r, by) or "-", empty()), r)
    rows = [{by: k, **finish(b)} for k, b in groups.items()]
    rows.sort(key=lambda row: -row["evaluated"])
    return {"by": by, "total": finish(total), "rows": rows}


def format_eval_table(report: dict, title: str = "") -> str:
    by, t = report["by"], report["total"]
    if not t["evaluated"] and not t["errors"]:
        return (title + "\n" if title else "") + "no evaluated calls in this window"
    header = [by, "evaluated", "held", "routed better", "baseline better", "score routed/baseline", "saved", "eval cost"]

    def cells(r: dict, name: str) -> list[str]:
        score = (f"{r['avg_routed_score']:.2f} / {r['avg_baseline_score']:.2f}"
                 if r["avg_routed_score"] is not None else "-")
        return [name, str(r["evaluated"]), f"{r['held_pct']:.0f}%" if r["held_pct"] is not None else "-",
                str(r["routed_better"]), str(r["baseline_better"]), score, f"{r['saved_pct']:.1f}%",
                _money(r["extra_cost_usd"])]

    rows = [cells(r, str(r[by])) for r in report["rows"]] + [cells(t, "TOTAL")]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def line(c):
        return "  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(c))

    out = [title] if title else []
    out += [line(header), line(["-" * w for w in widths]), *(line(r) for r in rows[:-1]),
            line(["-" * w for w in widths]), line(rows[-1])]
    out.append("held = judged equivalent or better than the model your code configured; "
               "scores 1-5 from the judge; saved = routed vs configured cost on the evaluated calls")
    if t["errors"]:
        out.append(f"{t['errors']} evaluation(s) failed and are not counted")
    return "\n".join(out)


def _money(v: float) -> str:
    return f"${v:,.4f}"


def format_table(report: dict, title: str = "") -> str:
    by, t = report["by"], report["total"]
    header = [by, "calls", "err", "in tok", "out tok", "cost", "unrouted", "saved", "avg s"]
    rows = [[str(r[by]), str(r["calls"]), str(r["errors"]), f"{r['input_tokens']:,}", f"{r['output_tokens']:,}",
             _money(r["cost_usd"]), _money(r["baseline_cost_usd"]), f"{r['saved_pct']:.1f}%", f"{r['avg_latency_s']:.2f}"]
            for r in report["rows"]]
    rows.append(["TOTAL", str(t["calls"]), str(t["errors"]), f"{t['input_tokens']:,}", f"{t['output_tokens']:,}",
                 _money(t["cost_usd"]), _money(t["baseline_cost_usd"]), f"{t['saved_pct']:.1f}%", f"{t['avg_latency_s']:.2f}"])
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    out = []
    if title:
        out.append(title)
    if report.get("first_ts"):
        fmt = "%Y-%m-%d %H:%M UTC"
        span = (datetime.fromtimestamp(report["first_ts"], timezone.utc).strftime(fmt), "to",
                datetime.fromtimestamp(report["last_ts"], timezone.utc).strftime(fmt))
        out.append("calls from " + " ".join(span))
    out += [line(header), line(["-" * w for w in widths]), *(line(r) for r in rows[:-1]),
            line(["-" * w for w in widths]), line(rows[-1])]
    out.append("cost = estimate from levrouter's price table; 'unrouted' = same tokens on the model your code configured")
    return "\n".join(out)


def format_recent(records: list[UsageRecord], n: int) -> str:
    recent = sorted(records, key=lambda r: r.ts)[-n:]
    lines = [f"{'time (UTC)':<19}  {'call':<6}  {'task':<16}  {'user':<12}  {'model':<24}  {'tier':<8}  "
             f"{'in':>7}  {'out':>7}  {'secs':>5}  {'cost':>9}"]
    for r in recent:
        when = datetime.fromtimestamp(r.ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        cost = _money(r.cost_usd) if r.cost_usd is not None else "-"
        flag = f"  ERROR {r.error_kind}" if r.error_kind else ""
        lines.append(f"{when:<19}  {r.call_id:<6}  {(r.task or '-')[:16]:<16}  {(r.user or '-')[:12]:<12}  "
                     f"{r.model[:24]:<24}  {r.tier:<8}  {r.input_tokens:>7,}  {r.output_tokens:>7,}  "
                     f"{r.latency_s:>5.2f}  {cost:>9}{flag}")
    return "\n".join(lines)


def iter_records(dicts: Iterable[dict]) -> Iterator[UsageRecord]:
    for d in dicts:
        yield UsageRecord.from_dict(d)
