"""Last-24h usage: memory window, rolling hourly files, live socket, CLI."""

import json
import multiprocessing
import os
import time
from datetime import datetime, timezone

import pytest
from conftest import SIMPLE, FixedClassifier
from langchain_anthropic import ChatAnthropic
from test_langchain_patch import fake_network  # noqa: F401 - fixture

import levrouter
from levrouter import Router, RouterConfig, live
from levrouter import usage_store as u
from levrouter.__main__ import main

HOUR = 3600
T0 = datetime(2026, 9, 29, 12, 30, tzinfo=timezone.utc).timestamp()


def rec(ts=T0, model="claude-haiku-4-5", task="faq", user=None, tin=1000, tout=500, cost=0.0035, base=0.014, err=None):
    return u.UsageRecord(ts, "c" + str(int(ts))[-5:], "anthropic", model, "claude-opus-5-5", "simple", task, user,
                         tin, tout, 0.5, cost, base, err)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


# -- memory ---------------------------------------------------------------------


def test_memory_window_and_cap():
    clock = Clock(T0)
    store = u.MemoryStore(window_h=24, max_records=3, clock=clock)
    store.add(rec(T0 - 25 * HOUR))  # already outside the window
    for i in reversed(range(4)):
        store.add(rec(T0 - i))
    assert len(store.query()) == 3  # cap
    clock.now = T0 + 24 * HOUR - 1.5  # the oldest kept record (T0-2) is now 24h+ old
    assert [r.ts for r in store.query()] == [T0 - 1, T0]
    assert [r.ts for r in store.query(since=T0 - 0.5)] == [T0]


# -- hourly files -------------------------------------------------------------


def test_hourly_files_prune_and_exact_window(tmp_path):
    clock = Clock(T0)
    store = u.HourlyFileStore(tmp_path / "usage", retention_h=24, clock=clock)
    for h in range(30):  # one call per hour for 30 hours, ending at T0
        store.add(rec(T0 - h * HOUR))
    names = [p.name for _, p in store.files()]
    # 24 full hours + the current partial hour survive; hours ending before T0-24h are gone
    assert len(names) == 25 and names[-1] == "2026-09-29T12.jsonl" and names[0] == "2026-09-28T12.jsonl"
    assert oct((tmp_path / "usage" / names[-1]).stat().st_mode & 0o777) == "0o600"
    assert oct((tmp_path / "usage").stat().st_mode & 0o777) == "0o700"

    records, bad = store.read(since=T0 - 24 * HOUR)
    assert len(records) == 25 and bad == 0  # T0-24h exactly is included
    records, _ = store.read(since=T0 - 90 * 60)
    assert [r.ts for r in records] == [T0 - HOUR, T0]

    clock.now = T0 + 2 * HOUR  # time passes; reading prunes too
    store.read()
    assert [p.name for _, p in store.files()][0] == "2026-09-28T14.jsonl"


def test_malformed_lines_skipped(tmp_path):
    store = u.HourlyFileStore(tmp_path, clock=Clock(T0))
    store.add(rec())
    with open(tmp_path / "2026-09-29T12.jsonl", "a") as f:
        f.write("{not json\n")
    records, bad = store.read()
    assert len(records) == 1 and bad == 1


def _writer(directory, n, tag):
    store = u.HourlyFileStore(directory)
    for i in range(n):
        store.add(rec(time.time(), task=f"{tag}-{i}" + "x" * 200))


def test_concurrent_writers_do_not_corrupt(tmp_path):
    procs = [multiprocessing.Process(target=_writer, args=(tmp_path, 300, t)) for t in ("a", "b", "c")]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    records, bad = u.HourlyFileStore(tmp_path).read()
    assert bad == 0 and len(records) == 900


# -- aggregation -------------------------------------------------------------


def test_aggregate_and_merge():
    rs = [rec(task="faq"), rec(task="faq"), rec(model="claude-opus-5-5", task="design", cost=0.014, err="RuntimeError")]
    rep = u.aggregate(rs, by="task")
    assert rep["total"]["calls"] == 3 and rep["total"]["errors"] == 1
    assert rep["total"]["input_tokens"] == 3000 and rep["total"]["cost_usd"] == pytest.approx(0.021)
    assert rep["total"]["saved_pct"] == 50.0  # 0.021 spent vs 0.042 unrouted
    assert [r["task"] for r in rep["rows"]] == ["design", "faq"]
    merged = u.merge([u.aggregate(rs[:2], by="task"), u.aggregate(rs[2:], by="task")])
    assert merged["total"] == rep["total"] and merged["rows"] == rep["rows"]
    assert u.aggregate(rs, user="nobody")["total"]["calls"] == 0


def test_parse_since_caps_at_retention():
    since, note = u.parse_since("7d", 24, now=T0)
    assert since == T0 - 24 * HOUR and "24h" in note
    assert u.parse_since("90m", 24, now=T0) == (T0 - 5400, "")


# -- live socket ---------------------------------------------------------------


def test_live_server_answers_and_stale_sockets_removed(tmp_path):
    store = u.MemoryStore()
    store.add(rec(time.time()))
    server = live.LiveServer(store, live.run_dir(tmp_path))
    assert server.start()
    try:
        answers = live.query_all(live.run_dir(tmp_path), {"op": "usage", "by": "model"})
        assert answers[0]["pid"] == os.getpid() and answers[0]["report"]["total"]["calls"] == 1
        assert live.query_all(live.run_dir(tmp_path), {"op": "shutdown"})[0]["error"] == "unknown op"
    finally:
        server.stop()
    stale = live.run_dir(tmp_path) / "999999.sock"  # a dead pid's leftover socket file
    stale.touch()
    assert live.query_all(live.run_dir(tmp_path), {"op": "usage"}) == [] and not stale.exists()


# -- end to end ------------------------------------------------------------------


def _enable():
    levrouter.enable(router=Router(RouterConfig(log_level=None, summary_at_exit=False), FixedClassifier(SIMPLE)))


def test_calls_land_in_memory_files_and_cli(fake_network, capsys):
    _enable()
    llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")
    llm.invoke("secret prompt text", config={"metadata": {"levrouter_task": "faq", "levrouter_user": "u1"}})
    llm.invoke("other", config={"metadata": {"levrouter_task": "faq", "levrouter_user": "jane@acme.com"}})

    mem = levrouter.usage(by="user")
    assert {r["user"] for r in mem["rows"]} == {"u1", "[EMAIL]"}
    assert levrouter.usage(user="u1")["total"]["calls"] == 1
    assert "TOTAL" in levrouter.usage_report(by="task")

    home = u.home_dir()
    raw = "".join(p.read_text() for p in (home / "usage").glob("*.jsonl"))
    assert "secret prompt" not in raw and "jane@acme.com" not in raw and raw.count("\n") == 2

    capsys.readouterr()
    assert main(["usage", "--by", "task", "--json"]) == 0
    from_files = json.loads(capsys.readouterr().out)
    assert main(["usage", "--live", "--by", "task", "--json"]) == 0
    from_live = json.loads(capsys.readouterr().out)
    assert from_files["total"] == from_live["total"] and from_files["total"]["calls"] == 2
    assert from_files["rows"][0]["task"] == "faq"

    assert main(["usage", "--recent", "5"]) == 0
    text = capsys.readouterr().out
    assert "Last 5 calls" in text and "claude-haiku-4-5" in text and "secret" not in text


def test_cli_without_data(capsys, tmp_path):
    assert main(["usage", "--dir", str(tmp_path)]) == 1
    assert "No usage files" in capsys.readouterr().out
    assert main(["usage", "--live", "--dir", str(tmp_path)]) == 1
    assert "No running app" in capsys.readouterr().out


def test_usage_can_be_turned_off(fake_network):
    levrouter.enable(router=Router(RouterConfig(log_level=None, summary_at_exit=False, usage_file=False, usage_live=False),
                                  FixedClassifier(SIMPLE)))
    ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("hi")
    assert not (u.home_dir() / "usage").exists() and not live.run_dir().exists()
    assert levrouter.usage()["total"]["calls"] == 1  # in-process view still works
