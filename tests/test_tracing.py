"""What a host application sees in its logs."""

import json
import logging

import pytest
from conftest import SIMPLE, FixedClassifier
from langchain_anthropic import ChatAnthropic
from test_langchain_patch import fake_network  # noqa: F401 - fixture

import levrouter
from levrouter import Router, RouterConfig


def _enable(**cfg):
    levrouter.enable(router=Router(RouterConfig(log_level=None, summary_at_exit=False, **cfg), FixedClassifier(SIMPLE)))


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "levrouter"]


def test_route_and_done_lines(fake_network, caplog):
    _enable()
    llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")
    with caplog.at_level(logging.INFO, logger="levrouter"):
        llm.invoke("What is the capital of France?", config={"metadata": {"levrouter_task": "geo_lookup"}})
    route, done = _lines(caplog)
    call_id = route.split("[")[1].split("]")[0]
    assert route.startswith("route [") and "task=geo_lookup" in route
    assert '"What is the capital of France?"' in route
    assert "-> claude-haiku-4-5 (simple; configured claude-opus-5-5) via fixed" in route
    assert done.startswith(f"done  [{call_id}] claude-haiku-4-5 in=1000 out=500")
    assert "$0.0035 (saved $0.0105 vs claude-opus-5-5)" in done


def test_langgraph_node_becomes_task(fake_network, caplog):
    _enable()
    with caplog.at_level(logging.INFO, logger="levrouter"):
        ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("hi", config={"metadata": {"langgraph_node": "triage"}})
    assert "task=triage" in _lines(caplog)[0]
    assert levrouter.stats()["by_task"] == {"triage": {"claude-haiku-4-5": 1}}


def test_prompt_hidden_when_disabled(fake_network, caplog):
    _enable(log_prompt_chars=0)
    with caplog.at_level(logging.INFO, logger="levrouter"):
        ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("secret customer data")
    assert "secret" not in " ".join(_lines(caplog))


def test_failure_is_logged_and_reraised(fake_network, monkeypatch, caplog):
    _enable()

    def boom(self, *a, **k):
        raise RuntimeError("rate limited")

    # Patch below the router: enable() already wrapped _generate, so replace the original it calls.
    levrouter.disable()
    monkeypatch.setattr(ChatAnthropic, "_generate", boom)
    _enable()
    with caplog.at_level(logging.INFO, logger="levrouter"), pytest.raises(RuntimeError):
        ChatAnthropic(model="claude-opus-5-5", api_key="test", max_retries=0).invoke("hi")
    fail = _lines(caplog)[-1]
    assert fail.startswith("fail  [") and "claude-haiku-4-5" in fail and "rate limited" in fail
    assert levrouter.stats()["errors"] == 1


def test_stream_logged_when_consumed(fake_network, caplog):
    _enable()
    with caplog.at_level(logging.INFO, logger="levrouter"):
        list(ChatAnthropic(model="claude-opus-5-5", api_key="test").stream("hi"))
    assert _lines(caplog)[-1].startswith("done  [")


def test_json_format_and_jsonl_file(fake_network, caplog, tmp_path):
    path = tmp_path / "calls.jsonl"
    _enable(log_format="json", log_path=str(path))
    with caplog.at_level(logging.INFO, logger="levrouter"):
        ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("hi")
    events = [json.loads(line) for line in _lines(caplog)]
    assert [e["event"] for e in events] == ["route", "done"]
    row = json.loads(path.read_text().splitlines()[0])
    assert row["model"] == "claude-haiku-4-5" and row["original_model"] == "claude-opus-5-5"
    assert row["call_id"] == events[0]["call_id"] and row["cost_usd"] == 0.0035


def test_default_handler_prints_to_stderr(fake_network, capsys):
    levrouter.enable(router=Router(RouterConfig(summary_at_exit=False), FixedClassifier(SIMPLE)))
    ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("hi")
    err = capsys.readouterr().err
    assert "levrouter INFO  route [" in err and "levrouter INFO  done  [" in err
