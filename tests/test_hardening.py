"""Production hardening: the data folder and routing bugs must never break the app's own calls."""

import shutil
import tempfile

import pytest
from conftest import SIMPLE, FixedClassifier
from langchain_anthropic import ChatAnthropic
from test_langchain_patch import _fake_generate

import levrouter
from levrouter import Router, RouterConfig


def test_enable_survives_an_unwritable_data_folder(monkeypatch, capsys):
    short = tempfile.mkdtemp(prefix="tt", dir="/tmp")  # short: a long home would move the socket folder elsewhere
    try:
        blocker = f"{short}/a-file"
        open(blocker, "w").write("not a folder")
        monkeypatch.setenv("LEVROUTER_HOME", f"{blocker}/sub")  # a path under a regular file: cannot be created
        levrouter.enable(RouterConfig(backend="heuristic"))
        levrouter.disable()
        levrouter.enable(RouterConfig(backend="heuristic"))
    finally:
        shutil.rmtree(short, ignore_errors=True)
    messages = [line for line in capsys.readouterr().err.splitlines() if "not usable" in line]
    assert messages and "LEVROUTER_HOME" in messages[0]
    assert len(messages) == len(set(messages))  # each problem is reported once


def test_router_bug_falls_back_to_the_configured_model(monkeypatch, capsys):
    monkeypatch.setattr(ChatAnthropic, "_generate", _fake_generate)
    router = Router(RouterConfig(backend="heuristic"), FixedClassifier(SIMPLE))

    def boom(*args, **kwargs):
        raise RuntimeError("bug inside the router")

    monkeypatch.setattr(router, "decide", boom)
    levrouter.enable(router=router)
    llm = ChatAnthropic(model="claude-sonnet-5", api_key="sk-ant-test")
    first, second = llm.invoke("hi"), llm.invoke("hi")
    assert first.content == second.content == "model=claude-sonnet-5"  # the configured model, unrouted
    warnings = [line for line in capsys.readouterr().err.splitlines() if "routing failed" in line]
    assert len(warnings) == 1 and "RuntimeError" in warnings[0] and "ChatAnthropic" in warnings[0]


def test_a_provider_error_still_reaches_the_app(monkeypatch):
    def failing(self, messages, stop=None, run_manager=None, **kwargs):
        raise ValueError("provider said no")

    monkeypatch.setattr(ChatAnthropic, "_generate", failing)
    levrouter.enable(router=Router(RouterConfig(backend="heuristic"), FixedClassifier(SIMPLE)))
    with pytest.raises(ValueError, match="provider said no"):
        ChatAnthropic(model="claude-sonnet-5", api_key="sk-ant-test").invoke("hi")
