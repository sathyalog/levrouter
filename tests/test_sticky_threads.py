"""One conversation stays on one model (moving up only), so the provider's prompt cache keeps working."""

from conftest import COMPLEX, SIMPLE, FixedClassifier
from langchain_core.messages import HumanMessage

import levrouter
from levrouter import Router, RouterConfig
from levrouter import router as router_module
from levrouter.features import extract


def _router(**cfg):
    clf = FixedClassifier(COMPLEX)
    return Router(RouterConfig(**cfg), clf), clf


def _decide(router, text, thread):
    return router.decide("anthropic", "claude-sonnet-5", extract([HumanMessage(text)]), thread=thread)


def test_thread_keeps_its_model_after_an_easy_turn():
    router, clf = _router()
    assert _decide(router, "Design the schema", "t1").model == "claude-opus-5-5"
    clf.signals = SIMPLE
    d = _decide(router, "thanks!", "t1")
    assert d.model == "claude-opus-5-5" and "sticky thread: kept complex" in d.reason


def test_thread_can_move_up():
    router, clf = _router()
    clf.signals = SIMPLE
    assert _decide(router, "hi", "t1").model == "claude-haiku-4-5"
    clf.signals = COMPLEX
    assert _decide(router, "now prove it", "t1").model == "claude-opus-5-5"
    clf.signals = SIMPLE
    assert _decide(router, "ok", "t1").model == "claude-opus-5-5"


def test_threads_are_independent_and_unthreaded_calls_route_freely():
    router, clf = _router()
    _decide(router, "Design the schema", "t1")
    clf.signals = SIMPLE
    assert _decide(router, "hi", "t2").model == "claude-haiku-4-5"
    assert _decide(router, "hi", None).model == "claude-haiku-4-5"


def test_expired_thread_is_forgotten(monkeypatch):
    router, clf = _router(thread_ttl_s=60)
    now = [1000.0]
    monkeypatch.setattr(router_module.time, "monotonic", lambda: now[0])
    _decide(router, "Design the schema", "t1")
    now[0] += 61
    clf.signals = SIMPLE
    assert _decide(router, "hi", "t1").model == "claude-haiku-4-5"


def test_sticky_threads_can_be_turned_off():
    router, clf = _router(sticky_threads=False)
    _decide(router, "Design the schema", "t1")
    clf.signals = SIMPLE
    assert _decide(router, "thanks!", "t1").model == "claude-haiku-4-5"


def test_langchain_thread_metadata(monkeypatch):
    from langchain_anthropic import ChatAnthropic
    from test_langchain_patch import _fake_generate

    monkeypatch.setattr(ChatAnthropic, "_generate", _fake_generate)
    clf = FixedClassifier(COMPLEX)
    levrouter.enable(router=Router(RouterConfig(), clf))
    llm = ChatAnthropic(model="claude-sonnet-5", api_key="sk-ant-test")
    conv = {"metadata": {"levrouter_thread": "conv-1"}}
    assert llm.invoke("Design the schema", config=conv).response_metadata["levrouter"]["model"] == "claude-opus-5-5"
    clf.signals = SIMPLE
    assert llm.invoke("thanks!", config=conv).response_metadata["levrouter"]["model"] == "claude-opus-5-5"
    assert llm.invoke("thanks!").response_metadata["levrouter"]["model"] == "claude-haiku-4-5"
