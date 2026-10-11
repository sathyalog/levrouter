"""lev-local wiring with a fake engine in place of the 4B model."""

import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from levrouter import Router, RouterConfig
from levrouter.classifiers.base import LEV_QUESTIONS
from levrouter.classifiers.lev_local import LevLocalClassifier
from levrouter.features import extract


class FakeEngine:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.seen = []

    def system_one(self, state, questions):
        self.seen.append((state, questions))
        time.sleep(self.delay)
        return SimpleNamespace(
            answers={
                "tier": SimpleNamespace(probabilities={"simple": 0.05, "standard": 0.1, "complex": 0.85}),
                "needs_reasoning": SimpleNamespace(noul=0.9),
                "needs_code": SimpleNamespace(noul=0.7),
            }
        )


@pytest.fixture(autouse=True)
def reset_engine():
    yield
    LevLocalClassifier._engine = None
    LevLocalClassifier._load_error = None
    LevLocalClassifier._loader = None


def _decide(clf, text="Prove the scheduler is deadlock-free"):
    return Router(RouterConfig(), clf).decide("anthropic", "claude-sonnet-5", extract([HumanMessage(text)]))


def test_uses_loaded_engine():
    engine = FakeEngine()
    LevLocalClassifier._engine = engine
    d = _decide(LevLocalClassifier())
    assert d.model == "claude-opus-5-5" and d.signals.source == "lev-local"
    state, questions = engine.seen[0]
    assert "deadlock-free" in state and questions is LEV_QUESTIONS


def test_slow_engine_times_out_to_heuristic():
    LevLocalClassifier._engine = FakeEngine(delay=0.5)
    d = _decide(LevLocalClassifier(timeout_s=0.05))
    assert d.signals.source == "heuristic" and "timeout" in d.reason


def test_load_failure_falls_back():
    LevLocalClassifier._load_error = ImportError("no torch")
    d = _decide(LevLocalClassifier(), "What is 2+2?")
    assert d.signals.source == "heuristic" and d.model == "claude-haiku-4-5"


def test_calls_after_a_timeout_do_not_queue_behind_the_stuck_decision():
    engine = FakeEngine(delay=0.6)
    LevLocalClassifier._engine = engine
    clf = LevLocalClassifier(timeout_s=0.05)
    first = _decide(clf, "first question")
    assert "timeout" in first.reason
    start = time.perf_counter()
    second = _decide(clf, "second question")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, "the second call must not wait for the stuck job"
    assert second.signals.source == "heuristic" and "LevBusy" in second.reason
    assert len(engine.seen) == 1, "no second job was queued"


def test_lev_is_used_again_once_the_stuck_decision_finishes():
    engine = FakeEngine(delay=0.15)
    LevLocalClassifier._engine = engine
    clf = LevLocalClassifier(timeout_s=0.05)
    _decide(clf, "first question")  # times out
    time.sleep(0.3)  # the abandoned job completes
    engine.delay = 0.0
    third = _decide(clf, "third question")
    assert third.signals.source == "lev-local"


def test_concurrent_callers_still_queue_normally_when_nothing_timed_out():
    import threading

    engine = FakeEngine(delay=0.05)
    LevLocalClassifier._engine = engine
    clf = LevLocalClassifier(timeout_s=2.0)
    sources = []
    threads = [threading.Thread(target=lambda i=i: sources.append(_decide(clf, f"q{i}").signals.source)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sources == ["lev-local"] * 4
