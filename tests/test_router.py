from conftest import COMPLEX, SIMPLE, STANDARD, FixedClassifier, signals
from langchain_core.messages import HumanMessage

from tokentriage import Router, RouterConfig
from tokentriage.classifiers.base import signals_from_answers
from tokentriage.features import extract


def _features(text="hello"):
    return extract([HumanMessage(text)])


def test_tiers_map_to_anthropic_models():
    for sig, model in [
        (SIMPLE, "claude-haiku-4-5"),
        (STANDARD, "claude-sonnet-5"),
        (COMPLEX, "claude-opus-5-5"),
    ]:
        router = Router(RouterConfig(), FixedClassifier(sig))
        assert router.decide("anthropic", "claude-opus-5-5", _features()).model == model


def test_tiers_map_to_openai_models():
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))
    assert router.decide("openai", "gpt-6-astra", _features()).model == "gpt-6-luna"


def test_low_confidence_bumps_up_one_tier():
    # Would be "standard", but no option reaches min_confidence, so it goes one tier up.
    uncertain = signals(0.4, 0.35, 0.25, reasoning=0.1, code=0.0)
    d = Router(RouterConfig(), FixedClassifier(uncertain)).decide("anthropic", "x", _features())
    assert d.tier == "complex" and "bumped" in d.reason


def test_override_and_upgrade_cap():
    router = Router(RouterConfig(), FixedClassifier(COMPLEX))
    assert router.decide("anthropic", "claude-opus-5-5", _features(), "simple").model == "claude-haiku-4-5"
    capped = Router(RouterConfig(allow_upgrade=False), FixedClassifier(COMPLEX))
    assert capped.decide("anthropic", "claude-sonnet-5", _features()).model == "claude-sonnet-5"


def test_cache_skips_second_classification():
    clf = FixedClassifier(SIMPLE)
    router = Router(RouterConfig(), clf)
    router.decide("anthropic", "x", _features("same prompt"))
    d = router.decide("anthropic", "x", _features("same prompt"))
    assert clf.calls == 1 and d.cached


def test_classifier_failure_falls_back_to_heuristic_and_is_not_cached():
    clf = FixedClassifier(error=TimeoutError())
    router = Router(RouterConfig(), clf)
    d = router.decide("anthropic", "claude-opus-5-5", _features("What is the capital of France?"))
    assert d.signals.source == "heuristic" and "fallback" in d.reason
    router.decide("anthropic", "claude-opus-5-5", _features("What is the capital of France?"))
    assert clf.calls == 2


def test_unknown_provider_keeps_model():
    d = Router(RouterConfig(), FixedClassifier(SIMPLE)).decide("cohere", "command-a", _features())
    assert d.model == "command-a"


def test_signals_from_lev_answer_dicts():
    answers = {
        "tier": {"type": "choice", "choice": "simple", "probabilities": {"simple": 0.8, "standard": 0.15, "complex": 0.05}, "confidence": 0.8},
        "needs_reasoning": {"type": "noul", "noul": 0.1},
        "needs_code": {"type": "noul", "noul": 0.0},
    }
    s = signals_from_answers(answers, "lev-http", 12.0)
    assert s.tier["simple"] == 0.8 and s.needs_reasoning == 0.1


def test_importing_tokentriage_does_not_print_the_lev_install_guide():
    import subprocess
    import sys

    code = "import sys, importlib.util; sys.modules['lev'] = None; import tokentriage"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "PREREQUISITE" not in out.stdout + out.stderr


def test_install_guide_is_shown_only_when_lev_local_is_built(monkeypatch, capsys):
    from tokentriage import setup_check
    from tokentriage.router import build_classifier

    shown = []
    monkeypatch.setattr(setup_check, "_is_lev_installed", lambda: False)
    monkeypatch.setattr(setup_check, "_get_project_root", lambda: None)
    monkeypatch.setattr(setup_check, "_check_pip_list", lambda: False)
    monkeypatch.setattr(setup_check, "show_lev_installation_guide", lambda: shown.append(1))
    monkeypatch.delattr(setup_check.validate_setup, "_checked", raising=False)

    build_classifier(RouterConfig(backend="lev-http", lev_url="http://localhost:8000"))
    build_classifier(RouterConfig(backend="heuristic"))
    assert shown == []
    build_classifier(RouterConfig(backend="lev-local"))
    build_classifier(RouterConfig(backend="lev-local"))
    assert shown == [1], "shown once, only for lev-local"
    monkeypatch.delattr(setup_check.validate_setup, "_checked", raising=False)
