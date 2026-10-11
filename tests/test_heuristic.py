from langchain_core.messages import HumanMessage, SystemMessage

from levrouter import Router, RouterConfig
from levrouter.classifiers.heuristic import HeuristicClassifier
from levrouter.features import extract


def _model(text, tools=None):
    router = Router(RouterConfig(backend="heuristic"), HeuristicClassifier())
    return router.decide("anthropic", "claude-opus-5-5", extract([HumanMessage(text)], tools)).model


def test_simple_prompts_go_to_haiku():
    assert _model("What is the capital of France?") == "claude-haiku-4-5"
    assert _model("Translate 'good morning' to Spanish") == "claude-haiku-4-5"
    assert _model("Fix the grammar: me and him goes to school") == "claude-haiku-4-5"


def test_complex_prompts_go_to_opus():
    assert _model(
        "Design a distributed rate limiter for a multi-region API. Explain the trade-offs "
        "between token bucket and sliding window, handle clock skew, and prove it never "
        "exceeds the global limit."
    ) == "claude-opus-5-5"
    assert _model(
        "Debug this race condition in my Python asyncio worker pool and refactor it:\n"
        "```python\nasync def worker(q):\n    while True:\n        item = await q.get()\n```"
    ) == "claude-opus-5-5"


def test_middle_prompts_go_to_sonnet():
    assert _model(
        "Write a friendly three-paragraph email to my team announcing that the office "
        "will be closed next Friday, and remind them to submit their timesheets."
    ) == "claude-sonnet-5"


def test_many_tools_push_toward_complex():
    tools = [{"name": f"t{i}"} for i in range(6)]
    assert _model("Look into the failing deploy and plan out a fix", tools) == "claude-opus-5-5"


def test_state_includes_system_and_context():
    f = extract([SystemMessage("You are terse."), HumanMessage("hi")])
    state = f.state()
    assert "You are terse." in state and "User request: hi" in state
