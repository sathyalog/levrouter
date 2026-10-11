"""Evaluation mode, batch evaluation, reports and the zero-code launcher (offline: fake models, fake judge)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SIMPLE, FixedClassifier
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

import levrouter
from levrouter import Router, RouterConfig
from levrouter import usage_store as u
from levrouter.__main__ import main
from levrouter.evaluate import (
    Evaluator, combine_judgements, compare_tool_calls, load_prompts, run_batch, sampled,
)

KEY = "sk-or-v1-TESTKEY000000000000000000"
BASE, ROUTED, JUDGE = "anthropic/claude-opus-5.5", "anthropic/claude-haiku-4.5", "anthropic/claude-opus-5.5"
USAGE = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}


class Script:
    """What the fake provider answers: per model, and what the judge says."""

    def __init__(self):
        self.calls = []                      # (model, kind) for every provider call
        self.judge = ['{"winner": "tie", "score_a": 5, "score_b": 5, "reason": "both correct"}'] * 2
        self.fields = {BASE: {"name": "Ana", "years": 5}, ROUTED: {"name": "Ana", "years": 5}}


@pytest.fixture
def fake(monkeypatch):
    script = Script()

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        system = messages[0].content if messages and messages[0].type == "system" else ""
        if "impartial expert evaluator" in str(system):
            script.calls.append((self.model_name, "judge"))
            text = script.judge[sum(1 for _, k in script.calls if k == "judge") - 1]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=USAGE))])
        kind = "tools" if kw.get("tools") else "text"
        script.calls.append((self.model_name, kind))
        if kind == "tools":
            name = kw["tools"][0]["function"]["name"]
            msg = AIMessage(content="", tool_calls=[{"name": name, "args": script.fields[self.model_name], "id": "t1"}],
                            usage_metadata=USAGE)
        else:
            msg = AIMessage(content=f"answer from {self.model_name}", usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        script.calls.append((self.model_name, "stream"))
        for i, part in enumerate(["answer from ", self.model_name]):
            yield ChatGenerationChunk(message=AIMessageChunk(content=part, usage_metadata=USAGE if i else None))

    monkeypatch.setattr(ChatOpenRouter, "_generate", _generate)
    monkeypatch.setattr(ChatOpenRouter, "_stream", _stream)
    return script


def _enable(**cfg):
    base = dict(mode="eval", eval_sample_rate=1.0, log_level=None, summary_at_exit=False)
    base.update(cfg)
    levrouter.enable(router=Router(RouterConfig(**base), FixedClassifier(SIMPLE)))
    return levrouter.integrations.langchain._State.evaluator


def _eval_rows():
    return [r for r in levrouter._UsageState.memory.query() if r.kind == "eval"]


# -- pure functions ----------------------------------------------------------------


def test_sampling_is_deterministic():
    picks = [sampled(f"call{i}", 0.3) for i in range(1000)]
    assert picks == [sampled(f"call{i}", 0.3) for i in range(1000)]
    assert 0.25 < sum(picks) / 1000 < 0.35
    assert sampled("x", 1.0) and not sampled("x", 0.0)


@pytest.mark.parametrize("first, second, verdict", [
    ("A", "B", "routed_better"),     # routed wins in both orders
    ("B", "A", "baseline_better"),
    ("A", "A", "equivalent"),        # position bias: each order prefers slot A -> disagreement
    ("TIE", "TIE", "equivalent"),
])
def test_combine_judgements(first, second, verdict):
    v, rs, bs, _ = combine_judgements({"winner": first, "a": 4, "b": 3, "reason": "r"},
                                      {"winner": second, "a": 3, "b": 4, "reason": "r"})
    assert v == verdict and rs == 4.0 and bs == 3.0


def test_compare_tool_calls():
    same = AIMessage(content="", tool_calls=[{"name": "Profile", "args": {"name": "Ana", "years": 5}, "id": "1"}])
    off = AIMessage(content="", tool_calls=[{"name": "Profile", "args": {"name": "ana ", "years": 4}, "id": "1"}])
    assert compare_tool_calls(same, same)[0] == "equivalent"
    verdict, rs, bs, reason = compare_tool_calls(off, same)
    assert verdict == "baseline_better" and rs == 3.0 and "1/2 fields" in reason


# -- eval mode in an app ---------------------------------------------------------------


def test_serves_baseline_and_records_judged_eval(fake):
    evaluator = _enable()
    out = ChatOpenRouter(model=BASE, api_key=KEY).invoke("What is the capital of France?")
    evaluator.flush()
    assert out.content == f"answer from {BASE}"                       # the app got the configured model
    assert "routing would use anthropic/claude-haiku-4.5" in out.response_metadata["levrouter"]["reason"]
    kinds = [k for _, k in fake.calls]
    assert kinds == ["text", "text", "judge", "judge"]                 # served, other model, judge x2
    assert fake.calls[1][0] == ROUTED and fake.calls[2][0] == JUDGE
    (row,) = _eval_rows()
    assert row.verdict == "equivalent" and row.eval_method == "judge" and row.model == ROUTED
    assert row.extra_cost_usd == pytest.approx(0.0035 + 2 * 0.014)    # haiku call + two opus judge calls


def test_serve_routed_and_regression_detected(fake):
    fake.judge = ['{"winner": "B", "score_a": 2, "score_b": 5, "reason": "routed answer missed a step"}',
                  '{"winner": "A", "score_a": 5, "score_b": 2, "reason": "routed answer missed a step"}']
    evaluator = _enable(eval_serve="routed")
    out = ChatOpenRouter(model=BASE, api_key=KEY).invoke("Explain photosynthesis briefly")
    evaluator.flush()
    assert out.content == f"answer from {ROUTED}"
    (row,) = _eval_rows()
    assert row.verdict == "baseline_better" and row.routed_score == 2.0 and row.baseline_score == 5.0


def test_structured_output_compared_field_by_field(fake):
    class Profile(BaseModel):
        name: str
        years: int

    fake.fields[ROUTED] = {"name": "Ana", "years": 4}
    evaluator = _enable()
    result = ChatOpenRouter(model=BASE, api_key=KEY).with_structured_output(Profile).invoke("Extract: Ana, 5 years")
    evaluator.flush()
    assert result == Profile(name="Ana", years=5)
    (row,) = _eval_rows()
    assert row.eval_method == "tool_calls" and row.verdict == "baseline_better"
    assert "judge" not in [k for _, k in fake.calls]


def test_streaming_is_evaluated_after_the_stream(fake):
    evaluator = _enable()
    text = "".join(c.content for c in ChatOpenRouter(model=BASE, api_key=KEY).stream("hi there"))
    evaluator.flush()
    assert text == f"answer from {BASE}" and len(_eval_rows()) == 1


def test_budget_pauses_evaluation(fake, caplog):
    evaluator = _enable(eval_budget_usd=0.01)
    llm = ChatOpenRouter(model=BASE, api_key=KEY)
    llm.invoke("first question")
    evaluator.flush()
    llm.invoke("second question")
    evaluator.flush()
    assert len(_eval_rows()) == 1                                      # the first eval spent past $0.01
    assert any("budget" in r.getMessage() for r in caplog.records)


def test_no_eval_when_routing_keeps_the_model(fake):
    evaluator = _enable()
    ChatOpenRouter(model=ROUTED, api_key=KEY).invoke("hi")             # already the simple tier
    evaluator.flush()
    assert _eval_rows() == [] and len(fake.calls) == 1


def test_eval_rows_in_files_reports_and_redacted_log(fake, tmp_path, capsys):
    log_path = tmp_path / "eval.jsonl"
    evaluator = _enable(eval_log_path=str(log_path))
    ChatOpenRouter(model=BASE, api_key=KEY).invoke("Email jane@acme.com about the refund",
                                                   config={"metadata": {"levrouter_task": "email"}})
    evaluator.flush()
    rows = json.loads(log_path.read_text().splitlines()[0])
    assert "jane@acme.com" not in json.dumps(rows) and "[EMAIL]" in rows["prompt"]
    assert oct(log_path.stat().st_mode & 0o777) == "0o600"

    assert levrouter.eval_report(by="task")["rows"][0]["task"] == "email"
    capsys.readouterr()
    assert main(["usage", "--by", "task"]) == 0
    out = capsys.readouterr().out
    assert "Quality of routing (evaluated calls)" in out and "100%" in out
    assert main(["eval", "report", "--by", "task", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"]["held_pct"] == 100.0


def test_old_usage_rows_without_eval_fields_still_load():
    rec = u.UsageRecord.from_dict({"ts": 1.0, "call_id": "x", "provider": "p", "model": "m", "original_model": "m",
                                   "tier": "simple", "task": None, "user": None, "input_tokens": 1,
                                   "output_tokens": 1, "latency_s": 0.1, "cost_usd": 0.0,
                                   "baseline_cost_usd": 0.0, "error_kind": None})
    assert rec.kind == "call" and rec.verdict is None


# -- batch ------------------------------------------------------------------------------


def test_run_batch_and_budget_stop(fake, tmp_path):
    prompts_file = tmp_path / "p.jsonl"
    prompts_file.write_text("\n".join(json.dumps({"prompt": f"question {i}", "task": "qa"}) for i in range(10)))
    prompts = load_prompts(str(prompts_file))
    evaluator = _enable(eval_serve="routed")
    result = run_batch(ChatOpenRouter(model=BASE, api_key=KEY), prompts, evaluator, budget_usd=0.08)
    assert result["stopped"] and 1 <= result["prompts"] < 10
    assert len(_eval_rows()) == result["prompts"]


def test_load_prompts_txt(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\nfirst\n\nsecond\n")
    assert [p["prompt"] for p in load_prompts(str(f))] == ["first", "second"]


# -- launcher -------------------------------------------------------------------------


def test_run_launcher_enables_routing_without_code_changes(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import levrouter.integrations.langchain as lc\nprint('enabled', lc._State.route_all)\n")
    other = tmp_path / "site"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import os\nos.environ['CHAINED'] = '1'\n")
    probe_chain = tmp_path / "probe2.py"
    probe_chain.write_text("import os\nprint('chained', os.environ.get('CHAINED'))\n")
    env = {**os.environ, "LEVROUTER_LOG_LEVEL": "WARNING", "PYTHONPATH": str(other)}
    exe = str(Path(sys.executable).with_name("levrouter"))
    out = subprocess.run([exe, "run", "--", sys.executable, str(probe)], env=env, capture_output=True, text=True)
    assert "enabled True" in out.stdout
    out = subprocess.run([exe, "run", "--", sys.executable, str(probe_chain)], env=env, capture_output=True, text=True)
    assert "chained 1" in out.stdout
    plain = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, text=True)
    assert "enabled False" in plain.stdout


def test_eval_route_log_names_the_compared_model(fake, caplog):
    import logging
    evaluator = _enable()
    with caplog.at_level(logging.INFO, logger="levrouter"):
        ChatOpenRouter(model=BASE, api_key=KEY).invoke("What is 2+2?")
        evaluator.flush()
    route = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("route"))
    assert f"eval: serving configured model, comparing with {ROUTED}" in route
