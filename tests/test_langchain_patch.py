"""Patch the real ChatAnthropic / ChatOpenAI classes, with their network call replaced by a fake."""

import pytest
from conftest import COMPLEX, SIMPLE, FixedClassifier
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import levrouter
from levrouter import Router, RouterConfig

USAGE = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}


def _model_of(llm):
    return getattr(llm, "model", None) or llm.model_name


def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
    tool_calls = []
    if kwargs.get("tools"):
        name = kwargs["tools"][0].get("name") or kwargs["tools"][0]["function"]["name"]
        tool_calls = [{"name": name, "args": {"answer": "4"}, "id": "call_1"}]
    msg = AIMessage(
        content="" if tool_calls else f"model={_model_of(self)}",
        tool_calls=tool_calls,
        usage_metadata=USAGE,
        response_metadata={"model": _model_of(self), "max_tokens": getattr(self, "max_tokens", None), "thinking": getattr(self, "thinking", None)},
    )
    return ChatResult(generations=[ChatGeneration(message=msg)])


async def _fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
    return _fake_generate(self, messages, stop, run_manager, **kwargs)


def _fake_stream(self, messages, stop=None, run_manager=None, **kwargs):
    for i, piece in enumerate(["model=", _model_of(self)]):
        yield ChatGenerationChunk(message=AIMessageChunk(content=piece, usage_metadata=USAGE if i == 1 else None))


async def _fake_astream(self, messages, stop=None, run_manager=None, **kwargs):
    for chunk in _fake_stream(self, messages, stop, run_manager, **kwargs):
        yield chunk


@pytest.fixture
def fake_network(monkeypatch):
    for cls in (ChatAnthropic, ChatOpenAI):
        monkeypatch.setattr(cls, "_generate", _fake_generate)
        monkeypatch.setattr(cls, "_agenerate", _fake_agenerate)
        monkeypatch.setattr(cls, "_stream", _fake_stream)
        monkeypatch.setattr(cls, "_astream", _fake_astream)


def _router(sig):
    return Router(RouterConfig(), FixedClassifier(sig))


def test_enable_routes_anthropic_without_mutating_instance(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")
    out = llm.invoke("What is 2+2?")
    assert out.content == "model=claude-haiku-4-5"
    assert out.response_metadata["levrouter"]["tier"] == "simple"
    assert llm.model == "claude-opus-5-5"


def test_enable_routes_openai(fake_network):
    levrouter.enable(router=_router(COMPLEX))
    out = ChatOpenAI(model="gpt-6-luna", api_key="test").invoke("Design a compiler")
    assert out.content == "model=gpt-6-astra"


def test_haiku_gets_compatible_settings(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    llm = ChatAnthropic(model="claude-opus-5-5", api_key="test", thinking={"type": "adaptive"})
    meta = llm.invoke("hi").response_metadata
    assert meta["max_tokens"] == 64_000 and meta["thinking"] is None


def test_bind_tools_and_structured_output_stay_routed(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")

    class Answer(BaseModel):
        answer: str

    msg = llm.bind_tools([Answer]).invoke("2+2?")
    assert msg.response_metadata["levrouter"]["model"] == "claude-haiku-4-5"
    assert llm.with_structured_output(Answer).invoke("2+2?") == Answer(answer="4")


def test_stream_and_async(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    llm = ChatAnthropic(model="claude-opus-5-5", api_key="test")
    text = "".join(c.content for c in llm.stream("hi"))
    assert text == "model=claude-haiku-4-5"


async def test_async_paths(fake_network):
    levrouter.enable(router=_router(COMPLEX))
    llm = ChatAnthropic(model="claude-haiku-4-5", api_key="test")
    assert (await llm.ainvoke("hard")).content == "model=claude-opus-5-5"
    chunks = [c.content async for c in llm.astream("hard")]
    assert "".join(chunks) == "model=claude-opus-5-5"


def test_route_opts_in_one_instance_only(fake_network):
    routed = levrouter.route(ChatAnthropic(model="claude-opus-5-5", api_key="test"), router=_router(SIMPLE))
    plain = ChatAnthropic(model="claude-opus-5-5", api_key="test")
    assert routed.invoke("hi").content == "model=claude-haiku-4-5"
    assert plain.invoke("hi").content == "model=claude-opus-5-5"


def test_exclude_and_metadata_controls(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    pinned = levrouter.exclude(ChatAnthropic(model="claude-opus-5-5", api_key="test"))
    assert pinned.invoke("hi").content == "model=claude-opus-5-5"
    llm = ChatAnthropic(model="claude-sonnet-5", api_key="test")
    assert llm.invoke("hi", config={"metadata": {"levrouter_disable": True}}).content == "model=claude-sonnet-5"
    assert llm.invoke("hi", config={"metadata": {"levrouter_tier": "complex"}}).content == "model=claude-opus-5-5"


def test_disable_restores_originals(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    assert ChatAnthropic._generate is not _fake_generate
    levrouter.disable()
    assert ChatAnthropic._generate is _fake_generate
    assert ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("hi").content == "model=claude-opus-5-5"


def test_stats_track_savings(fake_network):
    levrouter.enable(router=_router(SIMPLE))
    ChatAnthropic(model="claude-opus-5-5", api_key="test").invoke("hi")
    s = levrouter.stats()
    # Opus 5.5: 1000*4 + 500*20 = $0.014; Haiku 4.5: 1000*1 + 500*5 = $0.0035
    assert s["calls"] == 1 and s["baseline_cost_usd"] == 0.014 and s["cost_usd"] == 0.0035
    assert s["saved_pct"] == 75.0
