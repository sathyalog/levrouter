"""OpenAI SDK routing with the real `openai` client; only the HTTP transport is faked."""

import json

import httpx2
import pytest
from conftest import SIMPLE, FixedClassifier

import openai
import levrouter
from levrouter import Router, RouterConfig


@pytest.fixture
def sent(monkeypatch):
    """(host, model) pairs the SDK actually sent, in order."""
    calls: list[tuple[str, str]] = []

    def respond(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body["model"]))
        if body.get("stream"):
            return httpx2.Response(200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"},
                                   request=request)
        return httpx2.Response(200, request=request, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "4"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})

    async def respond_async(self, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx2.Client, "send", lambda self, request, **kw: respond(request))
    monkeypatch.setattr(httpx2.AsyncClient, "send", respond_async)
    return calls


def _enable(**cfg):
    levrouter.enable(router=Router(RouterConfig(**cfg), FixedClassifier(SIMPLE)), frameworks=("openai",))


def _ask(client, model="gpt-6-astra", **kw):
    return client.chat.completions.create(model=model, messages=[{"role": "user", "content": "What is 2+2?"}], **kw)


def test_sync_create_is_routed_and_counted(sent):
    _enable()
    reply = _ask(openai.OpenAI(api_key="sk-test"))
    assert sent == [("api.openai.com", "gpt-6-luna")] and reply.model == "gpt-6-luna"
    stats = levrouter.stats()
    assert stats["calls"] == 1 and stats["by_model"] == {"gpt-6-luna": 1}


async def test_async_create_is_routed(sent):
    _enable()
    reply = await openai.AsyncOpenAI(api_key="sk-test").chat.completions.create(
        model="gpt-6-astra", messages=[{"role": "user", "content": "What is 2+2?"}])
    assert sent == [("api.openai.com", "gpt-6-luna")] and reply.model == "gpt-6-luna"


def test_stream_is_routed(sent):
    _enable()
    _ask(openai.OpenAI(api_key="sk-test"), stream=True)
    assert sent == [("api.openai.com", "gpt-6-luna")]


def test_openrouter_endpoint_routes_within_the_vendor(sent):
    _enable()
    client = openai.OpenAI(api_key="sk-or-test", base_url="https://openrouter.ai/api/v1")
    _ask(client, model="anthropic/claude-opus-5.5")
    assert sent == [("openrouter.ai", "anthropic/claude-haiku-4.5")]


def test_unknown_host_and_disable_leave_calls_alone(sent):
    _enable()
    _ask(openai.OpenAI(api_key="sk-test", base_url="https://llm-proxy.example.com/v1"))
    levrouter.disable()
    assert not getattr(openai.resources.chat.completions.Completions.create, "__levrouter__", False)
    _ask(openai.OpenAI(api_key="sk-test"))
    assert [m for _, m in sent] == ["gpt-6-astra", "gpt-6-astra"]


def test_langchain_call_is_not_routed_twice(sent):
    from langchain_openai import ChatOpenAI

    levrouter.enable(router=Router(RouterConfig(), FixedClassifier(SIMPLE)), frameworks=("langchain", "openai"))
    ChatOpenAI(model="gpt-6-astra", api_key="sk-test").invoke("What is 2+2?")
    assert [m for _, m in sent] == ["gpt-6-luna"] and levrouter.stats()["calls"] == 1


def test_router_bug_falls_back_to_the_configured_model(sent, monkeypatch):
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))

    def boom(*args, **kwargs):
        raise RuntimeError("bug inside the router")

    monkeypatch.setattr(router, "decide", boom)
    levrouter.enable(router=router, frameworks=("openai",))
    _ask(openai.OpenAI(api_key="sk-test"))
    assert [m for _, m in sent] == ["gpt-6-astra"]
