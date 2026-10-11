"""Anthropic SDK routing with the real `anthropic` client; only the HTTP transport is faked."""

import json

import httpx2
import pytest
from conftest import SIMPLE, FixedClassifier

import anthropic
import levrouter
from levrouter import Router, RouterConfig

SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant",'
    '"model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


@pytest.fixture
def sent(monkeypatch):
    """Model ids the SDK actually sent to the API, in order."""
    models: list[str] = []

    def respond(request):
        body = json.loads(request.content)
        models.append(body["model"])
        if body.get("stream"):
            return httpx2.Response(200, text=SSE, headers={"content-type": "text/event-stream"}, request=request)
        return httpx2.Response(200, request=request, json={
            "id": "m1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": "4"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1}})

    async def respond_async(self, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx2.Client, "send", lambda self, request, **kw: respond(request))
    monkeypatch.setattr(httpx2.AsyncClient, "send", respond_async)
    return models


def _enable(**cfg):
    levrouter.enable(router=Router(RouterConfig(**cfg), FixedClassifier(SIMPLE)), frameworks=("anthropic",))


def _ask(client, **kw):
    return client.messages.create(model="claude-opus-5-5", max_tokens=50,
                                  messages=[{"role": "user", "content": "What is 2+2?"}], **kw)


def test_sync_create_is_routed_and_counted(sent):
    _enable()
    reply = _ask(anthropic.Anthropic(api_key="sk-ant-test"))
    assert sent == ["claude-haiku-4-5"] and reply.model == "claude-haiku-4-5"
    stats = levrouter.stats()
    assert stats["calls"] == 1 and stats["by_model"] == {"claude-haiku-4-5": 1}


def test_client_created_before_enable_is_routed(sent):
    client = anthropic.Anthropic(api_key="sk-ant-test")
    _enable()
    _ask(client)
    assert sent == ["claude-haiku-4-5"]


async def test_async_create_is_routed(sent):
    _enable()
    reply = await anthropic.AsyncAnthropic(api_key="sk-ant-test").messages.create(
        model="claude-opus-5-5", max_tokens=50, messages=[{"role": "user", "content": "What is 2+2?"}])
    assert sent == ["claude-haiku-4-5"] and reply.model == "claude-haiku-4-5"


def test_streams_are_routed(sent):
    _enable()
    client = anthropic.Anthropic(api_key="sk-ant-test")
    _ask(client, stream=True)
    with client.messages.stream(model="claude-opus-5-5", max_tokens=50,
                                messages=[{"role": "user", "content": "Hi"}]):
        pass
    assert sent == ["claude-haiku-4-5", "claude-haiku-4-5"]


def test_tool_calls_stay_routable(sent):
    _enable()
    tool = {"name": "add", "description": "add numbers", "input_schema": {"type": "object", "properties": {}}}
    _ask(anthropic.Anthropic(api_key="sk-ant-test"), tools=[tool])
    assert sent == ["claude-haiku-4-5"]


def test_system_prompt_reaches_the_classifier(sent):
    clf = FixedClassifier(SIMPLE)
    levrouter.enable(router=Router(RouterConfig(), clf), frameworks=("anthropic",))
    _ask(anthropic.Anthropic(api_key="sk-ant-test"), system="You are a tax adviser.")
    assert sent == ["claude-haiku-4-5"] and clf.calls == 1


def test_never_route_and_unknown_hosts_are_left_alone(sent):
    _enable(never_route=("claude-opus-*",))
    _ask(anthropic.Anthropic(api_key="sk-ant-test"))
    levrouter.disable()
    _enable()
    _ask(anthropic.Anthropic(api_key="sk-ant-test", base_url="https://llm-proxy.example.com"))
    assert sent == ["claude-opus-5-5", "claude-opus-5-5"]


def test_disable_restores_the_sdk(sent):
    _enable()
    levrouter.disable()
    assert not getattr(anthropic.resources.messages.Messages.create, "__levrouter__", False)
    _ask(anthropic.Anthropic(api_key="sk-ant-test"))
    assert sent == ["claude-opus-5-5"]


def test_langchain_call_is_not_routed_twice(sent):
    from langchain_anthropic import ChatAnthropic

    levrouter.enable(router=Router(RouterConfig(), FixedClassifier(SIMPLE)), frameworks=("langchain", "anthropic"))
    ChatAnthropic(model="claude-opus-5-5", api_key="sk-ant-test", max_tokens=50).invoke("What is 2+2?")
    assert sent == ["claude-haiku-4-5"] and levrouter.stats()["calls"] == 1


def test_router_bug_falls_back_to_the_configured_model(sent, monkeypatch):
    router = Router(RouterConfig(), FixedClassifier(SIMPLE))

    def boom(*args, **kwargs):
        raise RuntimeError("bug inside the router")

    monkeypatch.setattr(router, "decide", boom)
    levrouter.enable(router=router, frameworks=("anthropic",))
    _ask(anthropic.Anthropic(api_key="sk-ant-test"))
    assert sent == ["claude-opus-5-5"]
