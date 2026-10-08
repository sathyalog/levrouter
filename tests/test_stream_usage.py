"""Token usage of streamed SDK calls reaches the usage log (real SDK clients, faked HTTP transport)."""

import json

import httpx2
import openai
import pytest
from conftest import SIMPLE, FixedClassifier

import anthropic
import tokentriage
from tokentriage import Router, RouterConfig, tracing


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


ANTHROPIC_OK = _sse(
    ("message_start", {"type": "message_start", "message": {
        "id": "m1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 0}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "4"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 7}}),
    ("message_stop", {"type": "message_stop"}),
)
ANTHROPIC_ERROR = ANTHROPIC_OK.split("event: content_block_start")[0] + _sse(
    ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))


def _chunk(choices, usage=None):
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-6-luna", "choices": choices}
    if usage:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


OPENAI_CONTENT = (
    _chunk([{"index": 0, "delta": {"role": "assistant", "content": "4"}, "finish_reason": None}])
    + _chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}])
)
OPENAI_USAGE = _chunk([], {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18})


@pytest.fixture
def wire(monkeypatch):
    """Fake HTTP for both SDKs. `wire.bodies` are the request bodies; `wire.stream_body` is what streams return."""

    class Wire:
        bodies: list[dict] = []
        stream_body: str = ""
        stream_status = 200

    wire = Wire()
    wire.bodies = []

    def respond(request):
        body = json.loads(request.content)
        wire.bodies.append(body)
        return httpx2.Response(wire.stream_status, text=wire.stream_body,
                               headers={"content-type": "text/event-stream"}, request=request)

    async def respond_async(self, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx2.Client, "send", lambda self, request, **kw: respond(request))
    monkeypatch.setattr(httpx2.AsyncClient, "send", respond_async)
    return wire


@pytest.fixture
def ended(monkeypatch):
    """(usage, error) for every call tokentriage finished recording."""
    records: list[tuple[dict | None, BaseException | None]] = []
    real = tracing.end

    def spy(trace, usage, telemetry, error=None):
        records.append((usage, error))
        return real(trace, usage, telemetry, error)

    monkeypatch.setattr(tracing, "end", spy)
    return records


def _enable(*frameworks):
    tokentriage.enable(router=Router(RouterConfig(), FixedClassifier(SIMPLE)), frameworks=frameworks)


def _anthropic_args():
    return dict(model="claude-opus-5-5", max_tokens=50, messages=[{"role": "user", "content": "What is 2+2?"}])


def _openai_args(**kw):
    return dict(model="gpt-6-astra", messages=[{"role": "user", "content": "What is 2+2?"}], **kw)


def _tokens(record):
    usage = record[0]
    return usage["input_tokens"], usage["output_tokens"]


# -- Anthropic ---------------------------------------------------------------

def test_anthropic_create_stream_records_usage_when_exhausted(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    events = list(anthropic.Anthropic(api_key="sk-ant-test").messages.create(stream=True, **_anthropic_args()))
    assert [e.type for e in events][0] == "message_start" and events[-1].type == "message_stop"
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7) and ended[0][1] is None
    assert tokentriage.stats()["cost_usd"] > 0


def test_anthropic_early_close_still_records_once(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    stream = anthropic.Anthropic(api_key="sk-ant-test").messages.create(stream=True, **_anthropic_args())
    next(stream)  # message_start only
    stream.close()
    stream.close()
    del stream
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 0)


def test_anthropic_stream_as_context_manager(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    with anthropic.Anthropic(api_key="sk-ant-test").messages.create(stream=True, **_anthropic_args()) as stream:
        assert sum(1 for _ in stream) == 6
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7)


def test_anthropic_messages_stream_helper_keeps_its_api_and_records(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    with anthropic.Anthropic(api_key="sk-ant-test").messages.stream(**_anthropic_args()) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "4" and final.usage.output_tokens == 7
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7)
    assert wire.bodies[0]["model"] == "claude-haiku-4-5"


def test_anthropic_messages_stream_left_early_records_what_was_read(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    with anthropic.Anthropic(api_key="sk-ant-test").messages.stream(**_anthropic_args()) as stream:
        next(iter(stream))
    assert len(ended) == 1 and ended[0][1] is None


def test_anthropic_stream_error_is_recorded_and_still_raised(wire, ended):
    wire.stream_body = ANTHROPIC_ERROR
    _enable("anthropic")
    stream = anthropic.Anthropic(api_key="sk-ant-test").messages.create(stream=True, **_anthropic_args())
    with pytest.raises(anthropic.APIStatusError):
        list(stream)
    assert len(ended) == 1 and ended[0][1] is not None


async def test_anthropic_async_create_stream(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    client = anthropic.AsyncAnthropic(api_key="sk-ant-test")
    stream = await client.messages.create(stream=True, **_anthropic_args())
    events = [e async for e in stream]
    assert len(events) == 6 and len(ended) == 1 and _tokens(ended[0]) == (11, 7)


async def test_anthropic_async_stream_helper(wire, ended):
    wire.stream_body = ANTHROPIC_OK
    _enable("anthropic")
    client = anthropic.AsyncAnthropic(api_key="sk-ant-test")
    async with client.messages.stream(**_anthropic_args()) as stream:
        text = "".join([t async for t in stream.text_stream])
        final = await stream.get_final_message()
    assert text == "4" and final.usage.output_tokens == 7
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7)


def test_anthropic_non_stream_calls_are_unchanged(wire, ended, monkeypatch):
    _enable("anthropic")

    def respond(request):
        return httpx2.Response(200, request=request, json={
            "id": "m1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "4"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1}})

    monkeypatch.setattr(httpx2.Client, "send", lambda self, request, **kw: respond(request))
    anthropic.Anthropic(api_key="sk-ant-test").messages.create(**_anthropic_args())
    assert len(ended) == 1 and _tokens(ended[0]) == (5, 1)


# -- OpenAI ------------------------------------------------------------------

def test_openai_stream_asks_for_usage_records_it_and_hides_the_extra_chunk(wire, ended):
    wire.stream_body = OPENAI_CONTENT + OPENAI_USAGE + "data: [DONE]\n\n"
    _enable("openai")
    chunks = list(openai.OpenAI(api_key="sk-test").chat.completions.create(stream=True, **_openai_args()))
    assert wire.bodies[0]["stream_options"] == {"include_usage": True}
    assert len(chunks) == 2 and all(c.choices for c in chunks), "the injected usage-only chunk must not reach the caller"
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7)


def test_openai_callers_own_stream_options_are_respected_and_chunk_stays_visible(wire, ended):
    wire.stream_body = OPENAI_CONTENT + OPENAI_USAGE + "data: [DONE]\n\n"
    _enable("openai")
    chunks = list(openai.OpenAI(api_key="sk-test").chat.completions.create(
        stream=True, stream_options={"include_usage": True}, **_openai_args()))
    assert len(chunks) == 3 and not chunks[-1].choices
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7)


def test_openai_compatible_hosts_are_not_sent_stream_options(wire, ended):
    wire.stream_body = OPENAI_CONTENT + "data: [DONE]\n\n"
    _enable("openai")
    client = openai.OpenAI(api_key="gsk-test", base_url="https://api.groq.com/openai/v1")
    list(client.chat.completions.create(stream=True, model="llama-3.3-70b-versatile",
                                        messages=[{"role": "user", "content": "hi"}]))
    assert "stream_options" not in wire.bodies[0]
    assert len(ended) == 1 and wire.bodies[0]["model"] != "llama-3.3-70b-versatile", "the call was routed and recorded"


def test_openai_stream_without_usage_still_records_the_call(wire, ended):
    wire.stream_body = OPENAI_CONTENT + "data: [DONE]\n\n"
    _enable("openai")
    list(openai.OpenAI(api_key="sk-test").chat.completions.create(stream=True, **_openai_args()))
    assert len(ended) == 1 and ended[0][0] is None and ended[0][1] is None


def test_openai_stream_helper_records(wire, ended):
    wire.stream_body = OPENAI_CONTENT + OPENAI_USAGE + "data: [DONE]\n\n"
    _enable("openai")
    with openai.OpenAI(api_key="sk-test").chat.completions.stream(**_openai_args()) as stream:
        for _ in stream:
            pass
    assert len(ended) == 1 and _tokens(ended[0]) == (11, 7)


async def test_openai_async_create_stream(wire, ended):
    wire.stream_body = OPENAI_CONTENT + OPENAI_USAGE + "data: [DONE]\n\n"
    _enable("openai")
    stream = await openai.AsyncOpenAI(api_key="sk-test").chat.completions.create(stream=True, **_openai_args())
    chunks = [c async for c in stream]
    assert len(chunks) == 2 and len(ended) == 1 and _tokens(ended[0]) == (11, 7)
