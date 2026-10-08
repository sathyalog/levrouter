"""Routing shared by the provider SDK integrations (Anthropic, OpenAI).

The SDK resource classes are patched (like the LangChain classes), so every client, including
ones created before enable(), is covered. A routed call only changes the `model` argument
(plus settings the routed model would reject); the client, its key and endpoint are untouched.
"""

from __future__ import annotations

import fnmatch
import functools
import inspect
import logging
from contextvars import ContextVar
from typing import Any, Callable

from .. import tracing
from ..providers import OPENROUTER_HOSTS, PROVIDERS, host_of, openrouter_vendor, spec_for

log = logging.getLogger("tokentriage")

# True while a routed call runs, so the SDK call made inside a routed LangChain call (or an SDK
# method that calls another one) is not routed and counted a second time.
ROUTED: ContextVar[bool] = ContextVar("tokentriage_routed", default=False)


_failed: set[str] = set()


def fail_open(where: str, exc: Exception) -> None:
    """Routing itself failed: the call goes to the configured model. Warn the first time, then stay quiet."""
    if where in _failed:
        log.debug("tokentriage: routing failed for %s", where, exc_info=True)
        return
    _failed.add(where)
    log.warning("tokentriage: routing failed for %s calls (%s: %s); they use the configured model instead. "
                "Set log_level: DEBUG for the traceback.", where, type(exc).__name__, exc)
    log.debug("tokentriage: routing failure details", exc_info=True)


def provider_for(client: Any, allowed: frozenset[str], extra_hosts: dict[str, str] | None) -> str | None:
    """Provider served by this client's endpoint, if it is one this SDK may route."""
    host = host_of(str(getattr(client, "base_url", "") or ""))
    if host is None:
        return None
    provider = None
    for configured, name in (extra_hosts or {}).items():
        if host == configured.lower():
            provider = name
            break
    else:
        if any(host == h or host.endswith("." + h) for h in OPENROUTER_HOSTS):
            provider = "openrouter"
        else:
            provider = next((name for name, spec in PROVIDERS.items()
                             if any(host == h or host.endswith("." + h) for h in spec.hosts)), None)
    return provider if provider in allowed else None


def _apply_compat(provider: str, model: str, kwargs: dict) -> None:
    """Adjust settings the routed model would reject (mirrors langchain._compat_updates)."""
    spec = spec_for(provider, model)
    ceiling = spec.max_output if spec else None
    for field in ("max_tokens", "max_completion_tokens"):
        value = kwargs.get(field)
        if ceiling and isinstance(value, int) and value > ceiling:
            kwargs[field] = ceiling
    if provider == "anthropic":
        thinking = kwargs.get("thinking")
        kind = thinking.get("type") if isinstance(thinking, dict) else None
        if model.startswith("claude-haiku") and kind == "adaptive":
            kwargs.pop("thinking")  # Haiku 4.5 has no adaptive thinking
        elif not model.startswith("claude-haiku") and kind == "enabled":
            kwargs["thinking"] = {"type": "adaptive"}  # budget_tokens is rejected on Sonnet 5 / Opus 5.x
    elif provider == "openai" and model.startswith("gpt-4o"):
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning", None)


def _plan(resource: Any, kwargs: dict, allowed: frozenset[str], enabled: set[str], extract: Callable,
          stream: bool) -> tuple[dict, Any] | None:
    try:
        return _plan_unguarded(resource, kwargs, allowed, enabled, extract, stream)
    except Exception as exc:  # noqa: BLE001 - a routing bug must never break the app's own call
        fail_open(type(resource).__module__.split(".")[0], exc)
        return None


def _plan_unguarded(resource: Any, kwargs: dict, allowed: frozenset[str], enabled: set[str], extract: Callable,
                    stream: bool) -> tuple[dict, Any] | None:
    from . import langchain as lc

    router = lc._State.router
    if router is None or ROUTED.get():
        return None
    model, messages = kwargs.get("model"), kwargs.get("messages")
    if not isinstance(model, str) or messages is None:
        return None
    cfg = router.config
    provider = provider_for(getattr(resource, "_client", None), allowed, cfg.provider_hosts)
    if provider is None or provider not in enabled:
        return None
    if any(fnmatch.fnmatch(model, pat) for pat in cfg.never_route):
        return None
    if provider == "openrouter" and cfg.openrouter_mode == "family" and openrouter_vendor(model) not in cfg.openrouter_families:
        return None

    msgs = list(messages)
    if kwargs.get("system"):  # Anthropic passes the system prompt outside `messages`
        msgs.insert(0, {"role": "system", "content": kwargs["system"]})
    features = extract(msgs, kwargs.get("tools"), kwargs.get("tool_choice"))
    max_out = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    decision = router.decide(provider, model, features, None, max_out if isinstance(max_out, int) else None)
    routed = dict(kwargs, model=decision.model)
    _apply_compat(provider, decision.model, routed)
    if stream and provider == "openai" and "stream_options" not in routed:
        # Without this the stream carries no token counts. Other OpenAI-compatible hosts may reject the option.
        routed["stream_options"] = {"include_usage": True}
    return routed, tracing.begin(decision, features, None, cfg, stream)


def _usage(result: Any) -> dict | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    tin = getattr(usage, "input_tokens", None)
    tout = getattr(usage, "output_tokens", None)
    if tin is None:  # OpenAI chat completions
        tin, tout = getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)
    return {"input_tokens": int(tin or 0), "output_tokens": int(tout or 0)}


def _finish(trace: Any, result: Any, error: BaseException | None = None) -> None:
    from . import langchain as lc

    try:
        tracing.end(trace, _usage(result), lc._State.telemetry, error)
    except Exception:  # noqa: BLE001 - observability must never break the app's call
        log.debug("tokentriage: tracing failed", exc_info=True)


_INPUT_FIELDS = ("input_tokens", "prompt_tokens")
_OUTPUT_FIELDS = ("output_tokens", "completion_tokens")


def _merge(acc: dict, usage: Any) -> None:
    """Fold a usage object from a stream event into the running totals (input: largest seen, output: latest)."""
    if usage is None:
        return
    for name in _INPUT_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            acc["input_tokens"] = max(acc.get("input_tokens", 0), value)
    for name in _OUTPUT_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            acc["output_tokens"] = value


def _observe(acc: dict, event: Any) -> None:
    try:
        _merge(acc, getattr(event, "usage", None))  # Anthropic message_delta, OpenAI final chunk
        _merge(acc, getattr(getattr(event, "message", None), "usage", None))  # Anthropic message_start
    except Exception:  # noqa: BLE001 - counting tokens must never break the app's stream
        log.debug("tokentriage: could not read stream usage", exc_info=True)


def _snapshot_usage(stream: Any) -> dict:
    """Usage so far from a SDK stream helper's own accumulated snapshot (no further reading of the stream)."""
    acc: dict = {}
    for attr in ("current_message_snapshot", "current_completion_snapshot"):
        try:
            _merge(acc, getattr(getattr(stream, attr), "usage", None))
        except Exception:  # noqa: BLE001 - no snapshot yet (nothing was read) or not this SDK
            continue
    return acc


def _is_usage_only(chunk: Any) -> bool:
    """The extra final chunk the SDK sends for stream_options.include_usage: no choices, only usage."""
    return getattr(chunk, "usage", None) is not None and not getattr(chunk, "choices", True)


class _Finisher:
    """Records the call in the usage log exactly once, however the stream ends."""

    def __init__(self, trace: Any):
        self._trace = trace
        self.usage: dict = {}
        self.done = False

    def finish(self, error: BaseException | None = None) -> None:
        if self.done:
            return
        self.done = True
        _finish(self._trace, SimpleUsage(self.usage) if self.usage else None, error)


class SimpleUsage:
    """Adapter so _finish/_usage can read totals gathered from stream events."""

    def __init__(self, totals: dict):
        self.usage = type("U", (), totals)()


class _SyncStream:
    """Wraps an SDK event stream: passes every event through and records token usage when it ends."""

    def __init__(self, inner: Any, trace: Any, hide_usage_chunk: bool):
        self._inner, self._fin, self._hide = inner, _Finisher(trace), hide_usage_chunk
        self._it = iter(inner)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = next(self._it)
            except StopIteration:
                self._fin.finish()
                raise
            except BaseException as exc:
                self._fin.finish(exc)
                raise
            _observe(self._fin.usage, item)
            if self._hide and _is_usage_only(item):
                continue
            return item

    def __enter__(self):
        if hasattr(self._inner, "__enter__"):
            self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            if hasattr(self._inner, "__exit__"):
                return self._inner.__exit__(*exc)
            return None
        finally:
            self._fin.finish()

    def close(self):
        try:
            self._inner.close()
        finally:
            self._fin.finish()

    def __del__(self):
        try:
            self._fin.finish()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _AsyncStream:
    def __init__(self, inner: Any, trace: Any, hide_usage_chunk: bool):
        self._inner, self._fin, self._hide = inner, _Finisher(trace), hide_usage_chunk
        self._it = inner.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                item = await self._it.__anext__()
            except StopAsyncIteration:
                self._fin.finish()
                raise
            except BaseException as exc:
                self._fin.finish(exc)
                raise
            _observe(self._fin.usage, item)
            if self._hide and _is_usage_only(item):
                continue
            return item

    async def __aenter__(self):
        if hasattr(self._inner, "__aenter__"):
            await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            if hasattr(self._inner, "__aexit__"):
                return await self._inner.__aexit__(*exc)
            return None
        finally:
            self._fin.finish()

    async def close(self):
        try:
            await self._inner.close()
        finally:
            self._fin.finish()

    def __del__(self):
        try:
            self._fin.finish()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _SyncManager:
    """Wraps `messages.stream()` / `chat.completions.stream()`: the caller still gets the SDK's own stream object."""

    def __init__(self, inner: Any, trace: Any):
        self._inner, self._fin, self._stream = inner, _Finisher(trace), None

    def __enter__(self):
        # The SDK's helper calls its own (patched) create() here; that inner call must not be routed and counted again.
        token = ROUTED.set(True)
        try:
            self._stream = self._inner.__enter__()
        finally:
            ROUTED.reset(token)
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        if self._stream is not None:
            self._fin.usage = _snapshot_usage(self._stream)
        try:
            return self._inner.__exit__(exc_type, exc, tb)
        finally:
            self._fin.finish(exc if isinstance(exc, Exception) else None)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _AsyncManager:
    def __init__(self, inner: Any, trace: Any):
        self._inner, self._fin, self._stream = inner, _Finisher(trace), None

    async def __aenter__(self):
        token = ROUTED.set(True)
        try:
            self._stream = await self._inner.__aenter__()
        finally:
            ROUTED.reset(token)
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        if self._stream is not None:
            self._fin.usage = _snapshot_usage(self._stream)
        try:
            return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            self._fin.finish(exc if isinstance(exc, Exception) else None)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _tap(result: Any, trace: Any, hide_usage_chunk: bool) -> Any:
    """Return `result` wrapped so its token usage is recorded when the stream ends. Unknown shapes pass through."""
    try:
        has = lambda name: hasattr(type(result), name)  # noqa: E731
        if has("__aenter__") and not has("__aiter__"):
            return _AsyncManager(result, trace)
        if has("__enter__") and not has("__iter__"):
            return _SyncManager(result, trace)
        if has("__aiter__"):
            return _AsyncStream(result, trace, hide_usage_chunk)
        if has("__iter__"):
            return _SyncStream(result, trace, hide_usage_chunk)
    except Exception:  # noqa: BLE001 - never break the app's call
        log.debug("tokentriage: could not wrap stream", exc_info=True)
    return result


async def _await(pending: Any, trace: Any, stream: bool, hide_usage_chunk: bool = False) -> Any:
    token = ROUTED.set(True)
    try:
        result = await pending
    except BaseException as exc:
        _finish(trace, None, exc)
        raise
    finally:
        ROUTED.reset(token)
    if stream:
        return _tap(result, trace, hide_usage_chunk)
    _finish(trace, result)
    return result


def _wrap(orig: Callable, allowed: frozenset[str], enabled: set[str], extract: Callable) -> Callable:
    @functools.wraps(orig)
    def method(self, *args, **kwargs):
        stream = orig.__name__ == "stream" or bool(kwargs.get("stream"))
        planned = None if args else _plan(self, kwargs, allowed, enabled, extract, stream)
        if planned is None:
            return orig(self, *args, **kwargs)
        routed, trace = planned
        injected = stream and "stream_options" in routed and "stream_options" not in kwargs
        token = ROUTED.set(True)
        try:
            result = orig(self, **routed)
        except BaseException as exc:
            _finish(trace, None, exc)
            raise
        finally:
            ROUTED.reset(token)
        if inspect.isawaitable(result):  # async clients return a coroutine from a sync-looking method
            return _await(result, trace, stream, injected)
        if stream:
            return _tap(result, trace, injected)
        _finish(trace, result)
        return result

    method.__tokentriage__ = True
    return method


class SdkPatches:
    """The methods one SDK integration has patched, so they can be restored."""

    def __init__(self, allowed: frozenset[str]):
        self.allowed = allowed
        self.enabled: set[str] = set()
        self.patches: list[tuple[type, str, Any]] = []

    def patch(self, classes: tuple[type, ...], names: tuple[str, ...], extract: Callable) -> None:
        for cls in classes:
            for name in names:
                orig = cls.__dict__.get(name)
                if orig is None or getattr(orig, "__tokentriage__", False):
                    continue
                setattr(cls, name, _wrap(orig, self.allowed, self.enabled, extract))
                self.patches.append((cls, name, orig))

    def restore(self) -> None:
        for cls, name, orig in reversed(self.patches):
            setattr(cls, name, orig)
        self.patches.clear()
        self.enabled.clear()
