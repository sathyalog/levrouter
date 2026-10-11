"""Route LangChain chat model calls by patching their generate/stream methods.

Supported classes: ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI, ChatGroq, ChatDeepSeek,
ChatMistralAI, ChatXAI, ChatHuggingFace (see providers.py). The patched method works out
which provider the instance really talks to (from its endpoint, not its class), classifies
the request, makes a copy of the chat model with the routed model id (`model_copy`), and
runs the original method on that copy. The shared instance is never mutated, and
`bind_tools` / `with_structured_output` / chains keep working because they all end up in
these methods.

Guardrails applied here, before any model is swapped:
- the endpoint must be a known host of the provider (else: not routed, logged once);
- the routed model always comes from the same provider's tier map, so the instance's
  API key and endpoint are only ever used with that provider's own models;
- fine-tuned / custom models (RouterConfig.never_route) are never swapped;
- settings the routed model would reject (max tokens, thinking) are adjusted on the copy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import functools
import importlib
import logging
import weakref
from typing import Any

from langchain_core.outputs import ChatResult

from .. import tracing
from ..features import extract
from ..providers import ALL_PROVIDERS, PROVIDER_CLASSES, TARGETS, Target, openrouter_vendor, resolve_provider, spec_for
from ..router import Router
from ..telemetry import Telemetry
from ..tracing import CallTrace
from ._sdk_common import ROUTED, fail_open

log = logging.getLogger("levrouter")

METHODS = ("_generate", "_agenerate", "_stream", "_astream")


class _State:
    router: Router | None = None
    telemetry: Telemetry | None = None
    route_all = False
    enabled_providers: set[str] = set()
    patches: list[tuple[type, str, bool, Any]] = []
    patched_classes: set[str] = set()
    # Chat models are pydantic and unhashable, so track them by id with a weakref
    # that removes the entry when the instance is collected.
    opt_in: dict[int, weakref.ref] = {}
    excluded: dict[int, weakref.ref] = {}
    # Routed copies currently running: their own nested calls pass straight through.
    in_flight: set[int] = set()
    # (class, reason) pairs already logged as "not routed", so the log is not flooded.
    skip_logged: set[tuple[str, str]] = set()
    # Evaluation mode (evaluate.Evaluator), set by levrouter.enable().
    evaluator: Any = None


def _track(registry: dict[int, weakref.ref], obj: Any) -> None:
    key = id(obj)
    registry[key] = weakref.ref(obj, lambda _r, key=key: registry.pop(key, None))


# -- per-call planning -------------------------------------------------------


def _should_route(llm: Any) -> bool:
    key = id(llm)
    if _State.router is None or key in _State.in_flight or key in _State.excluded:
        return False
    return _State.route_all or key in _State.opt_in


def _skip(target: Target, reason: str) -> None:
    key = (target.class_name, reason)
    if key not in _State.skip_logged:
        _State.skip_logged.add(key)
        log.info("skip  %s calls are not routed: %s", target.class_name, reason)


def _split_args(args: tuple, kwargs: dict) -> tuple[list, Any]:
    messages = args[0] if args else kwargs.get("messages", [])
    run_manager = args[2] if len(args) > 2 else kwargs.get("run_manager")
    return messages, run_manager


def _max_tokens(llm: Any, target: Target) -> tuple[str | None, int | None]:
    for name in target.max_tokens_fields:
        value = getattr(llm, name, None)
        if isinstance(value, int):
            return name, value
    return None, None


def _compat_updates(provider: str, llm: Any, target: Target, model: str) -> dict:
    """Adjust settings the routed model would reject."""
    updates: dict[str, Any] = {}
    spec = spec_for(provider, model)
    ceiling = spec.max_output if spec else None
    field, current_max = _max_tokens(llm, target)
    if field and ceiling and current_max and current_max > ceiling:
        updates[field] = ceiling

    if provider == "anthropic":
        thinking = getattr(llm, "thinking", None) or {}
        kind = thinking.get("type") if isinstance(thinking, dict) else None
        if model.startswith("claude-haiku") and kind == "adaptive":
            updates["thinking"] = None  # Haiku 4.5 has no adaptive thinking
        elif not model.startswith("claude-haiku") and kind == "enabled":
            updates["thinking"] = {"type": "adaptive"}  # budget_tokens is rejected on Sonnet 5 / Opus 5.x
    elif provider == "openai" and model.startswith("gpt-4o"):
        if getattr(llm, "reasoning_effort", None):
            updates["reasoning_effort"] = None
        if getattr(llm, "reasoning", None):
            updates["reasoning"] = None
    return updates


def _plan(llm: Any, target: Target, args: tuple, kwargs: dict, stream: bool) -> tuple[Any, CallTrace, dict | None] | None:
    try:
        return _plan_unguarded(llm, target, args, kwargs, stream)
    except Exception as exc:  # noqa: BLE001 - a routing bug must never break the app's own call
        fail_open(target.class_name, exc)
        return None


def _plan_unguarded(llm: Any, target: Target, args: tuple, kwargs: dict, stream: bool) -> tuple[Any, CallTrace, dict | None] | None:
    if not _should_route(llm):
        return None
    messages, run_manager = _split_args(args, kwargs)
    metadata = getattr(run_manager, "metadata", None) or {}
    if metadata.get("levrouter_disable"):
        return None
    cfg = _State.router.config

    provider, why = resolve_provider(llm, target, cfg.provider_hosts)
    if provider is None:
        _skip(target, why)
        return None
    if provider not in _State.enabled_providers and _State.route_all:
        _skip(target, f"provider {provider!r} not enabled")
        return None

    current = getattr(llm, target.model_field, None)
    if not current:
        return None
    if any(fnmatch.fnmatch(current, pat) for pat in cfg.never_route):
        _skip(target, f"model {current!r} matches never_route")
        return None
    if provider == "openrouter" and cfg.openrouter_mode == "family" and openrouter_vendor(current) not in cfg.openrouter_families:
        _skip(target, f"no OpenRouter family for vendor {openrouter_vendor(current)!r}")
        return None

    features = extract(messages, kwargs.get("tools"), kwargs.get("tool_choice"))
    _, max_out = _max_tokens(llm, target)
    thread = metadata.get("levrouter_thread") or metadata.get("thread_id")  # LangGraph adds thread_id
    decision = _State.router.decide(provider, current, features, metadata.get("levrouter_tier"), max_out,
                                    thread=str(thread) if thread else None)
    routed_model = decision.model
    routed_updates = _compat_updates(provider, llm, target, routed_model)

    ev = None
    if cfg.mode == "eval" and _State.evaluator is not None:
        serve = cfg.eval_serve
        if serve == "baseline" and routed_model != current:
            decision = dataclasses.replace(
                decision, model=current,
                reason=f"eval: serving configured model; routing would use {routed_model}; {decision.reason}")
        judge = cfg.eval_judge
        if judge == "top":
            tiers, _ = _State.router.tiers_for(provider, current, features)
            judge = (tiers or {}).get("complex", current)
        ev = {"provider": provider, "llm": llm, "model_field": target.model_field, "messages": list(messages),
              "kwargs": {k: v for k, v in kwargs.items() if k not in ("run_manager", "messages", "stop")},
              "routed_model": routed_model, "baseline_model": current, "served": serve, "tier": decision.tier,
              "judge_model": judge, "updates": routed_updates,
              "judge_updates": _compat_updates(provider, llm, target, judge)}

    trace = tracing.begin(decision, features, run_manager, cfg, stream)
    served_model = decision.model
    updates = routed_updates if served_model == routed_model else {}
    update = {target.model_field: served_model, **updates}
    if ev is not None:
        ev.update(task=trace.task, user=trace.user, call_id=trace.call_id)
    # Always copy, even for the same model: the copy's id marks this call as in flight
    # without blocking concurrent calls on the shared instance.
    return llm.model_copy(update=update), trace, ev


def _evaluate(ev: dict | None, message: Any) -> None:
    """Hand a finished call to the evaluator, if it is sampled. Never raises."""
    evaluator = _State.evaluator
    if ev is None or evaluator is None or message is None:
        return
    try:
        if not evaluator.should_evaluate(ev["call_id"], ev["routed_model"], ev["baseline_model"]):
            return
        from ..evaluate import EvalJob

        evaluator.submit(EvalJob(served_message=message, served_usage=dict(getattr(message, "usage_metadata", None) or {}), **ev))
    except Exception:  # noqa: BLE001 - evaluation must never break the app's call
        log.debug("levrouter: could not start evaluation", exc_info=True)


def _first_message(result: ChatResult) -> Any:
    return result.generations[0].message if result.generations else None


def _meta(trace: CallTrace) -> dict:
    return {**trace.decision.as_dict(), "call_id": trace.call_id, "task": trace.task, "user": trace.user}


def _annotate(result: ChatResult, trace: CallTrace) -> dict | None:
    usage = None
    for gen in result.generations:
        msg = getattr(gen, "message", None)
        if msg is None:
            continue
        msg.response_metadata["levrouter"] = _meta(trace)
        usage = usage or getattr(msg, "usage_metadata", None)
    return usage


def _add_usage(total: dict, usage: dict | None) -> None:
    for k in ("input_tokens", "output_tokens"):
        total[k] = total.get(k, 0) + int((usage or {}).get(k, 0))


def _finish(trace: CallTrace, usage: dict | None, error: BaseException | None = None) -> None:
    try:
        tracing.end(trace, usage, _State.telemetry, error)
    except Exception:  # noqa: BLE001 - observability must never break the app's call
        log.debug("levrouter: tracing failed", exc_info=True)


# -- wrappers ----------------------------------------------------------------


@contextlib.contextmanager
def _sdk_layer_off():
    """The provider SDK call LangChain makes here was already decided at this layer."""
    token = ROUTED.set(True)
    try:
        yield
    finally:
        try:
            ROUTED.reset(token)
        except ValueError:  # resumed in another context; just clear the flag there
            ROUTED.set(False)


def _sdk_off_iter(gen):
    try:
        while True:
            with _sdk_layer_off():
                try:
                    item = next(gen)
                except StopIteration:
                    return
            yield item
    finally:
        gen.close()


async def _sdk_off_aiter(agen):
    try:
        while True:
            with _sdk_layer_off():
                try:
                    item = await agen.__anext__()
                except StopAsyncIteration:
                    return
            yield item
    finally:
        await agen.aclose()


def _wrap(target: Target, name: str, orig: Any) -> Any:
    if name == "_generate":

        @functools.wraps(orig)
        def _generate(self, *args, **kwargs):
            plan = _plan(self, target, args, kwargs, stream=False)
            if plan is None:
                with _sdk_layer_off():
                    return orig(self, *args, **kwargs)
            routed, trace, ev = plan
            _State.in_flight.add(id(routed))
            try:
                with _sdk_layer_off():
                    result = orig(routed, *args, **kwargs)
            except BaseException as exc:
                _finish(trace, None, exc)
                raise
            finally:
                _State.in_flight.discard(id(routed))
            _finish(trace, _annotate(result, trace))
            _evaluate(ev, _first_message(result))
            return result

        return _generate

    if name == "_agenerate":

        @functools.wraps(orig)
        async def _agenerate(self, *args, **kwargs):
            plan = _plan(self, target, args, kwargs, stream=False)
            if plan is None:
                with _sdk_layer_off():
                    return await orig(self, *args, **kwargs)
            routed, trace, ev = plan
            _State.in_flight.add(id(routed))
            try:
                with _sdk_layer_off():
                    result = await orig(routed, *args, **kwargs)
            except BaseException as exc:
                _finish(trace, None, exc)
                raise
            finally:
                _State.in_flight.discard(id(routed))
            _finish(trace, _annotate(result, trace))
            _evaluate(ev, _first_message(result))
            return result

        return _agenerate

    if name == "_stream":

        @functools.wraps(orig)
        def _stream(self, *args, **kwargs):
            plan = _plan(self, target, args, kwargs, stream=True)
            if plan is None:
                yield from _sdk_off_iter(orig(self, *args, **kwargs))
                return
            routed, trace, ev = plan
            usage: dict = {}
            first = True
            error: BaseException | None = None
            full = None
            _State.in_flight.add(id(routed))
            try:
                for chunk in _sdk_off_iter(orig(routed, *args, **kwargs)):
                    if first:
                        chunk.message.response_metadata["levrouter"] = _meta(trace)
                        first = False
                    _add_usage(usage, getattr(chunk.message, "usage_metadata", None))
                    if ev is not None:
                        full = chunk.message if full is None else full + chunk.message
                    yield chunk
            except GeneratorExit:
                raise  # consumer stopped early: still a finished call, logged below
            except BaseException as exc:
                error = exc
                raise
            finally:
                _State.in_flight.discard(id(routed))
                _finish(trace, usage, error)
            _evaluate(ev, full)

        return _stream

    @functools.wraps(orig)
    async def _astream(self, *args, **kwargs):
        plan = _plan(self, target, args, kwargs, stream=True)
        if plan is None:
            async for chunk in _sdk_off_aiter(orig(self, *args, **kwargs)):
                yield chunk
            return
        routed, trace, ev = plan
        usage: dict = {}
        first = True
        error: BaseException | None = None
        full = None
        _State.in_flight.add(id(routed))
        try:
            async for chunk in _sdk_off_aiter(orig(routed, *args, **kwargs)):
                if first:
                    chunk.message.response_metadata["levrouter"] = _meta(trace)
                    first = False
                _add_usage(usage, getattr(chunk.message, "usage_metadata", None))
                if ev is not None:
                    full = chunk.message if full is None else full + chunk.message
                yield chunk
        except GeneratorExit:
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            _State.in_flight.discard(id(routed))
            _finish(trace, usage, error)
        _evaluate(ev, full)

    return _astream


# -- install / uninstall -----------------------------------------------------


def patch_class(cls: type, target: Target) -> None:
    """Patch one chat model class. Exposed so tests and custom subclasses can use it."""
    for name in METHODS:
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "__levrouter__", False):
            continue
        had_own = name in cls.__dict__
        wrapped = _wrap(target, name, orig)
        wrapped.__levrouter__ = True
        setattr(cls, name, wrapped)
        _State.patches.append((cls, name, had_own, orig))


def load_class(target: Target) -> type | None:
    try:
        return getattr(importlib.import_module(target.module), target.class_name)
    except ImportError:
        return None


def install(providers: tuple[str, ...]) -> list[str]:
    """Patch the LangChain classes that can reach these providers. Returns providers enabled."""
    enabled = []
    for provider in providers:
        if provider not in ALL_PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(ALL_PROVIDERS)}")
        found = False
        for class_name in PROVIDER_CLASSES[provider]:
            target = TARGETS[class_name]
            cls = load_class(target)
            if cls is None:
                continue
            found = True
            if class_name not in _State.patched_classes:
                patch_class(cls, target)
                _State.patched_classes.add(class_name)
        if found:
            _State.enabled_providers.add(provider)
            enabled.append(provider)
        else:
            log.debug("levrouter: no LangChain package installed for %s; skipping", provider)
    return enabled


def target_for(llm: Any) -> Target | None:
    for target in TARGETS.values():
        cls = load_class(target)
        if cls is not None and isinstance(llm, cls):
            return target
    return None


def uninstall() -> None:
    for cls, name, had_own, orig in reversed(_State.patches):
        if had_own:
            setattr(cls, name, orig)
        else:
            delattr(cls, name)
    _State.patches.clear()
    _State.patched_classes.clear()
    _State.enabled_providers.clear()
    _State.opt_in.clear()
    _State.excluded.clear()
    _State.skip_logged.clear()
    if _State.evaluator is not None:
        _State.evaluator.shutdown()
    _State.evaluator = None
    _State.route_all = False
    _State.router = None
    _State.telemetry = None
    tracing.teardown_logging()
