"""levrouter: pick the cheapest capable model per request, inside the same provider.

Powered by lev (https://github.com/InterfazeAI/lev) from the InterfazeAI team: every routing
decision is made by lev, and this package is a thin wrapper that applies those decisions to
LangChain chat models.

    import levrouter
    levrouter.enable()                       # every supported chat model call is routed

    llm = levrouter.route(ChatAnthropic(model="claude-opus-5-5"))   # or opt in one instance
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from . import live, pii, tracing
from . import usage_store as _usage
from .config import DEFAULT_TIERS, PRICES, TIERS, RouterConfig
from .integrations import langchain as _lc
from .providers import ALL_PROVIDERS, PROVIDERS, resolve_provider
from .router import Decision, Router
from .telemetry import Telemetry
from .config_loader import env_kill_switch, load_config, resolve_config
from .setup_check import validate_setup
from .cli_setup import setup_interactive

# The lev installation guide is shown only when lev-local is actually built (router.build_classifier),
# not on import: lev-http and heuristic users don't need lev installed.

log = logging.getLogger("levrouter")

__all__ = [
    "DEFAULT_TIERS",
    "PRICES",
    "PROVIDERS",
    "TIERS",
    "Decision",
    "Router",
    "RouterConfig",
    "disable",
    "enable",
    "exclude",
    "pii",
    "route",
    "stats",
    "eval_report",
    "usage",
    "usage_report",
    "warmup",
    "reset_thread",
    "load_config",
    "setup_interactive",
]

__version__ = "0.1.0"

_DEFAULT_PROVIDERS = ALL_PROVIDERS

M = TypeVar("M")


def _configure(config: RouterConfig | None, router: Router | None) -> None:
    if router is not None:
        _lc._State.router = router
    elif config is not None or _lc._State.router is None:
        _lc._State.router = Router(config or resolve_config().cfg)
    cfg = _lc._State.router.config
    if _lc._State.telemetry is None or config is not None or router is not None:
        _lc._State.telemetry = Telemetry(cfg.log_path)
    tracing.setup_logging(cfg, _lc._State.telemetry)
    _setup_usage(cfg)
    if _lc._State.evaluator is not None and _lc._State.evaluator.config is not cfg:
        _lc._State.evaluator.shutdown()
        _lc._State.evaluator = None
    if cfg.mode == "eval" and _lc._State.evaluator is None:
        from .evaluate import Evaluator

        _lc._State.evaluator = Evaluator(cfg, tracing.record_eval)


class _UsageState:
    # The in-memory window lives for the whole process, across reconfiguration.
    memory: _usage.MemoryStore | None = None
    server: live.LiveServer | None = None
    configured_for: int | None = None


_usage_warned: set[str] = set()


def _usage_unavailable(what: str, home: Any, exc: OSError) -> None:
    """The data folder can't be used (read-only filesystem, no home folder): say so once and go on."""
    if what not in _usage_warned:
        _usage_warned.add(what)
        log.warning("levrouter: turning off %s: %s is not usable (%s). Set LEVROUTER_HOME to a writable folder, "
                    "or set usage_file / usage_live to false.", what, home, exc.strerror or exc)


def _setup_usage(cfg: RouterConfig) -> None:
    """Wire the last-24h stores for this config (idempotent per config object)."""
    if _UsageState.configured_for == id(cfg):
        return
    _UsageState.configured_for = id(cfg)
    home = _usage.home_dir(cfg.usage_home)
    if _UsageState.memory is None:
        _UsageState.memory = _usage.MemoryStore(cfg.usage_retention_hours, cfg.usage_memory_max)
    files = None
    if cfg.usage_file:
        try:
            files = _usage.HourlyFileStore(home / "usage", cfg.usage_retention_hours)
        except OSError as exc:
            _usage_unavailable("usage files", home, exc)
    tracing.set_usage_sink(_usage.UsageSink(_UsageState.memory, files))

    if _UsageState.server is not None:
        _UsageState.server.stop()
        _UsageState.server = None
    if cfg.usage_live:
        try:
            server = live.LiveServer(_UsageState.memory, live.run_dir(home))
            if server.start():
                _UsageState.server = server
        except OSError as exc:
            _usage_unavailable("the live usage view", home, exc)


def _teardown_usage() -> None:
    tracing.set_usage_sink(None)
    if _UsageState.server is not None:
        _UsageState.server.stop()
    _UsageState.server = None
    _UsageState.memory = None
    _UsageState.configured_for = None


def enable(
    config: RouterConfig | None = None,
    *,
    providers: tuple[str, ...] | None = None,
    router: Router | None = None,
    frameworks: tuple[str, ...] | None = None,
) -> list[str]:
    """Route every supported chat model call in this process.

    With no config, the settings come from levrouter.yaml (found like load_config does), then the
    LEVROUTER_ENABLED / _BACKEND / _MODE / _LOG_LEVEL overrides. LEVROUTER_ENABLED=false turns routing
    off whatever config is passed.

    providers: which providers to route (default: the file's, else all, including "openrouter").
    Returns the ones whose LangChain package is installed.

    frameworks: which SDK frameworks to enable routing for (default: the file's, else ["langchain"]).
    Supports: "langchain", "anthropic", "openai".
    """
    if config is None and router is None:
        resolved = resolve_config()
        if not resolved.enabled:
            log.info("levrouter: disabled (enabled: false or LEVROUTER_ENABLED=false); not routing")
            return []
        if _lc._State.router is None:
            config = resolved.cfg
        frameworks = frameworks or resolved.frameworks
        providers = providers or resolved.providers
    elif env_kill_switch():
        log.info("levrouter: disabled by LEVROUTER_ENABLED=false; not routing")
        return []
    frameworks = frameworks or ("langchain",)
    providers = providers or _DEFAULT_PROVIDERS
    _configure(config, router)

    installed = []
    
    # Install each framework's routing
    for framework in frameworks:
        if framework == "langchain":
            _lc._State.route_all = True
            installed.extend(_lc.install(providers))
        elif framework == "anthropic":
            from .integrations import anthropic_sdk
            installed.extend(anthropic_sdk.install(providers))
        elif framework == "openai":
            from .integrations import openai_sdk
            installed.extend(openai_sdk.install(providers))
        else:
            raise ValueError(f"Unknown framework: {framework}. Choose from: langchain, anthropic, openai")

    return list(dict.fromkeys(installed))


def route(llm: M, config: RouterConfig | None = None, *, router: Router | None = None) -> M:
    """Opt one chat model instance into routing and return it (same object).

    `llm.bind_tools(...)` and `llm.with_structured_output(...)` stay routed.
    """
    target = _lc.target_for(llm)
    if target is None:
        supported = ", ".join(sorted(_lc.TARGETS))
        raise TypeError(f"levrouter.route() supports {supported}; got {type(llm).__name__}")
    _configure(config, router)
    provider, why = resolve_provider(llm, target, _lc._State.router.config.provider_hosts)
    if provider is None:
        raise ValueError(
            f"cannot route this {target.class_name}: {why}. "
            "Add the host to RouterConfig(provider_hosts={...}) if it serves a known provider."
        )
    _lc.install((provider,))
    _lc._track(_lc._State.opt_in, llm)
    return llm


def exclude(llm: Any) -> Any:
    """Never route this instance, even after enable(). Returns it."""
    _lc._track(_lc._State.excluded, llm)
    return llm


def disable() -> None:
    """Restore the original chat model methods and stop the live usage socket."""
    from .integrations import anthropic_sdk, openai_sdk

    anthropic_sdk.uninstall()
    openai_sdk.uninstall()
    _lc.uninstall()
    _teardown_usage()


def stats() -> dict:
    """Calls, models chosen, and actual vs baseline cost since enable()."""
    return _lc._State.telemetry.summary() if _lc._State.telemetry else {}


def usage(since: str | None = "24h", by: str = "model", user: str | None = None, task: str | None = None) -> dict:
    """This process's token usage and estimated cost from memory (last 24h at most).

    by: "model" | "provider" | "tier" | "task" | "user". Filter with user= (the value passed as
    metadata={"levrouter_user": ...}) to show an end user their own usage in your app's UI.
    """
    store = _UsageState.memory
    if store is None:
        return _usage.aggregate([], by=by)
    start, note = _usage.parse_since(since, store.window_s / 3600)
    report = _usage.aggregate(store.query(start), by=by, user=user, task=task)
    if note:
        report["note"] = note
    return report


def eval_report(since: str | None = "24h", by: str = "tier", user: str | None = None, task: str | None = None) -> dict:
    """Quality of routing from this process's evaluated calls (mode="eval"): held %, scores, savings."""
    store = _UsageState.memory
    if store is None:
        return _usage.aggregate_eval([], by=by)
    start, _ = _usage.parse_since(since, store.window_s / 3600)
    return _usage.aggregate_eval(store.query(start), by=by, user=user, task=task)


def usage_report(since: str | None = "24h", by: str = "model", user: str | None = None, task: str | None = None) -> str:
    """usage() as a printable table."""
    return _usage.format_table(usage(since, by, user, task), title=f"levrouter usage, last {since or '24h'} (this process)")


def reset_thread(thread_id: str) -> None:
    """Forget the model a conversation is kept on (e.g. after trimming or summarising its history)."""
    if _lc._State.router is not None:
        _lc._State.router.forget_thread(str(thread_id))


def warmup() -> None:
    """Load lev now (backend='lev-local'), so the first request is not delayed."""
    if _lc._State.router is None:
        _configure(None, None)
    classifier = _lc._State.router.classifier
    if hasattr(classifier, "warmup"):
        classifier.warmup()
