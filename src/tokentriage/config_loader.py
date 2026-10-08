"""Load tokentriage configuration from files or environment."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .config import TIERS, RouterConfig
from .providers import PROVIDERS

log = logging.getLogger("tokentriage")


def find_config_file(start_dir: Path | None = None, max_depth: int = 5) -> Path | None:
    """Find tokentriage config file in current or parent directories."""
    start_dir = start_dir or Path.cwd()
    current = start_dir
    
    for _ in range(max_depth):
        for name in ["tokentriage.yaml", "tokentriage.yml", "tokentriage.toml", ".tokentriage"]:
            path = current / name
            if path.is_file():  # ~/.tokentriage is the data folder, not a config file
                return path
        
        # Also check for [tool.tokentriage] in pyproject.toml
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                    if data.get("tool", {}).get("tokentriage"):
                        return pyproject
            except Exception:
                pass
        
        if current.parent == current:
            break
        current = current.parent
    
    return None


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError("pyyaml not installed; install with: pip install pyyaml")


def load_toml(path: Path) -> dict[str, Any]:
    """Load TOML config file."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            raise ImportError("tomli not installed for Python < 3.11; install with: pip install tomli")
    
    with open(path, "rb") as f:
        data = tomllib.load(f)
        if path.name == "pyproject.toml":
            return data.get("tool", {}).get("tokentriage", {})
        return data


def load_config_file(path: Path) -> dict[str, Any]:
    """Load config from file, auto-detect format."""
    if path.suffix in {".yaml", ".yml"} or path.name == ".tokentriage":
        return load_yaml(path)
    elif path.suffix == ".toml" or path.name == "pyproject.toml":
        return load_toml(path)
    else:
        raise ValueError(f"Unknown config format: {path.name}")


# Environment variables tokentriage reads. Everything else is configured in tokentriage.yaml.
#   secrets:    TOKENTRIAGE_LEV_API_KEY (read by the lev-http client), provider API keys (read by the SDKs)
#   locations:  TOKENTRIAGE_HOME (data folder), TOKENTRIAGE_CONFIG (path to the YAML file)
#   overrides:  TOKENTRIAGE_ENABLED (kill switch), TOKENTRIAGE_LOG_LEVEL, TOKENTRIAGE_BACKEND, TOKENTRIAGE_MODE
KEPT_ENV = frozenset({
    "TOKENTRIAGE_HOME", "TOKENTRIAGE_CONFIG", "TOKENTRIAGE_ENABLED", "TOKENTRIAGE_LOG_LEVEL",
    "TOKENTRIAGE_BACKEND", "TOKENTRIAGE_MODE", "TOKENTRIAGE_LEV_API_KEY",
})
INTERNAL_ENV = frozenset({"TOKENTRIAGE_AUTOENABLE"})  # set by `tokentriage run`

_BACKENDS = ("lev-local", "lev-http", "heuristic")
_MODES = ("route", "eval")
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TRUE = ("1", "true", "yes", "on")
_warned_env: set[str] = set()

# Startup messages go through the "tokentriage" logger. Before routing is enabled that logger has no handler
# yet, so they are written to stderr instead, honouring log_level. log_level=None leaves it all to the app.
_notify_level: int | None = logging.INFO
_HEADER_NAME = re.compile(r"[A-Za-z0-9-]+")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _set_notify_level(level: str | None) -> None:
    global _notify_level
    resolved = None if level is None else logging.getLevelName(str(level).upper())
    _notify_level = resolved if (level is None or isinstance(resolved, int)) else logging.INFO


def _notify(level: int, message: str) -> None:
    if log.handlers or _notify_level is None:
        log.log(level, message)
    elif level >= _notify_level:
        print(message, file=sys.stderr)


@dataclass
class Resolved:
    """Settings after reading the file and the kept environment variables."""

    cfg: RouterConfig
    frameworks: tuple[str, ...]
    providers: tuple[str, ...] | None
    enabled: bool
    source: Path | None


def env_kill_switch() -> bool:
    """True when TOKENTRIAGE_ENABLED is set to a false value: routing must stay off."""
    raw = os.environ.get("TOKENTRIAGE_ENABLED", "").strip()
    return bool(raw) and raw.lower() not in _TRUE


def _warn_once(key: str, message: str) -> None:
    if key not in _warned_env:
        _warned_env.add(key)
        _notify(logging.WARNING, f"⚠️  tokentriage: {message}")


def _apply_env(cfg: RouterConfig) -> None:
    """The kept override variables win over the file."""
    for name, field, allowed in (("TOKENTRIAGE_BACKEND", "backend", _BACKENDS), ("TOKENTRIAGE_MODE", "mode", _MODES)):
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            continue
        if raw in allowed:
            setattr(cfg, field, raw)
        else:
            _warn_once(name, f"{name}={raw!r} ignored; expected one of {', '.join(allowed)}")
    level = os.environ.get("TOKENTRIAGE_LOG_LEVEL", "").strip().upper()
    if level:
        if level in _LOG_LEVELS:
            cfg.log_level = level
        else:
            _warn_once("TOKENTRIAGE_LOG_LEVEL", f"TOKENTRIAGE_LOG_LEVEL={level!r} ignored; expected one of {', '.join(_LOG_LEVELS)}")


def _warn_removed_env(referenced: frozenset[str] = frozenset()) -> None:
    """`referenced` are variables named under lev_headers, which may legitimately start with TOKENTRIAGE_."""
    removed = sorted(k for k in os.environ if k.startswith("TOKENTRIAGE_") and k not in KEPT_ENV | INTERNAL_ENV | referenced)
    fresh = [k for k in removed if k not in _warned_env]
    if fresh:
        _warned_env.update(fresh)
        _notify(logging.WARNING, f"⚠️  tokentriage: ignoring environment variables that are no longer used: {', '.join(fresh)}. "
              "Set these in tokentriage.yaml (see docs/deployment.md).")


def _trigger_lev_download(checkpoint: str = "interfaze-ai/lev") -> bool:
    """Make sure lev is downloaded (blocking, with progress), then load and smoke-test it.

    LEV is a 4B decision model from InterfazeAI that classifies LLM requests
    into complexity tiers (simple/standard/complex). It is a ~0.2 GB adapter on a
    ~9.3 GB base model, cached in ~/.cache/huggingface/hub/.

    Returns: True if successful, False if fallback to heuristic needed.
    """
    try:
        import lev

        from .lev_download import ensure_lev_cached

        if not ensure_lev_cached(checkpoint):
            return False

        import time
        _notify(logging.INFO, "🔄 tokentriage: loading LEV into memory from the local cache...")
        start = time.perf_counter()
        model = lev.load(checkpoint)
        load_time = time.perf_counter() - start

        _notify(logging.INFO, f"✅ tokentriage: LEV loaded into memory in {load_time:.1f}s")

        # Test the model with a sample request to verify it works
        _notify(logging.INFO, "🧪 Testing LEV classification...")
        test_questions = {
            "tier": {
                "type": "choice",
                "instructions": "How capable a language model does this request need?",
                "criteria": {
                    "simple": "short factual answer, lookup, greeting",
                    "standard": "multi-paragraph writing, summarizing",
                    "complex": "multi-step reasoning, system design",
                },
            }
        }

        # Test with a simple request
        result = model.system_one("What is 2+2?", test_questions)
        tier_probs = result.answers["tier"].probabilities

        _notify(logging.INFO, f"   Sample: 'What is 2+2?'")
        _notify(logging.INFO, f"   → Simple: {tier_probs['simple']:.1%}")
        _notify(logging.INFO, f"   → Standard: {tier_probs['standard']:.1%}")
        _notify(logging.INFO, f"   → Complex: {tier_probs['complex']:.1%}")

        _notify(logging.INFO, "✅ tokentriage: LEV model ready for routing!")
        _notify(logging.INFO, "   Requests will now be routed based on complexity analysis")

        # Hand the loaded engine to the classifier so warmup() does not load a second ~9 GB copy.
        from .classifiers.lev_local import LevLocalClassifier
        LevLocalClassifier._engine = model

        return True

    except ImportError:
        print("⚠️  tokentriage: lev package not installed")
        print("   LEV is a separate prerequisite (not on PyPI)")
        print()
        print("   To install LEV:")
        print("   pip install 'lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev'")
        print()
        print("   Or use the heuristic backend for now (75% accurate, instant)")
        print()
        return False

    except Exception as e:
        _notify(logging.WARNING, f"⚠️  tokentriage: LEV download/test failed: {e}")
        _notify(logging.WARNING, "   Falling back to heuristic routing (75% accurate, <1ms)")
        return False


def _family_newest(model: str, available: list[str]) -> str | None:
    """Newest available id with the same name words, e.g. claude-sonnet-5 -> claude-sonnet-5-5."""
    from .providers import canonical_id

    def words(mid: str) -> set[str]:
        return {w for w in canonical_id(mid).split("-") if w and not w.isdigit()}

    def version(mid: str) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", canonical_id(mid)))

    same = [m for m in available if words(m) == words(model)]
    return max(same, key=version) if same else None


def _check_tier_models(cfg: RouterConfig) -> None:
    """Warn about tier models your API key can't use, per the last `tokentriage models refresh`.

    Reads a local file only: no network at startup, and never blocks.
    """
    from .providers import available_models_path, canonical_id

    try:
        providers = json.loads(available_models_path().read_text())["providers"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    for provider, ids in providers.items():
        known = {canonical_id(i) for i in ids} | set(ids)
        for tier, model in (cfg.tiers.get(provider) or {}).items():
            if model in known or canonical_id(model) in known:
                continue
            hint = _family_newest(model, ids)
            _notify(logging.WARNING, f"⚠️  tokentriage: {provider} {tier} tier uses {model!r}, which isn't in the models your "
                  f"{provider} key can use" + (f" (newest similar: {hint!r})" if hint else "")
                  + ". Set it under `models:` in tokentriage.yaml, or run `tokentriage models refresh`.")


def _clean_lev_headers(raw: Any) -> dict[str, str]:
    """`lev_headers: {Header-Name: ENV_VAR_NAME}`. Bad entries are reported and dropped, never fatal."""
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        _warn_once("lev_headers", "lev_headers ignored; expected a mapping of header name to environment variable name")
        return {}
    clean: dict[str, str] = {}
    for header, var in raw.items():
        if not (isinstance(header, str) and _HEADER_NAME.fullmatch(header) and isinstance(var, str) and _ENV_NAME.fullmatch(var)):
            _warn_once(f"lev_headers:{header}", f"lev_headers entry {header!r} ignored; use "
                       "{Header-Name: ENV_VAR_NAME} (the value is the NAME of an environment variable, not the secret itself)")
            continue
        clean[header] = var
    return clean


def _models_to_tiers(models: dict) -> tuple[dict, dict]:
    """`models: {simple: {providers: ["anthropic/claude-haiku-4-5"]}}` -> (tiers, openrouter_families).

    "provider/model" sets that provider's tier; "openrouter/vendor/model" sets the OpenRouter family
    tier to the OpenRouter id "vendor/model". The first entry per provider and tier wins.
    """
    tiers: dict[str, dict[str, str]] = {}
    families: dict[str, dict[str, str]] = {}
    for tier, spec in models.items():
        entries = spec.get("providers", []) if isinstance(spec, dict) else []
        if tier not in TIERS:
            _notify(logging.WARNING, f"⚠️  tokentriage: models.{tier} ignored (tiers are {', '.join(TIERS)})")
            continue
        for entry in entries:
            head, _, rest = str(entry).partition("/")
            if head == "openrouter" and "/" in rest:
                families.setdefault(rest.split("/", 1)[0], {}).setdefault(tier, rest)
            elif head in PROVIDERS and rest:
                tiers.setdefault(head, {}).setdefault(tier, rest)
            else:
                _notify(logging.WARNING, f"⚠️  tokentriage: models.{tier} entry {entry!r} ignored; use provider/model "
                      f"({', '.join(PROVIDERS)}) or openrouter/vendor/model")
    return tiers, families


def resolve_config(path: str | Path | None = None) -> Resolved:
    """Read the settings: the file (an explicit path, else $TOKENTRIAGE_CONFIG, else found by searching
    the current folder and its parents), then the kept environment variables on top.
    """
    chosen = path or os.environ.get("TOKENTRIAGE_CONFIG") or None
    source = Path(chosen) if chosen else find_config_file()
    config_dict: dict[str, Any] = dict(load_config_file(source)) if source else {}
    if source:
        log.debug("tokentriage: loaded config file %s", source)

    frameworks = tuple(config_dict.pop("frameworks", None) or ("langchain",))
    providers = tuple(config_dict.pop("providers")) if config_dict.get("providers") else None

    # Flatten nested 'router' dict into top-level config
    if "router" in config_dict:
        config_dict.update(config_dict.pop("router") or {})

    enabled = bool(config_dict.pop("enabled", True))
    _set_notify_level(config_dict.get("log_level", "INFO"))  # the file's level, before any warning is shown
    config_dict["lev_headers"] = _clean_lev_headers(config_dict.get("lev_headers"))
    _warn_removed_env(frozenset(config_dict["lev_headers"].values()))
    if "lev_api_key" in config_dict:
        config_dict.pop("lev_api_key")
        _warn_once("lev_api_key", "lev_api_key in a config file is ignored (files get committed). "
                   "Set the TOKENTRIAGE_LEV_API_KEY environment variable instead.")
    if "models" in config_dict:
        tiers, families = _models_to_tiers(config_dict.pop("models") or {})
        # An explicit `tiers:` / `openrouter_families:` section wins over `models:`.
        for provider, mapping in (config_dict.get("tiers") or {}).items():
            tiers.setdefault(provider, {}).update(mapping)
        for vendor, mapping in (config_dict.get("openrouter_families") or {}).items():
            families.setdefault(vendor, {}).update(mapping)
        config_dict["tiers"], config_dict["openrouter_families"] = tiers, families

    # Keep only real RouterConfig fields. Anything else (e.g. `features:`) is reported,
    # never allowed to make the whole file silently fall back to defaults.
    valid = {f.name for f in fields(RouterConfig)}
    ignored = sorted(k for k in config_dict if k not in valid)
    if ignored:
        _warn_once(f"keys:{ignored}", f"ignoring config keys that are not RouterConfig settings: {', '.join(ignored)}")
    config_dict = {k: v for k, v in config_dict.items() if k in valid}
    for key in ("never_route", "openrouter_vendors"):
        if isinstance(config_dict.get(key), list):
            config_dict[key] = tuple(config_dict[key])
    cfg = RouterConfig(**config_dict)
    _set_notify_level(cfg.log_level)
    _apply_env(cfg)
    _set_notify_level(cfg.log_level)
    if env_kill_switch():
        enabled = False
    return Resolved(cfg, frameworks, providers, enabled, source)


def _start_http_warmup(cfg: RouterConfig) -> threading.Thread | None:
    """Wake a scale-to-zero lev server in the background; never blocks startup and never raises."""

    def run() -> None:
        try:
            from .router import build_classifier

            build_classifier(cfg).warmup()
            _notify(logging.INFO, "tokentriage: lev-http server is warm")
        except Exception as exc:  # noqa: BLE001 - a failed warm-up only means the first call may fall back
            _notify(logging.WARNING, f"tokentriage: lev-http warm-up failed ({type(exc).__name__}); the first call may use the heuristic")

    thread = threading.Thread(target=run, name="tokentriage-warmup", daemon=True)
    thread.start()
    return thread


def load_config(config_path: str | None = None, auto_enable: bool = True) -> tuple[RouterConfig, tuple[str, ...], tuple[str, ...] | None]:
    """Load tokentriage.yaml (see resolve_config), prepare lev if needed, and enable routing.

    Args:
        config_path: Optional path to a config file (else $TOKENTRIAGE_CONFIG, else searched for)
        auto_enable: If True, automatically enable routing and download LEV model if lev-local

    Returns: (RouterConfig, frameworks, providers)
    """
    resolved = resolve_config(config_path)
    cfg, frameworks, providers, enabled = resolved.cfg, resolved.frameworks, resolved.providers, resolved.enabled
    log.debug("tokentriage: backend=%s timeout_s=%s block_on_load=%s auto_enable=%s",
              cfg.backend, cfg.timeout_s, cfg.block_on_load, auto_enable)

    if not enabled:
        _notify(logging.WARNING, "tokentriage: disabled (enabled: false or TOKENTRIAGE_ENABLED=false); calls are not routed")
        return cfg, frameworks, providers

    _check_tier_models(cfg)

    # If lev-local backend is configured, trigger model download
    if auto_enable and cfg.backend == "lev-local":
        if not _trigger_lev_download(cfg.lev_checkpoint):
            # Fallback to heuristic if download fails
            _notify(logging.WARNING, "   Switching to heuristic backend for now")
            cfg.backend = "heuristic"

    # Warmup LEV BEFORE enabling hooks if block_on_load is set
    if cfg.backend == "lev-local" and cfg.block_on_load:
        try:
            from .classifiers.lev_local import LevLocalClassifier
            _notify(logging.INFO, "⏳ tokentriage: Preloading LEV model (this may take 5-10 seconds)...")
            classifier = LevLocalClassifier(cfg.lev_checkpoint, cfg.timeout_s, cfg.block_on_load)
            classifier.warmup()
            _notify(logging.INFO, "✅ tokentriage: LEV model preloaded and ready")
        except Exception as e:
            _notify(logging.WARNING, f"⚠️  tokentriage: LEV preload failed: {e}")

    if cfg.backend == "lev-http" and cfg.warm_on_load:
        _start_http_warmup(cfg)

    # Auto-enable routing if requested
    if auto_enable:
        from . import enable
        try:
            kwargs = {"frameworks": frameworks}
            if providers:
                kwargs["providers"] = providers
            installed = enable(cfg, **kwargs)
            if installed:
                _notify(logging.INFO, f"✅ tokentriage: Routing enabled for {', '.join(installed)}")
        except Exception as e:
            _notify(logging.WARNING, f"⚠️  tokentriage: Failed to enable routing: {e}")

    return cfg, frameworks, providers
