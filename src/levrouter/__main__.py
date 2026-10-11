"""levrouter command line.

    levrouter check                 is the environment ready? (lev, device, provider packages, usage store)
    levrouter usage [--live] ...    token usage and estimated cost for the last 24h
    levrouter run -- <command>      run a program with routing enabled, without editing it
    levrouter models refresh        update prices/capabilities and the model ids your API keys can use
    levrouter openrouter refresh    update the OpenRouter model catalog only (prices, input types)
    levrouter eval run FILE ...     measure quality: routed vs configured model on your prompts (paid calls)
    levrouter eval report ...       quality of routing from evaluated calls (last 24h)

Only `run` (through the program it starts) calls LLM provider APIs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from . import __version__, live
from . import usage_store as u
from .config_loader import resolve_config
from .providers import PROVIDER_CLASSES, TARGETS
from .cli_setup import setup_cmd

LEV_INSTALL = 'pip install "lev[serve] @ git+https://github.com/InterfazeAI/lev#subdirectory=packages/lev"'


def _version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def check() -> int:
    ok = "✓"
    no = "✗"
    print(f"levrouter {__version__}  (Python {sys.version.split()[0]})")
    if sys.version_info < (3, 12):
        print(f"  {no} Python 3.12+ is required (lev's minimum)")

    print("\nClassifier: lev by InterfazeAI (powers every routing decision)")
    lev_version = _version("lev")
    if lev_version is None:
        print(f"  {no} lev not installed -> routing uses the keyword heuristic")
        print(f"    install: {LEV_INSTALL}")
    else:
        print(f"  {ok} lev {lev_version}")
        if importlib.util.find_spec("torch") is None:
            print(f"  {no} torch missing: lev was installed without [serve]. Reinstall with: {LEV_INSTALL}")
        else:
            import torch

            if torch.cuda.is_available():
                print(f"  {ok} CUDA GPU: {torch.cuda.get_device_name(0)} (fast decisions)")
            else:
                print(f"  ! no CUDA GPU: lev runs on CPU, seconds per decision. Raise timeout_s or use backend='lev-http'.")

    resolved = resolve_config()
    cfg = resolved.cfg
    where = str(resolved.source) if resolved.source else "no levrouter.yaml found, using defaults"
    print(f"\nConfig ({where}):")
    print(f"  enabled={resolved.enabled} backend={cfg.backend} timeout_s={cfg.timeout_s} block_on_load={cfg.block_on_load} "
          f"log_level={cfg.log_level}")
    print(f"  frameworks={', '.join(resolved.frameworks)} sticky_threads={cfg.sticky_threads} "
          f"thread_ttl_s={cfg.thread_ttl_s} escalate_after_repeats={cfg.escalate_after_repeats}")

    print("\nProvider packages:")
    for provider, classes in PROVIDER_CLASSES.items():
        own = TARGETS[classes[0]]
        installed = importlib.util.find_spec(own.module) is not None
        extra = f"pip install 'levrouter[{provider}]'"
        print(f"  {ok if installed else '-'} {provider:<12} {own.class_name:<24} {'' if installed else extra}")

    home = u.home_dir(cfg.usage_home)
    store = u.HourlyFileStore(home / "usage", cfg.usage_retention_hours)
    files = store.files()
    print(f"\nUsage store: {home}")
    print(f"  files: {'on' if cfg.usage_file else 'off'} - {len(files)} hour file(s)"
          + (f", oldest {files[0][1].stem} UTC" if files else ""))
    sockets = sorted(live.run_dir(home).glob("*.sock")) if live.run_dir(home).is_dir() else []
    print(f"  live:  {'on' if cfg.usage_live else 'off'} - {len(sockets)} socket(s) in {live.run_dir(home)}")
    return 0


def usage_cmd(args: argparse.Namespace) -> int:
    cfg = resolve_config().cfg
    home = u.home_dir(args.dir or cfg.usage_home)
    retention = cfg.usage_retention_hours
    request = {"op": "usage", "since": args.since, "by": args.by, "user": args.user, "task": args.task,
               "recent": args.recent}

    if args.live:
        answers = live.query_all(live.run_dir(home), request)
        good = [a for a in answers if "report" in a]
        if not good:
            msg = {"error": "no running levrouter process found", "run_dir": str(live.run_dir(home))}
            print(json.dumps(msg) if args.json else
                  f"No running app with levrouter found (looked in {live.run_dir(home)}).\n"
                  "Live numbers exist only while the app runs; use `levrouter usage` (no --live) to read the "
                  "rolling files.")
            return 1
        report = u.merge([a["report"] for a in good])
        recent = [u.UsageRecord.from_dict(r) for a in good for r in a.get("recent", [])]
        note = next((a["note"] for a in good if a.get("note")), "")
        source = f"live memory of {len(good)} running process(es)"
        if args.verbose:
            for a in answers:
                print(f"  pid {a.get('pid')}: " + (a["error"] if "error" in a else f"{a['records']} calls in memory"),
                      file=sys.stderr)
    else:
        store = u.HourlyFileStore(home / "usage", retention)
        if not store.files():
            msg = {"error": "no usage files", "dir": str(store.dir)}
            print(json.dumps(msg) if args.json else
                  f"No usage files in {store.dir}.\n"
                  "They are written by apps that call levrouter.enable() with usage_file=True (the default). "
                  "Point to another location with --dir or LEVROUTER_HOME, or run `levrouter check`.")
            return 1
        since, note = u.parse_since(args.since, retention)
        records, bad = store.read(since)
        report = u.aggregate(records, by=args.by, user=args.user, task=args.task)
        recent = records
        source = f"rolling files in {store.dir}" + (f" ({bad} malformed line(s) skipped)" if bad else "")

    if args.json:
        out = {"source": source, "since": args.since, "note": note, **report}
        if args.recent:
            out["recent"] = [r.__dict__ for r in sorted(recent, key=lambda r: r.ts)[-args.recent:]]
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(u.format_table(report, title=f"levrouter usage, last {args.since} - {source}"))
    if not args.live:
        quality = u.aggregate_eval(records, by=args.by, user=args.user, task=args.task)
        if quality["total"]["evaluated"] or quality["total"]["errors"]:
            print()
            print(u.format_eval_table(quality, title="Quality of routing (evaluated calls)"))
    if note:
        print(f"note: {note}")
    if args.recent:
        print(f"\nLast {args.recent} calls:")
        print(u.format_recent(recent, args.recent))
    return 0


def run_cmd(args: argparse.Namespace) -> int:
    import os
    import subprocess

    command = list(args.command_args)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("usage: levrouter run [--mode eval] [--backend heuristic] -- <command> [args...]", file=sys.stderr)
        return 2
    autoload = Path(__file__).resolve().parent / "_autoload"
    env = dict(os.environ)
    env["LEVROUTER_AUTOENABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(autoload), env.get("PYTHONPATH", "")) if p)
    if args.mode:
        env["LEVROUTER_MODE"] = args.mode
    if args.backend:
        env["LEVROUTER_BACKEND"] = args.backend
    try:
        return subprocess.call(command, env=env)
    except FileNotFoundError:
        print(f"levrouter run: command not found: {command[0]}", file=sys.stderr)
        return 127


def openrouter_cmd(args: argparse.Namespace) -> int:
    import urllib.request

    from .providers import build_openrouter_catalog, openrouter_catalog, openrouter_catalog_path

    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"User-Agent": "levrouter"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        catalog = build_openrouter_catalog(json.load(resp)["data"])
    path = openrouter_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=0, sort_keys=True))
    openrouter_catalog.cache_clear()
    print(f"saved {len(catalog['models'])} OpenRouter models to {path}")
    return 0


# Provider -> (API key variable, models endpoint, header carrying the key). Each is listed only when its key is set.
MODEL_LISTS = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/models?limit=1000", "x-api-key"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1/models", "Authorization"),
    "gemini": ("GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", "x-goog-api-key"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1/models", "Authorization"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/models", "Authorization"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1/models", "Authorization"),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1/models", "Authorization"),
}


def _list_models(url: str, header: str, key: str) -> list[str]:
    import urllib.request

    headers = {"User-Agent": "levrouter", header: f"Bearer {key}" if header == "Authorization" else key}
    if header == "x-api-key":
        headers["anthropic-version"] = "2023-06-01"
    ids: list[str] = []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
            page = json.load(resp)
        rows = page.get("data") or page.get("models") or []
        ids += [str(r.get("id") or r.get("name", "")).removeprefix("models/") for r in rows]
        url = ""
        if page.get("has_more") and page.get("last_id"):  # Anthropic pagination
            url = f"{MODEL_LISTS['anthropic'][1]}&after_id={page['last_id']}"
    return sorted(i for i in ids if i)


def models_cmd(args: argparse.Namespace) -> int:
    """Refresh prices and capabilities (OpenRouter catalogue) and the model ids your keys can use."""
    import os
    import time as _time

    from .providers import available_models_path

    status = 0
    try:
        openrouter_cmd(args)
    except Exception as exc:  # noqa: BLE001 - keep going: the per-provider lists are still useful
        print(f"could not refresh the OpenRouter catalogue: {exc}", file=sys.stderr)
        status = 1
    available: dict[str, list[str]] = {}
    for provider, (env, url, header) in MODEL_LISTS.items():
        key = os.environ.get(env)
        if not key:
            print(f"  - {provider:<10} skipped ({env} not set)")
            continue
        try:
            available[provider] = _list_models(url, header, key)
            print(f"  ✓ {provider:<10} {len(available[provider])} models available to your key")
        except Exception as exc:  # noqa: BLE001 - one provider failing must not stop the others
            print(f"  ✗ {provider:<10} {type(exc).__name__}: {exc}", file=sys.stderr)
            status = 1
    if available:
        path = available_models_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                                    "providers": available}, indent=1))
        print(f"saved available models to {path}")
    return status


def eval_cmd(args: argparse.Namespace) -> int:
    import time as _time

    import levrouter

    from .evaluate import load_prompts, make_chat_model, run_batch

    cfg_env = resolve_config().cfg
    home = u.home_dir(cfg_env.usage_home)

    if args.action == "report":
        store = u.HourlyFileStore(home / "usage", cfg_env.usage_retention_hours)
        since, note = u.parse_since(args.since, cfg_env.usage_retention_hours)
        records, _ = store.read(since)
        report = u.aggregate_eval(records, by=args.by)
        if args.json:
            print(json.dumps(report, indent=2))
            return 0
        print(u.format_eval_table(report, title=f"Quality of routing, last {args.since} - {store.dir}"))
        if args.regressions and args.log:
            _print_regressions(args.log, args.regressions)
        return 0

    # eval run
    prompts = load_prompts(args.file)[: args.limit or None]
    if not prompts:
        print("no prompts found", file=sys.stderr)
        return 1
    log_path = args.log or str(home / f"eval-{_time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config().cfg
    cfg.mode = "eval"
    cfg.eval_sample_rate = 1.0
    cfg.eval_serve = "routed"
    cfg.eval_budget_usd = args.budget
    cfg.eval_log_path = log_path
    cfg.summary_at_exit = False
    if args.judge:
        cfg.eval_judge = args.judge
    if args.backend:
        cfg.backend = args.backend
    levrouter.enable(cfg)
    from .integrations import langchain as lc

    llm = make_chat_model(args.provider, args.model)
    print(f"evaluating {len(prompts)} prompts: {args.provider} {args.model}, judge {cfg.eval_judge}, budget ${args.budget:.2f}",
          file=sys.stderr)

    def progress(i, n, meta, spent):
        print(f"  [{i}/{n}] {meta.get('tier', '?'):<8} {meta.get('model', '?'):<34} spent ${spent:.4f}", file=sys.stderr)

    result = run_batch(llm, prompts, lc._State.evaluator, args.budget, progress)
    records = [r for r in levrouter._UsageState.memory.query() if r.kind == "eval"]
    report = u.aggregate_eval(records, by=args.by)
    if args.json:
        print(json.dumps({"run": result, "quality": report, "log": log_path}, indent=2))
        return 0
    print()
    print(u.format_eval_table(report, title="Quality of routing on your prompts"))
    print(f"\nprompts {result['prompts']} (kept on the configured model: {result['same_model']}) · "
          f"spent ${result['spent_usd']:.4f} of ${args.budget:.2f}" + (f" · stopped: {result['stopped']}" if result["stopped"] else ""))
    _print_regressions(log_path, args.regressions)
    print(f"details (redacted prompts and answers): {log_path}")
    return 0


def _print_regressions(path: str, n: int) -> None:
    try:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    except OSError:
        return
    worse = [r for r in rows if r.get("verdict") == "baseline_better"][:n]
    if not worse:
        print("\nno regressions: every evaluated answer held up")
        return
    print(f"\nRegressions ({len(worse)} shown): the configured model answered better")
    for r in worse:
        print(f"  [{r['call_id']}] {r['routed_model']} vs {r['baseline_model']}: {r.get('reason', '')}")
        print(f"      prompt: {r['prompt'][:160]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="levrouter")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("setup", help="interactive setup wizard for levrouter configuration")
    sub.add_parser("check", help="check the lev prerequisite, device, provider packages and usage store")
    p = sub.add_parser("usage", help="token usage and estimated cost (default: last 24h from the rolling files)")
    p.add_argument("--since", default="24h", help="window: 30m, 6h, 24h, or an ISO time (max: retention, 24h)")
    p.add_argument("--by", default="model", choices=u.GROUP_KEYS, help="group rows by this field")
    p.add_argument("--user", help="only calls tagged metadata={'levrouter_user': ...} with this value")
    p.add_argument("--task", help="only calls with this task label")
    p.add_argument("--live", action="store_true", help="ask running app processes (in-memory, until restart)")
    p.add_argument("--dir", help="levrouter home to read (default: $LEVROUTER_HOME or ~/.levrouter)")
    p.add_argument("--recent", type=int, default=0, metavar="N", help="also list the last N calls")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-v", "--verbose", action="store_true", help="with --live: one line per process")
    r = sub.add_parser("run", help="run a program with levrouter enabled (no code changes)")
    r.add_argument("--mode", choices=["route", "eval"], help="sets LEVROUTER_MODE for the program")
    r.add_argument("--backend", choices=["lev-local", "lev-http", "heuristic"], help="sets LEVROUTER_BACKEND")
    r.add_argument("command_args", nargs=argparse.REMAINDER, help="-- the command to run")
    m = sub.add_parser("models", help="keep model data current: prices, capabilities and the models your keys can use")
    m.add_argument("action", choices=["refresh"], help="refresh: OpenRouter catalogue + each provider's model list")
    o = sub.add_parser("openrouter", help="OpenRouter catalog commands")
    o.add_argument("action", choices=["refresh"], help="refresh: download current models, prices and input types")
    ev = sub.add_parser("eval", help="evaluation: does the routed model hold up against the configured one?")
    evs = ev.add_subparsers(dest="action", required=True)
    er = evs.add_parser("run", help="route prompts from a file, call both models, judge them (makes paid API calls)")
    er.add_argument("file", help=".txt (one prompt per line) or .jsonl ({prompt, system?, task?})")
    er.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "groq", "deepseek",
                                                          "mistral", "xai", "openrouter"])
    er.add_argument("--model", required=True, help="the model your code configures (the baseline)")
    er.add_argument("--budget", type=float, default=0.50, help="stop when this much has been spent (USD, default 0.50)")
    er.add_argument("--limit", type=int, default=0, help="use only the first N prompts")
    er.add_argument("--judge", help='judge model (default "top": the provider\'s top tier)')
    er.add_argument("--backend", choices=["lev-local", "lev-http", "heuristic"])
    er.add_argument("--by", default="tier", choices=u.GROUP_KEYS)
    er.add_argument("--regressions", type=int, default=5, help="show up to N regressions")
    er.add_argument("--log", help="where to write the redacted details (default ~/.levrouter/eval-<time>.jsonl)")
    er.add_argument("--json", action="store_true")
    ep = evs.add_parser("report", help="quality from evaluated calls in the usage files")
    ep.add_argument("--since", default="24h")
    ep.add_argument("--by", default="tier", choices=u.GROUP_KEYS)
    ep.add_argument("--regressions", type=int, default=0, help="with --log: show up to N regressions")
    ep.add_argument("--log", help="an eval log file (eval_log_path) for the regression list")
    ep.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "setup":
        return setup_cmd()
    if args.command == "eval":
        return eval_cmd(args)
    if args.command == "usage":
        return usage_cmd(args)
    if args.command == "run":
        return run_cmd(args)
    if args.command == "openrouter":
        return openrouter_cmd(args)
    if args.command == "models":
        return models_cmd(args)
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
