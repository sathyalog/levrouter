# Configuration

Pass a `RouterConfig` to `tokentriage.enable()` or `tokentriage.route()`. If you pass none, `enable()` builds one from
`tokentriage.yaml` (found the way `load_config()` finds it) and a few environment variables; see
[deployment.md](deployment.md).

```python
import tokentriage

tokentriage.enable(tokentriage.RouterConfig(
    backend="lev-local",
    min_tier="standard",
    log_path="tokentriage.jsonl",
    tiers={"openai": {"simple": "gpt-4o-mini", "standard": "gpt-6-sol", "complex": "gpt-6-astra"}},
))
```

You can also load a YAML file (needs the `[yaml]` extra):

```python
tokentriage.enable(tokentriage.RouterConfig.from_yaml("tokentriage.yaml"))
```

## All options

### Classifier

| Option | Default | Meaning |
|---|---|---|
| `backend` | `"lev-local"` | `"lev-local"` (lev in-process), `"lev-http"` (a `lev serve` server) or `"heuristic"` (keyword rules, no model) |
| `lev_checkpoint` | `"interfaze-ai/lev"` | Hugging Face id or local path passed to `lev.load()` |
| `lev_url` | `"http://localhost:8000"` | `lev serve` address for `backend="lev-http"` |
| `lev_api_key` | `None` | Bearer token for a protected `lev serve`. Set it in code only; put it in `TOKENTRIAGE_LEV_API_KEY` otherwise (in a config file it is ignored with a warning). Never logged. |
| `lev_headers` | `{}` | `lev-http` only: extra request headers as `{Header-Name: ENV_VAR_NAME}`. The file holds only variable *names*; the values are read from the environment, so nothing secret is committed. Example for Modal: `{Modal-Key: MODAL_KEY, Modal-Secret: MODAL_SECRET}`. An unset variable skips that header with a warning. |
| `warm_on_load` | `False` | `lev-http` only: `load_config()` sends one background request so a scale-to-zero server starts before the first real call. It never blocks startup; a failure is logged and ignored. The request is billable, so it is off by default. |
| `timeout_s` | `1.5` | Seconds per decision. On timeout the heuristic decides that call. |
| `block_on_load` | `False` | `False`: the first call starts loading lev in the background, and the heuristic routes until lev is ready |
| `cache_size` | `1024` | Decisions cached, keyed by a SHA-256 of the request state |
| `sticky_threads` | `True` | Keep a conversation on one model so the provider's prompt cache keeps working. The thread id comes from LangGraph's `thread_id` or `metadata={"tokentriage_thread": ...}`; a thread can move up a tier, never down |
| `thread_ttl_s` | `300` | Seconds an idle conversation keeps its model; matches Anthropic's default prompt-cache lifetime (use `3600` with the 1-hour cache) |
| `escalate_after_repeats` | `2` | When the latest question repeats this many earlier ones (rephrased, or "that's wrong, try again"), move up a tier, and one more per further repeat. `0` turns it off |
| `max_state_chars` | `2000` | Characters of the user message sent to the classifier (start and end are kept) |

### Tiers and thresholds

| Option | Default | Meaning |
|---|---|---|
| `tiers` | see the tier table in the [README](../README.md#how-it-works) | `{provider: {"simple": id, "standard": id, "complex": id}}`. Merged over the defaults. |
| `complex_threshold` | `0.5` | Complex if `P(complex) + 0.25·P(reasoning) + 0.15·P(code)` reaches this |
| `simple_threshold` | `0.55` | Simple if `P(simple)` reaches this and reasoning is unlikely |
| `min_confidence` | `0.45` | Below this top probability, the decision moves one tier up |

### Guardrails

| Option | Default | Meaning |
|---|---|---|
| `allow_upgrade` | `True` | `False`: never route to a model pricier than the one your code configured |
| `min_tier` | `None` | Quality floor, e.g. `"standard"` |
| `never_route` | `("ft:*", "*tunedModels/*", "*:ft-*", "accounts/*", "openrouter/*", "*:free", "~*", "*:batch")` | Model patterns that are never swapped: fine-tunes, custom deployments, OpenRouter routers, free models, floating aliases |
| `provider_hosts` | `{}` | Map extra endpoint hosts to a provider, e.g. `{"llm-gateway.corp.com": "anthropic"}` |
| `capability_guard` | `True` | Move up a tier when a model lacks the context window, tool calling, or an attached input type (image, PDF/file, audio, video) |
| `redact_pii` | `True` | Redact PII and secrets in everything tokentriage writes |
| `allow_remote_lev` | `False` | Allow a non-local `lev_url` (https is still required) |
| `redact_classifier_input` | `True` | Redact the state sent to a `lev-http` server |

### OpenRouter

| Option | Default | Meaning |
|---|---|---|
| `openrouter_mode` | `"family"` | `"family"`: stay with your model's vendor. `"ladder"`: cheapest capable model across `openrouter_vendors` |
| `openrouter_vendors` | `None` | Vendors the ladder may use, e.g. `("anthropic", "google")`; `None` = all known families |
| `openrouter_families` | built-in | Per-vendor tier overrides, e.g. `{"anthropic": {"simple": "anthropic/claude-haiku-4.5"}}` |

### Evaluation

| Option | Default | Meaning |
|---|---|---|
| `mode` | `"route"` | `"eval"` also runs the other model on sampled calls and judges the two answers |
| `eval_sample_rate` | `0.2` | Share of calls evaluated (chosen deterministically per call id) |
| `eval_serve` | `"baseline"` | Answer the app receives in eval mode: `"baseline"` (the configured model) or `"routed"` |
| `eval_budget_usd` | `1.0` | Cap per 24 h on the extra spend (second call + judge); evaluation pauses when reached |
| `eval_judge` | `"top"` | Judge model: the same provider's top tier, or a model id served by the same endpoint and key |
| `eval_log_path` | `None` | Opt-in JSONL of redacted prompts and both answers, for reviewing regressions (0600) |

### Logging and tracing

| Option | Default | Meaning |
|---|---|---|
| `log_level` | `"INFO"` | Level of the `tokentriage` logger, printed to stderr. `None` = attach no handler and use your app's logging |
| `log_format` | `"text"` | `"text"` or `"json"` |
| `log_prompt_chars` | `60` | Characters of the redacted prompt shown in log lines. `0` hides prompts. |
| `log_path` | `None` | Append one JSON object per call to this file (0600) |
| `summary_at_exit` | `True` | Log a per-model / per-task summary when the process exits |
| `otel` | `True` | Emit an OpenTelemetry span per call when `opentelemetry-api` is installed |

### Usage (last 24h)

| Option | Default | Meaning |
|---|---|---|
| `usage_file` | `True` | Rolling hourly usage files, read by `tokentriage usage` |
| `usage_live` | `True` | In-memory usage answered over a local socket, read by `tokentriage usage --live` |
| `usage_retention_hours` | `24` | Window kept in the files and in memory |
| `usage_memory_max` | `100000` | Maximum calls kept in memory per process |
| `usage_home` | `None` | Folder for usage files and sockets. Default: `$TOKENTRIAGE_HOME`, else `~/.tokentriage` |

## Environment variables

Settings live in `tokentriage.yaml` (every option above is a key under `router:`) or in the `RouterConfig` you pass in
code. Only these environment variables are read:

| Variable | Purpose |
|---|---|
| `TOKENTRIAGE_CONFIG` | Path to the config file, instead of searching for `tokentriage.yaml` |
| `TOKENTRIAGE_HOME` | The data folder (default `~/.tokentriage`) |
| `TOKENTRIAGE_ENABLED` | Kill switch: any value other than `1`, `true`, `yes` or `on` turns routing off |
| `TOKENTRIAGE_LOG_LEVEL`, `TOKENTRIAGE_BACKEND`, `TOKENTRIAGE_MODE` | Override `log_level`, `backend` and `mode` from the file |
| `TOKENTRIAGE_LEV_API_KEY` | Secret: bearer key for a protected `lev serve` |

The override variables beat the file, and the file beats the defaults. Earlier versions also accepted
`TOKENTRIAGE_<OPTION>` for every option, `TOKENTRIAGE_<PROVIDER>_<TIER>`, `TOKENTRIAGE_PROVIDER_HOSTS`,
`TOKENTRIAGE_FRAMEWORKS` and `TOKENTRIAGE_PROVIDERS`; these are now ignored with a warning, and the setting goes in the
YAML. What each variable does, and the old-to-new mapping: [deployment.md](deployment.md).

Invalid values fail when the config is built, before any call is routed. Examples: an unknown `min_tier`, a tier map
missing a tier, or a remote `lev_url` that isn't allowed.

## Per-call control

Per-call control goes through LangChain's standard `config` argument:

```python
llm.invoke(q, config={"metadata": {"tokentriage_tier": "complex"}})    # force a tier for this call
llm.invoke(q, config={"metadata": {"tokentriage_disable": True}})      # use the configured model as-is
llm.invoke(q, config={"metadata": {"tokentriage_task": "faq"}})        # label the task (logs, usage)
llm.invoke(q, config={"metadata": {"tokentriage_user": user.id}})      # attribute usage to an end user
```

For whole instances:

```python
tokentriage.route(llm)      # opt one instance in (instead of enable())
tokentriage.exclude(llm)    # never route this instance, even after enable()
tokentriage.disable()       # restore the original LangChain methods
```

## Settings adjusted on the routed copy

When a cheaper model would reject a setting, tokentriage changes it on the per-call copy only. Your instance is never
modified.

- **`max_tokens` / `max_output_tokens`:** clamped to the target model's limit, for example Haiku 4.5's 64k.
- **Anthropic thinking:**
  - adaptive thinking is removed when the target is Haiku 4.5;
  - `budget_tokens` thinking becomes adaptive on Sonnet 5 / Opus 5.5.
- **OpenAI reasoning:** `reasoning_effort` / `reasoning` are removed for `gpt-4o*` targets.

---

*tokentriage is powered by [lev](https://github.com/InterfazeAI/lev) from the InterfazeAI team.*
