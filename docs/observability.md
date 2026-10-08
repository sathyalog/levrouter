# Observability: logs, traces and usage

## Log lines

With routing enabled, every call writes a `route` line (the decision) and a `done` line (tokens, latency, cost) to
stderr. A failed call writes a `fail` line instead. A summary is logged when the process exits.

```
22:18:31 tokentriage INFO  route [b0f129] task=faq "What's the capital of Australia?" -> claude-haiku-4-5 (simple; configured claude-opus-5-5) via lev-local 38ms
22:18:31 tokentriage INFO  done  [b0f129] claude-haiku-4-5 in=900 out=400 1.12s $0.0029 (saved $0.0087 vs claude-opus-5-5)
22:18:34 tokentriage INFO  summary 2 calls (claude-haiku-4-5 x1, claude-sonnet-5 x1) | cost $0.0087 vs $0.0232 if unrouted | saved $0.0145 (62.5%)
```

- **`[b0f129]`** is the call id. It links a `route` line to its `done` or `fail` line.
- **`task=`** is the first of these that is set:
  1. `metadata["tokentriage_task"]`
  2. the LangGraph node name (`langgraph_node`)
  3. `metadata["task"]`
  4. the first user tag on the run
- **Streamed calls** (LangChain, Anthropic SDK, OpenAI SDK) get their `done` line, with tokens and cost, when the stream ends or is closed. A stream you abandon without closing is recorded when it is garbage-collected; one cut off early shows the tokens received so far. For OpenAI streams tokentriage adds `stream_options={"include_usage": True}` (only for OpenAI itself, and only if you did not set `stream_options`) and hides the extra usage-only chunk from your code.
- **`user=`** appears when `metadata["tokentriage_user"]` (or `user_id`) is set.
- **The quoted text** is the start of the user prompt, already redacted. Set `log_prompt_chars=0` to leave it out.
- **`via lev-local` / `via heuristic`** says what made the decision. `configured` is the model your code asked for.
- **`skip`** lines appear once per reason when a call isn't routed. Examples: an unknown endpoint, a provider that
  isn't enabled, or a fine-tuned model.
- **At `DEBUG`,** a `reason` line explains each tier choice, e.g. `complex score 0.81 >= 0.5`.

Name tasks in your own code:

```python
llm.invoke(question, config={"metadata": {"tokentriage_task": "faq"}})
chain.invoke(inputs, config={"tags": ["contract_summary"]})
```

**Using your own logging:** set `log_level=None`. tokentriage then attaches no handler, and records flow through the
`tokentriage` logger into your app's logging setup. Use `log_format="json"` for log aggregators; every line is then one
JSON object with an `event` field (`route`, `done`, `error`, `summary`).

## Other places the decision appears

- **Response metadata:** `msg.response_metadata["tokentriage"]` holds provider, tier, model, original model, lev
  probabilities, reason, call id, task and user.
- **LangSmith:** the same metadata shows in each run's outputs. JSON logs include `run_id`, so you can match a log line
  to its LangSmith run.
- **JSONL call log:** `log_path="tokentriage.jsonl"` appends one object per call:
  - call_id, run_id, task, user;
  - the redacted prompt preview;
  - tier, models and lev probabilities;
  - tokens, latency, cost and the redacted error.

  The file is created with 0600 permissions.
- **OpenTelemetry:** with the `[otel]` extra and a tracer provider configured, each call emits a `tokentriage.call` span.
  - Its attributes: `tokentriage.model`, `tokentriage.original_model`, `tokentriage.provider`, `tokentriage.tier`,
    `tokentriage.task`, `tokentriage.user`, `tokentriage.p_<tier>`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`,
    `gen_ai.usage.output_tokens` and `tokentriage.cost_usd`.
  - Spans never contain prompt text.
- **In code:** `tokentriage.stats()` returns lifetime totals for the process: calls, by model/tier/task, cost against the
  unrouted baseline.

## Token usage and cost, last 24 hours

Usage is kept in two places, both read with `tokentriage usage`. There is no web route and no network port.

| | Rolling files (default) | Live memory (`--live`) |
|---|---|---|
| Location | `~/.tokentriage/usage/` (or `$TOKENTRIAGE_HOME/usage/`) | inside each running app process |
| Survives app restart | yes | no |
| Combines all workers on the machine | yes, they share the files | yes, the CLI asks each process |
| Needs the app running | no | yes |

```bash
tokentriage usage                          # last 24h by model, from the rolling files
tokentriage usage --by task --since 6h     # or: provider, tier, user
tokentriage usage --user u123              # one end user
tokentriage usage --recent 20              # also list the last 20 calls
tokentriage usage --live                   # ask the running processes' memory
tokentriage usage --json                   # for scripts and CI
```

### How the rolling files work

```
~/.tokentriage/usage/
  2026-09-28T13.jsonl    <- oldest hour kept
  ...
  2026-09-29T12.jsonl    <- current UTC hour
```

- **Every finished call** appends one line to the current hour's file, straight away.
- **Cleanup:** when a process starts writing a new hour, and whenever the CLI reads, files older than
  `usage_retention_hours` are deleted whole. At most 25 files exist at a time.
- **Safe with several workers:** nothing is ever rewritten, so workers can append to the same file, and a crash loses at
  most one line.
- **Exact window:** reads filter by timestamp, so "last 24h" is exact even though the oldest file is only partly inside
  the window.

### How the live view works

- Each process keeps its last 24 hours in memory (up to `usage_memory_max` calls).
- It answers over a Unix socket file in `~/.tokentriage/run/`, a folder only your user can open. It is not a network port.
- `tokentriage usage --live` asks every live socket and merges the answers. It also removes socket files left behind by
  processes that crashed.
- On platforms without Unix sockets, live mode switches itself off and the file view still works.

### In your application

```python
llm.invoke(question, config={"metadata": {"tokentriage_user": current_user.id}})

tokentriage.usage(user=current_user.id, since="24h")   # dict: totals + rows, from this process's memory
print(tokentriage.usage_report(by="task"))             # the same table as the CLI
```

### Docker, servers and Kubernetes

Full examples for each environment: [deployment.md](deployment.md).

Run the CLI where the app runs:

```bash
docker exec <container> tokentriage usage
ssh <server> tokentriage usage
kubectl exec <pod> -- tokentriage usage
```

- **Keeping files across container restarts:** put the tokentriage home on a volume, for example
  `-e TOKENTRIAGE_HOME=/data/tokentriage -v ./data:/data`. You can then also read it from the host:
  `tokentriage usage --dir ./data/tokentriage`.
- **Several replicas:** each replica reports its own usage. For a combined figure, ship the JSON log lines or the
  OpenTelemetry spans to your log or monitoring platform.

### What the usage records contain

- **Included:** time, call id, provider, models, tier, task and user labels (redacted), token counts, latency,
  estimated cost, and the error type.
- **Not included:** prompt text and error messages.
- **Permissions:** files are 0600 and folders 0700.
- **Costs are estimates** from tokentriage's price table. "Unrouted" means the same tokens priced on the model your code
  configured.

---

*tokentriage is powered by [lev](https://github.com/InterfazeAI/lev) from the InterfazeAI team.*
