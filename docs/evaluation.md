# Evaluation mode

levrouter saves money by sending easy requests to cheaper models. Evaluation mode answers the question that
follows: **do the cheaper answers hold up?** It measures this on your own prompts and traffic.

## Three ways to use it

**1. On a file of prompts (before touching your app)**

```bash
levrouter eval run prompts.jsonl --provider openrouter --model anthropic/claude-opus-5.5 --budget 0.50
```

- **Input:** a `.txt` file (one prompt per line) or a `.jsonl` file with `{"prompt": ..., "system": ..., "task": ...}`.
- **What happens:** every prompt is routed. The routed model answers, the model you pass with `--model` answers too,
  and the judge compares them.
- **Stopping:** the run stops when `--budget` (USD) is spent.
- **Output:** a quality table by tier (`--by task|model|...`), the regressions, and a redacted details file.
- **Keys:** read from the environment as usual, e.g. `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY`.

**2. In your app, with zero risk**

```bash
LEVROUTER_MODE=eval levrouter run -- python app.py     # or RouterConfig(mode="eval")
```

- With the default `eval_serve="baseline"`, users keep getting the configured model's answer.
- For a sample of calls (`eval_sample_rate`, 20% by default), the routed model is also asked, in a background thread,
  and the two answers are compared.
- Nothing is delayed. Evaluation errors are logged and never reach the app.

**3. Reading the results**

```bash
levrouter usage --by task                       # usage table + "Quality of routing" table
levrouter eval report --by tier                 # quality only
levrouter eval report --log eval.jsonl --regressions 10
```

In code, `levrouter.eval_report(by="task")` returns the same numbers.

## How answers are compared

- **Text answers:** a judge model rates both answers from 1 to 5 and picks a winner. It does this **twice, with the
  order swapped**, because LLM judges tend to favour whichever answer comes first:
  - both passes agree on a winner → `routed_better` or `baseline_better`;
  - a tie or disagreement → `equivalent`.
- **Tool calls and structured output** (`with_structured_output`, `bind_tools`): compared field by field against the
  configured model's answer, with no judge call.
  - Tool names must match.
  - 90% or more of fields equal → `equivalent`, otherwise `baseline_better`.
  - The routed score is 1 + 4 × (share of matching fields).
- **Held** means `equivalent` or `routed_better`. That's the headline number.

**The field comparison has a noise floor.** Models aren't fully deterministic, even at temperature 0. In a real
trial, Claude Sonnet 5 matched *its own* structured output on only 89–94% of fields across four resume-parsing cases,
and one of three self-comparisons fell under the 90% bar. So treat a single `baseline_better` on structured output as a
hint, not proof. Look at the rate over many calls, and at which fields differ (in `eval_log_path`).

## The judge

- **Default:** `eval_judge="top"`, the same provider's top tier, e.g. Claude Opus judges Claude answers. Prompts and
  answers go only to the provider your app already uses, with the key it already has.
- **A specific model:** `eval_judge="anthropic/claude-sonnet-5"` or any model served by the same endpoint. That's
  cheaper, and slightly less reliable.
- **Limits:** the judge is an LLM. It can be wrong, especially on specialised or subjective tasks, so review the
  listed regressions.

## Cost and budget

On each evaluated call, evaluation pays for the second model's answer plus two judge calls. `eval_budget_usd`
(default $1 per rolling 24 hours) caps that extra spend. When it's reached, evaluation pauses with one log line and
serving continues normally. In batch runs, `--budget` caps the whole run, including the routed answers.

## Privacy

- **Eval rows** in the usage files hold only models, verdicts, scores and costs, never text.
- **Prompts and answers** are written only if you set `eval_log_path` (batch runs create one under `~/.levrouter/`).
  They're PII-redacted, and the file is created with 0600 permissions.

---

*levrouter is powered by [lev](https://github.com/InterfazeAI/lev) from the InterfazeAI team.*
