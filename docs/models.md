# Keeping model data current

New models and dated versions appear constantly. levrouter uses model data for prices (cost and savings logs, `allow_upgrade`, OpenRouter ladder mode) and for guardrails (context size, input types, tool support). It takes that data from three layers, each overriding the one before:

| Layer | What | Where |
|---|---|---|
| 1. Built in | Default tier models, with their prices and capabilities | Shipped with levrouter (`providers.py`, `data/openrouter_models.json`) |
| 2. Refreshed | Current prices, context sizes and input types for ~200 models, plus the models each of your API keys can use | `~/.levrouter/openrouter_models.json`, `~/.levrouter/available_models.json` (or `$LEVROUTER_HOME`) |
| 3. Yours | The model for each tier | `models:` in `levrouter.yaml` |

## Refresh

```bash
levrouter models refresh
```

1. **Prices and capabilities:** downloads OpenRouter's public model catalogue. No key is needed. Direct providers use it for any model missing from the built-in list, so a new `claude-opus-4-8` gets its price and guardrail checks. Dated ids such as `claude-sonnet-4-5-20250929` match their undated name.
2. **Available models:** for each provider whose key is set, lists the models that key can use. The keys are `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY` and `XAI_API_KEY`. Providers without a key are skipped, and keys are never printed. The `levrouter` command doesn't read `.env` files, so load the keys into your shell first: `set -a; source .env; set +a`.

`levrouter openrouter refresh` still updates the catalogue alone.

## Startup check

When `available_models.json` exists, `load_config()` warns about any tier model your key can't use, and suggests the newest model with the same name, for example:

```
⚠️  levrouter: anthropic standard tier uses 'claude-sonnet-5', which isn't in the models your anthropic key
    can use (newest similar: 'claude-sonnet-5-5'). Set it under `models:` in levrouter.yaml, or run `levrouter models refresh`.
```

It only reads the saved file: startup makes no network calls and never waits.

## Why tiers don't update themselves

A new model can answer differently and cost differently, so levrouter never swaps a tier model on its own. It tells you, and you choose:

```yaml
models:
  standard:
    providers: [anthropic/claude-sonnet-5-5]
```

Run the refresh weekly or in CI, so the warnings show up before a provider retires a model.
