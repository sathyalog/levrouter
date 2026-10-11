# Security, privacy and guardrails

## What leaves your process

| Data | Where it goes |
|---|---|
| Prompts | Only to the LLM provider your code already calls. levrouter adds no other destination. |
| Classifier input, `lev-local` (default) | Stays in your process. |
| Classifier input, `lev-http` | Only to a local or private-network server. A remote server needs `allow_remote_lev=True` **and** https. PII is redacted before sending, and redirects are refused. |
| Logs, JSONL call log, OpenTelemetry spans | A **redacted** prompt preview (logs and JSONL only, `log_prompt_chars=0` removes it), task and user labels, models, tokens, cost, redacted error text. Spans never contain prompt text. |
| Usage files and live socket | No prompt text. Local files (0600) and a local socket in a 0700 folder. No network port. |
| Decision cache | Keyed by a SHA-256 digest; no prompt text is kept. |

## PII redaction

`levrouter.pii.redact()` runs on everything levrouter writes. It masks:

- **Contact details:** emails, phone numbers
- **Financial:** payment card numbers (Luhn-checked), IBANs
- **Government IDs:** US SSNs, Indian Aadhaar and PAN numbers
- **Network:** IPv4 addresses
- **Secrets:**
  - private keys, bearer tokens, JWTs;
  - `password=` / `api_key=` style assignments;
  - provider keys: Anthropic, OpenAI, Google, Groq, Hugging Face, xAI, AWS, GitHub, Slack and Stripe.

**Limits:**
- The redactor matches patterns. It catches structured identifiers, not names or free-text descriptions of people.
- It errs on the side of redacting: long digit runs, such as order numbers, may be masked as phone numbers.
- Treat it as defence in depth, not as anonymisation.

**If you have GDPR, DPDP, HIPAA or similar obligations:**
- set `log_prompt_chars=0`;
- keep `log_path` and `LEVROUTER_HOME` on access-controlled storage;
- cover the provider call itself under your existing data agreements with that provider.

## API keys

- **levrouter never stores or logs provider API keys.** They stay inside your chat model objects as `SecretStr`. The one place it reads them is `levrouter models refresh`, which sends each key only to that provider's own model-list endpoint, to list the models the key can use.
- **Same key, same provider.** The per-call copy uses the same key and endpoint. The model is only ever switched to
  another model from the same provider, so a key is never sent to a different provider.
- **One credential of its own.** The only one levrouter handles is the optional `lev serve` token (`lev_api_key` /
  `LEVROUTER_LEV_API_KEY`). It is excluded from `repr` and logs. Keep it in the environment: a `lev_api_key` in a config
  file is ignored with a warning, because config files get committed.
- **Tested.** The test suite checks that a key typed into a prompt never appears in stderr, the JSONL file or the usage
  files.

## Guardrails

These rules decide whether a call may be routed, and to which model.

1. **The endpoint decides the provider.** The provider is worked out from the chat model's endpoint, not its class.
   - A `ChatOpenAI` on `api.groq.com` is routed with Groq models.
   - An unrecognised endpoint (OpenRouter, a proxy, a self-hosted server) is not routed, and one `skip` line is logged.
     Map your own gateway with `provider_hosts`.
   - Hugging Face dedicated endpoints and local pipelines serve one fixed model, so they are never routed.
2. **Same provider only.** Routing never crosses providers.
3. **Fine-tunes stay put.** Models matching `never_route` (`ft:*`, `*tunedModels/*`, `*:ft-*`, `accounts/*` by
   default) are never swapped.
4. **Capability check.** If the chosen model can't serve the request, the next tier up is used. The checks are:
   - the estimated prompt plus `max_tokens` fits its context window;
   - it has vision if the request has images;
   - it supports tool calling if the request has tools.

   - it accepts a forced tool call when the request forces one. `with_structured_output` does this by default, and
     Claude Opus 5.5 and Fable 5.1 reject it with a 400 error.

   If no higher tier fits, cheaper tiers are tried. If no tier fits, your configured model is kept.
5. **Quality floor and cost cap.**
   - `min_tier` stops routing below a tier.
   - `allow_upgrade=False` never picks a pricier model than the one you configured. That protects against prompts
     written to talk the classifier into an expensive model.
6. **Settings are adjusted on the copy only.** See [configuration](configuration.md#settings-adjusted-on-the-routed-copy).
7. **Fail at startup, not mid-request.** Invalid configuration raises when the config is built.

## lev-http endpoint rules

- **Always allowed:** `localhost`, loopback, private-network IPs, `*.local` / `*.internal` / `*.localhost` / `*.svc` hosts, and
  service names with no dot that start with a letter (Docker Compose `lev`, Kubernetes `lev-service`). A public host always
  contains a dot, so a dotless name can only resolve through your local or cluster DNS. Numeric forms such as `134744072`
  are not accepted, because they resolve to public addresses.
- **The check is by name only,** with no DNS lookup. If you can't trust the resolver a service name goes through, give the
  server an IP address or use `allow_remote_lev=True` with `https://`.
- **Anything else** needs `allow_remote_lev=True`, and the URL must use `https://`.
- **Redirects are refused,** so a prompt can't be forwarded to another host.
- **Redaction:** the state is redacted before sending unless `redact_classifier_input=False`.

## Reporting a vulnerability

Please report security issues privately to the maintainer rather than in a public issue.

---

*levrouter is powered by [lev](https://github.com/InterfazeAI/lev) from the InterfazeAI team.*
