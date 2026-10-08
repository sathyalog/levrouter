# Deployment and environment variables

**The rule:** every setting lives in `tokentriage.yaml`. The environment holds only four kinds of things: **secrets**, **locations**, a **kill switch**, and three **quick overrides**. This page lists each variable, shows how to configure tokentriage in each kind of environment, and says what happens when something goes wrong.

## Environment variables

| Variable | What it does | Default | Set it when |
|---|---|---|---|
| `TOKENTRIAGE_CONFIG` | Path to the config file (`.yaml`, `.yml` or `.toml`) | the file is searched for automatically (see below) | you keep one file per environment (`tokentriage.prod.yaml`), a Kubernetes ConfigMap is mounted outside the app folder, or the app starts from another folder |
| `TOKENTRIAGE_HOME` | The data folder (see below) | `~/.tokentriage` | the data should live on a mounted volume, or the home folder isn't writable |
| `TOKENTRIAGE_ENABLED` | **Kill switch.** Any value other than `1`, `true`, `yes` or `on` turns routing off | unset (routing on) | something looks wrong in production and you want routing off without a file change or redeploy |
| `TOKENTRIAGE_LOG_LEVEL` | Overrides `router.log_level` (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | the file's value | you need `DEBUG` (it logs why each model was picked) without editing the file |
| `TOKENTRIAGE_BACKEND` | Overrides `router.backend` (`lev-local`, `lev-http`, `heuristic`) | the file's value | an emergency fallback, for example `heuristic` while lev is down |
| `TOKENTRIAGE_MODE` | Overrides `router.mode` (`route`, `eval`) | the file's value | running [evaluation mode](evaluation.md); `tokentriage run --mode eval` sets it for you |
| `TOKENTRIAGE_LEV_API_KEY` | **Secret.** Bearer key sent to a protected `lev serve` server (`lev-http` backend) | none | the lev server checks a key |
| any name you choose | **Secret.** Values for `router.lev_headers`, which maps header names to variable names (for example `Modal-Key: MODAL_KEY`); see the README's lev-http authentication notes | none | the lev server wants headers other than a Bearer key (Modal proxy auth, Cloudflare Access, a gateway) |

An invalid value for `TOKENTRIAGE_BACKEND`, `_MODE` or `_LOG_LEVEL` is ignored with a warning, never silently applied.

Startup messages (warnings about the config, `models:` entries, removed variables, and so on) go through the `tokentriage` logger and honour `log_level`. Before routing is enabled they are written to stderr, never stdout.

**Variables that belong to other software** (tokentriage reads some, the rest are read by the libraries it routes):

| Variable | Read by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and the other providers' keys | the LangChain and SDK clients | **Secrets.** `tokentriage models refresh` also reads them, to list the models each key can use |
| `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `GROQ_API_BASE`, `OPENROUTER_BASE_URL` | the SDKs; tokentriage reads them to tell which provider a client talks to | custom endpoints |
| `HF_HOME`, `HF_TOKEN`, `HF_HUB_OFFLINE` | Hugging Face, when lev is downloaded or loaded | where the ~9.5 GB lev files are cached, a download token, offline mode |
| `TOKENTRIAGE_AUTOENABLE` | set by `tokentriage run` itself | internal; don't set it |

### Precedence

1. The override variables: `TOKENTRIAGE_ENABLED`, `_BACKEND`, `_MODE`, `_LOG_LEVEL`.
2. `tokentriage.yaml` (or the file named by `TOKENTRIAGE_CONFIG`).
3. The defaults.

Settings you pass in code (`tokentriage.enable(RouterConfig(...))`) are used exactly as given; only `TOKENTRIAGE_ENABLED=false` still applies to them, so the kill switch always works.

### How each entry point reads settings

| You run | Settings come from |
|---|---|
| `tokentriage.load_config()` | the file, then the overrides. Also prepares lev and enables routing |
| `tokentriage.enable()` with no arguments | the same file and overrides, including its `frameworks` and `providers`. (Earlier versions ignored the file here and used defaults.) |
| `tokentriage.enable(RouterConfig(...))` | what you pass |
| `tokentriage run -- python app.py` | the file and overrides, with no code changes in the app; `--mode` and `--backend` set the matching variables |
| `tokentriage check`, `usage`, `eval` | the file and overrides. `check` prints the file's path and the real values |

**Finding the file.** `load_config()` and the other entry points look for `tokentriage.yaml`, `tokentriage.yml`, `tokentriage.toml` or `.tokentriage` in the working folder and up to four parent folders, then for `[tool.tokentriage]` in a `pyproject.toml`. `TOKENTRIAGE_CONFIG`, or `load_config("path/to/file.yaml")`, skips the search. `tokentriage.yaml` works without any variable.

### `TOKENTRIAGE_HOME`: what is in the data folder

The folder is optional to set; the default is `~/.tokentriage`. It holds:

| Path | Contents |
|---|---|
| `usage/` | the rolling 24-hour usage files that `tokentriage usage` reads |
| `run/` | the sockets behind `tokentriage usage --live` |
| `openrouter_models.json`, `available_models.json` | model data from `tokentriage models refresh` |
| `eval-<time>.jsonl` | the default output of `tokentriage eval run` |

`router.usage_home` in the YAML moves `usage/`, `run/` and the default eval output; the two model files always follow `TOKENTRIAGE_HOME`. If the folder can't be created, tokentriage logs one warning and carries on without usage files or the live view.

### Secrets stay out of the file

`lev_api_key` in a config file is **ignored with a warning**, because config files get committed to git. Use `TOKENTRIAGE_LEV_API_KEY`. (`RouterConfig(lev_api_key=...)` in code still works.) Provider API keys are only ever read from the environment by the SDKs.

### Variables that no longer exist

Earlier versions let you set almost anything with `TOKENTRIAGE_<NAME>`. These are now ignored, with one warning naming them. Put the setting in the YAML:

| Old variable | Now |
|---|---|
| `TOKENTRIAGE_<NAME>` for any setting, such as `_MIN_TIER`, `_TIMEOUT_S`, `_USAGE_FILE`, `_STICKY_THREADS` | `router.<name>` |
| `TOKENTRIAGE_<PROVIDER>_<TIER>`, such as `TOKENTRIAGE_OPENAI_SIMPLE` | `models:` (for example `simple: {providers: [openai/gpt-6-luna]}`) |
| `TOKENTRIAGE_PROVIDER_HOSTS` | `router.provider_hosts` |
| `TOKENTRIAGE_FRAMEWORKS`, `TOKENTRIAGE_PROVIDERS` | `frameworks:` and `providers:` |
| `TOKENTRIAGE_USAGE_HOME` | `router.usage_home` (or `TOKENTRIAGE_HOME`) |

## Which classifier for which environment

| Environment | Recommended `backend` | Why |
|---|---|---|
| Laptop with less than ~16 GB of RAM | `heuristic`, or `lev-http` to a bigger machine | lev-local needs ~10 GB of free memory; below that it swaps and every decision times out |
| CI and tests | `heuristic` (or `TOKENTRIAGE_ENABLED=false`) | no download, no model, instant |
| One server with a GPU or 16 GB+ RAM | `lev-local` | most accurate, no network hop |
| Containers and serverless | `heuristic` or `lev-http`; avoid `lev-local` | a cold start would load a ~9.5 GB model |
| Kubernetes, many replicas | `lev-http` to one shared lev Deployment | each `lev-local` replica loads its own copy of the model |
| No internet access | `lev-local` with a pre-filled cache, or `lev-http` to an internal server | see "Air-gapped" below |

Whatever the backend, a decision that fails or exceeds `timeout_s` falls back to the heuristic for that call.

## Recipes

### Local development

```bash
tokentriage setup                       # writes tokentriage.yaml
TOKENTRIAGE_LOG_LEVEL=DEBUG python app.py   # see why each model was picked
tokentriage check                       # shows the file and what it produces
```

### CI and tests

```yaml
router:
  backend: heuristic
  usage_file: false     # no files written
  usage_live: false     # no sockets
```

Or leave the file alone and run with `TOKENTRIAGE_ENABLED=false`, so every call uses the model written in the code.

### Docker

```dockerfile
FROM python:3.12
WORKDIR /app
COPY tokentriage.yaml .          # settings are part of the image; secrets are not
COPY requirements.txt .
RUN pip install -r requirements.txt
# lev-local only: download lev at build time, so containers start without a 9.5 GB download
ENV HF_HOME=/opt/hf
RUN python -c "from tokentriage.lev_download import ensure_lev_cached; ensure_lev_cached()"
COPY . .
CMD ["python", "-m", "myapp"]
```

```bash
docker run -e ANTHROPIC_API_KEY \
           -e TOKENTRIAGE_HOME=/data/tokentriage -v ./data:/data \
           my-app
```

Mount a volume on `TOKENTRIAGE_HOME` to keep usage files across restarts. For lev-local without a build-time download, mount a volume on `HF_HOME` so the model downloads once.

### Docker Compose

With `lev-local` in the app's own container there is no URL to configure. To share one lev server between containers, run it as a service and point the app at the service name:

```yaml
# docker-compose.yml
services:
  lev:
    image: my-lev-server          # an image with lev installed, running:
    command: lev serve --checkpoint interfaze-ai/lev --host 0.0.0.0 --port 8000
    environment:
      HF_HOME: /opt/hf
    volumes: ["hf-cache:/opt/hf"]
  app:
    build: .
    environment:
      TOKENTRIAGE_LEV_API_KEY:    # taken from your shell or an .env file; never put it in tokentriage.yaml
      ANTHROPIC_API_KEY:
      TOKENTRIAGE_HOME: /data/tokentriage
    volumes: ["./data:/data"]
    depends_on: [lev]
volumes:
  hf-cache:
```

```yaml
# tokentriage.yaml, copied into the app image
router:
  backend: lev-http
  lev_url: http://lev:8000        # the Compose service name counts as a private address
  timeout_s: 2.0
```

### Kubernetes

The settings go in a ConfigMap, the keys in Secrets, and lev runs as its own Deployment so replicas share it:

```yaml
apiVersion: v1
kind: ConfigMap
metadata: {name: tokentriage-config}
data:
  tokentriage.yaml: |
    router:
      backend: lev-http
      lev_url: http://lev-service:8000   # a Service in the same namespace; see the note below
      timeout_s: 2.0
      log_format: json
    frameworks: [langchain]
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: my-app}
spec:
  template:
    spec:
      containers:
        - name: app
          image: my-app:1.0
          env:
            - {name: TOKENTRIAGE_CONFIG, value: /etc/tokentriage/tokentriage.yaml}
            - {name: TOKENTRIAGE_HOME, value: /tmp/tokentriage}
            - name: TOKENTRIAGE_LEV_API_KEY
              valueFrom: {secretKeyRef: {name: lev, key: api-key}}
            - name: ANTHROPIC_API_KEY
              valueFrom: {secretKeyRef: {name: llm-keys, key: anthropic}}
          securityContext: {readOnlyRootFilesystem: true}
          volumeMounts:
            - {name: tokentriage-config, mountPath: /etc/tokentriage}
            - {name: tmp, mountPath: /tmp}
      volumes:
        - {name: tokentriage-config, configMap: {name: tokentriage-config}}
        - {name: tmp, emptyDir: {}}
```

- **`TOKENTRIAGE_CONFIG`** is what makes the mounted file found: `/etc/tokentriage` is not in the app's working folder.
- **`readOnlyRootFilesystem`:** `TOKENTRIAGE_HOME=/tmp/tokentriage` on an `emptyDir` gives tokentriage somewhere to write. Without a writable folder it logs one warning and runs without usage files (or set `usage_file: false` and `usage_live: false`).
- **The lev server** runs `lev serve --checkpoint interfaze-ai/lev --host 0.0.0.0 --port 8000` on a GPU node or one with enough RAM. Without `--checkpoint`, `lev serve` serves the untrained base model. `lev serve` has no login of its own: put it behind something that checks the bearer key (a proxy), and give the app the same key as `TOKENTRIAGE_LEV_API_KEY`.
- **In-cluster `lev_url`:** tokentriage only sends request text to local or private addresses unless `allow_remote_lev: true` (which also requires `https://`). Kubernetes service names count as private: `http://lev-service:8000`, `http://lev-service.other-namespace.svc:8000` and the full `*.svc.cluster.local` name all work, as do IP addresses and `.internal` / `.local` names. The check is by name only (no DNS lookup).

### Serverless and short-lived jobs

Use `heuristic` or `lev-http`, never `lev-local`, so a cold start doesn't load a model. Set `usage_file: false` and `usage_live: false` if the filesystem is read-only or discarded after each run. Send the JSON logs (`log_format: json`) to your log service instead.

### Several workers or replicas

| What | Scope |
|---|---|
| The lev-local model | **per process**: N workers hold N copies (~10 GB each); use `lev-http` with several workers |
| The decision cache | per process |
| Conversation memory (`sticky_threads`) | **per process**: a conversation whose calls reach different workers or replicas loses its sticky model. Use session affinity at your load balancer, or accept per-call routing |
| Usage files | shared by the workers on one machine (appends are safe); `tokentriage usage --live` merges them |
| Usage across replicas | each replica reports its own; combine with the JSON logs or OpenTelemetry |

### Behind a corporate gateway

If your chat model calls a gateway (an API management layer) instead of the provider, tokentriage doesn't recognise the host and leaves the call alone. Tell it which provider the gateway serves:

```yaml
router:
  provider_hosts:
    llm-gateway.corp.com: anthropic
```

### Air-gapped

Fill the Hugging Face cache on a connected machine (`ensure_lev_cached()` as in the Docker recipe), copy the folder, and set `HF_HOME` to it and `HF_HUB_OFFLINE=1`. tokentriage then uses the cached lev without any network access (checked: `ensure_lev_cached()` returns normally with `HF_HUB_OFFLINE=1` and a filled cache). A `lev serve` server on the internal network works the same way for `lev-http`.

## When something goes wrong

| Situation | What happens |
|---|---|
| lev takes longer than `timeout_s`, or fails | that call uses the heuristic |
| lev isn't installed, or its download fails | `load_config()` says why and uses the heuristic |
| a bug inside tokentriage's routing | one warning, and the call goes to the model written in your code |
| the data folder can't be created | one warning; no usage files or live view, routing continues |
| the provider returns an error | passed to your code unchanged |
| `TOKENTRIAGE_ENABLED=false` | no routing at all |

## Not routed today

Only the LangChain classes listed in the README, plus the Anthropic and OpenAI SDKs, are routed. Calls through `AzureChatOpenAI`, the AWS Bedrock chat classes and the Google Vertex AI chat classes go to the model written in your code, with no message.

## Production checklist

- Pin both packages: `tokentriage @ git+https://github.com/sathyalog/tokentriage.git@<tag-or-commit>`, and lev to a commit.
- Provider keys and `TOKENTRIAGE_LEV_API_KEY` come from your secret store; none of them is in `tokentriage.yaml`.
- `timeout_s` is set to what your latency allows.
- The data folder is writable, or `usage_file` and `usage_live` are `false`.
- `log_format: json` if you collect logs.
- `tokentriage check` shows the settings you expect, and `tokentriage models refresh` runs on a schedule.
