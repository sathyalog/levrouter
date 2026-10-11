# Config-Based Zero-Code Setup

Enable levrouter for your entire project with a single configuration file—no code changes needed.

> **New in v0.1.0+:** `load_config()` now does everything automatically:
> - ✅ Validates LEV prerequisites with helpful installation guide
> - ✅ Downloads the lev model (~9.5 GB, first run only) with a progress percentage, and waits until it is complete
> - ✅ Tests LEV with sample classification
> - ✅ Enables routing for all configured frameworks
> - ✅ Falls back gracefully to heuristic if LEV unavailable
>
> See the [README](../README.md) for the full setup.

## Quick Start

### Step 1: Create `levrouter.yaml`

```yaml
# In your project root
enabled: true
frameworks:
  - langchain
  - anthropic
  - openai

router:
  backend: lev-local
  timeout_s: 2.0
```

### Step 2: Add One Line to Your Entry Point

```python
# main.py, app.py, __init__.py, or settings.py
import levrouter
levrouter.load_config()

# Everything else stays the same
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
llm1 = ChatAnthropic(model="claude-opus-5-5")
client = Anthropic()

llm1.invoke("Hi")  # ✨ Automatically routed
client.messages.create(model="claude-opus-5-5", messages=[...])  # ✨ Automatically routed
```

### Step 3: Done!

All LLM calls across your codebase are routed based on complexity.

---

## Configuration Files

### YAML Format (`levrouter.yaml`)

```yaml
enabled: true

frameworks:
  - langchain
  - anthropic
  - openai

providers:
  - anthropic
  - openai

router:
  backend: lev-local          # or lev-http, heuristic
  timeout_s: 2.0
  min_tier: null              # or standard, complex
  allow_upgrade: true
```

### TOML Format (`pyproject.toml`)

```toml
[tool.levrouter]
enabled = true
frameworks = ["langchain", "anthropic", "openai"]
providers = ["anthropic", "openai"]

[tool.levrouter.router]
backend = "lev-local"
timeout_s = 2.0
```

### Environment Variables

Settings are not set through environment variables. A few variables remain (the config path, the data folder, a kill
switch, three overrides and the secrets); see [deployment.md](deployment.md) for each one.

---

## Load Configuration Anywhere

```python
import levrouter

# Auto-finds levrouter.yaml or [tool.levrouter] in pyproject.toml
config, frameworks, providers = levrouter.load_config()

# Use the config
levrouter.enable(config=config, frameworks=frameworks, providers=providers)
```

Or let `enable()` handle it:

```python
# Enable will call load_config() internally
levrouter.enable(frameworks=["langchain", "anthropic", "openai"])
```

---

## Production Deployment

Docker, Kubernetes (ConfigMap, Secrets, a shared lev server), serverless, several workers and air-gapped setups are
covered, with working examples, in [deployment.md](deployment.md).

---

## Configuration Precedence

1. **Override variables** (`LEVROUTER_ENABLED`, `_BACKEND`, `_MODE`, `_LOG_LEVEL`)
2. **Config file** (`levrouter.yaml`, or the file named by `LEVROUTER_CONFIG` or passed to `load_config()`)
3. **Defaults** (lev-local backend, LangChain framework)

Settings passed in code (`enable(RouterConfig(...))`) are used as given; only the `LEVROUTER_ENABLED=false` kill
switch still applies to them. To use a different backend in one environment, point `LEVROUTER_CONFIG` at a different
file, or set `LEVROUTER_BACKEND` for a quick override.

---

## Configuration Schema

```yaml
enabled: bool = True                    # Enable routing
frameworks: list[str] = ["langchain"]   # Which SDKs to route (langchain, anthropic, openai)
providers: list[str] = []               # Filter providers (optional, empty = all)

router:
  backend: str = "lev-local"            # lev-local, lev-http, heuristic (default: lev-local)
  lev_checkpoint: str = "interfaze-ai/lev"  # Hugging Face model ID for lev-local
  lev_url: str = "http://localhost:8000"    # lev serve address for lev-http
  lev_api_key: str | None = None        # Bearer token for protected lev serve
  timeout_s: float = 1.5                # Timeout for lev decision (seconds)
  block_on_load: bool = False           # Block on LEV load or route with heuristic
  cache_size: int = 1024                # Decision cache size
  max_state_chars: int = 2000           # Max characters of user message sent to classifier
  
  # Thresholds for complexity classification
  complex_threshold: float = 0.5        # Complex if score reaches this
  simple_threshold: float = 0.55        # Simple if score reaches this
  min_confidence: float = 0.45          # Move up a tier if confidence below this
  
  # Guardrails
  allow_upgrade: bool = True            # Allow routing to pricier model?
  min_tier: str | None = None           # Quality floor (simple/standard/complex)
  never_route: tuple[str, ...] = (...)  # Model patterns that are never swapped
  provider_hosts: dict[str, str] = {}   # Extra endpoint hosts to treat as provider
  capability_guard: bool = True         # Move up tier if model lacks capability
  redact_pii: bool = True               # Redact PII in everything levrouter writes
  allow_remote_lev: bool = False        # Allow non-local lev_url (lev-http only)
  redact_classifier_input: bool = True  # Redact state sent to lev-http server

log_level: str = "INFO"                 # DEBUG, INFO, WARNING, ERROR
log_format: str = "text"                # text or json
log_prompt_chars: int = 60              # Characters of prompt shown in logs
log_path: str | None = None             # JSONL file for call logs
summary_at_exit: bool = True            # Log summary when process exits

# Usage tracking (last 24h)
usage_file: bool = True                 # Use rolling hourly usage files
usage_live: bool = True                 # Use in-memory live usage socket
usage_retention_hours: int = 24         # How long to keep usage data
usage_memory_max: int = 100000          # Max calls kept in memory per process
```

---

## Migration: Code-Based → Config-Based

### Before (Scattered Throughout Codebase)

```python
# file1.py
import levrouter
levrouter.enable(frameworks=["langchain"])

# file2.py
import levrouter
levrouter.enable(frameworks=["langchain", "anthropic"])

# file3.py
import levrouter
levrouter.enable(frameworks=["langchain"])
# ... repeated 50+ times
```

### After (Single Config File)

```yaml
# levrouter.yaml
frameworks:
  - langchain
  - anthropic
  - openai
```

```python
# main.py (only place with levrouter code)
import levrouter
levrouter.load_config()
```

All 50+ files now route automatically with zero changes to their code.

---

## Troubleshooting

**Config not found?**
- Check file location: it should be in the project root (or a parent folder), or set `LEVROUTER_CONFIG`, or pass the path: `levrouter.load_config("path/to/levrouter.yaml")`
- Supported names: `levrouter.yaml`, `levrouter.yml`, `pyproject.toml`

**Framework not loaded?**
- Verify SDK is installed: `python -c "import anthropic"` or `import openai`
- Check framework name is correct: lowercase ("anthropic", not "Anthropic")

**Changes not taking effect?**
- `load_config()` must be called BEFORE creating LLM instances
- Restart your app after changing config file
- Set `LEVROUTER_LOG_LEVEL=DEBUG` to see what's loading
