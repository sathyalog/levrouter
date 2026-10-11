# Troubleshooting: API Authentication Error (401)

If you're seeing `anthropic.AuthenticationError: Error code: 401 - 'API key is invalid.'` when using levrouter with LangChain, this guide will help you diagnose and fix the issue.

---

## Quick Diagnosis

Answer these 3 questions to identify the root cause:

1. **Does your code have `levrouter.load_config()` at the top?**
   - If NO → Jump to [Root Cause #1](#root-cause-1-missing-levrouter-initialization)
   - If YES → Continue to question 2

2. **Does your `levrouter.yaml` use `openrouter/...` format?**
   - If YES → Check if you have `OPENROUTER_API_KEY` set → Jump to [Root Cause #2](#root-cause-2-api-key-type-mismatch)
   - If NO → Continue to question 3

3. **Did you generate `levrouter.yaml` BEFORE September 30, 2026?**
   - If YES → Jump to [Root Cause #3](#root-cause-3-out-of-sync-configuration)
   - If NO → See [Advanced Debugging](#advanced-debugging)

---

## Root Cause #1: Missing levrouter Initialization

**Symptom:** Code runs but `levrouter.yaml` is ignored, direct API calls fail with 401

**Why This Happens:**
- When `levrouter.load_config()` is NOT called, levrouter doesn't intercept LLM calls
- Your code makes direct API calls to Anthropic/OpenAI/etc without routing
- The direct API uses whatever credentials are in the environment

**Quick Fix:**

Add this to the **TOP** of your Python file (before any LangChain imports):

\`\`\`python
import levrouter
levrouter.load_config()

# Now safe to import LangChain
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
\`\`\`

**Example - Before:**
\`\`\`python
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
# ❌ levrouter.load_config() is missing!

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
\`\`\`

**Example - After:**
\`\`\`python
from dotenv import load_dotenv

load_dotenv()

# ✅ Initialize levrouter FIRST
import levrouter
levrouter.load_config()

# NOW import LangChain
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
\`\`\`

---

## Root Cause #2: API Key Type Mismatch

**Symptom:** 401 "API key is invalid" error even though the key looks correct

**Why This Happens:**
- Your `levrouter.yaml` is configured for one provider (e.g., OpenRouter)
- But your `.env` has an API key for a different provider (e.g., Anthropic)
- The wrong API key is sent to the provider

**Examples:**

| levrouter.yaml Config | .env Should Have | ❌ Wrong | ✅ Correct |
|---|---|---|---|
| \`openrouter/anthropic/claude-...\` | OpenRouter | \`ANTHROPIC_API_KEY=sk-ant-...\` | \`OPENROUTER_API_KEY=sk-or-v1-...\` |
| \`anthropic/claude-...\` | Anthropic | \`OPENROUTER_API_KEY=sk-or-v1-...\` | \`ANTHROPIC_API_KEY=sk-ant-...\` |
| \`openai/gpt-...\` | OpenAI | \`ANTHROPIC_API_KEY=sk-ant-...\` | \`OPENAI_API_KEY=sk-...\` |

**How to Fix:**

### Option A: Use Direct Anthropic

If you have an **Anthropic API key** (sk-ant-...):

**Step 1:** Delete old config
\`\`\`bash
rm levrouter.yaml
\`\`\`

**Step 2:** Regenerate with fixed setup
\`\`\`bash
./.venv/bin/levrouter setup
\`\`\`

**Step 3:** When prompted, select:
- Backend: Choose one (1=lev-local, 2=lev-http, 3=heuristic)
- Provider: \`1\` (anthropic)
- Customize tiers: No (or yes if you want custom models)

**Step 4:** Verify your \`.env\` has:
\`\`\`bash
ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE
\`\`\`

**Step 5:** Result - your \`levrouter.yaml\` should look like:
\`\`\`yaml
enabled: true
router:
  backend: heuristic
  timeout_s: 2.0
models:
  simple:
    providers:
    - anthropic/claude-3-5-haiku
  standard:
    providers:
    - anthropic/claude-3-5-sonnet
  complex:
    providers:
    - anthropic/claude-3-5-opus
\`\`\`

### Option B: Use OpenRouter

If you have an **OpenRouter API key** (sk-or-v1-...):

**Step 1:** Delete old config
\`\`\`bash
rm levrouter.yaml
\`\`\`

**Step 2:** Regenerate with fixed setup
\`\`\`bash
./.venv/bin/levrouter setup
\`\`\`

**Step 3:** When prompted, select:
- Backend: Choose one
- Provider: \`9\` (openrouter)
- Which provider's models?: \`1\` (anthropic) or your choice
- Customize tiers: No (or yes if you want custom models)

**Step 4:** Verify your \`.env\` has:
\`\`\`bash
OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE
\`\`\`

**Step 5:** Result - your \`levrouter.yaml\` should look like:
\`\`\`yaml
enabled: true
router:
  backend: heuristic
  timeout_s: 2.0
models:
  simple:
    providers:
    - openrouter/anthropic/claude-3-5-haiku
  standard:
    providers:
    - openrouter/anthropic/claude-3-5-sonnet
  complex:
    providers:
    - openrouter/anthropic/claude-3-5-opus
\`\`\`

---

## Root Cause #3: Out-of-Sync Configuration

**Symptom:** 401 error even after fixing code and checking API key

**Why This Happens:**
- You ran \`levrouter setup\` before September 30, 2026
- The old setup wizard had a bug: it hardcoded \`openrouter/\` prefix for ALL providers
- Your config has wrong format even though you selected Anthropic
- The bug is now FIXED, but old configs still exist

**Example of the bug:**
\`\`\`yaml
# OLD (before fix) - even if you selected "anthropic"!
models:
  simple:
    providers:
    - openrouter/anthropic/claude-3-5-haiku  # ❌ Wrong!
\`\`\`

**How to Fix:**

**Step 1:** Delete old config
\`\`\`bash
rm levrouter.yaml
\`\`\`

**Step 2:** Re-run setup with the FIXED version
\`\`\`bash
# Make sure you have the latest levrouter
uv sync --upgrade-package levrouter

# Or update from repo
git pull  # if you're running from source
\`\`\`

**Step 3:** Generate new config
\`\`\`bash
./.venv/bin/levrouter setup
\`\`\`

**Step 4:** The new config will use the CORRECT format:
\`\`\`yaml
# ✅ NEW (after fix)
models:
  simple:
    providers:
    - anthropic/claude-3-5-haiku  # Correct!
\`\`\`

---

## Verification Steps

**Once you've applied a fix, verify it works:**

### 1. Check levrouter.yaml Format

Look at your config and verify it matches your chosen provider:

\`\`\`bash
cat levrouter.yaml
\`\`\`

**For Anthropic:**
\`\`\`yaml
models:
  simple:
    providers:
    - anthropic/claude-*
\`\`\`

**For OpenRouter:**
\`\`\`yaml
models:
  simple:
    providers:
    - openrouter/anthropic/claude-*
\`\`\`

### 2. Verify Code Initialization

Make sure your Python file has this at the top:

\`\`\`python
import levrouter
levrouter.load_config()
\`\`\`

### 3. Test with a Simple Query

Add this debug test to your code:

\`\`\`python
import levrouter
levrouter.load_config()

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-haiku")

# Simple test
response = llm.invoke("Say 'hello' in one word")
print(response.content)
\`\`\`

**Expected output:**
\`\`\`
Hello
\`\`\`

If this works, levrouter is routing correctly!

---

## Prevention Tips

**Avoid this error in the future:**

### 1. Always Initialize First
\`\`\`python
# ✅ CORRECT
import levrouter
levrouter.load_config()

from langchain_anthropic import ChatAnthropic
\`\`\`

### 2. Match API Keys to Providers

Before running your code, verify:
- If \`levrouter.yaml\` says \`openrouter/...\` → Need \`OPENROUTER_API_KEY\`
- If \`levrouter.yaml\` says \`anthropic/...\` → Need \`ANTHROPIC_API_KEY\`
- If \`levrouter.yaml\` says \`openai/...\` → Need \`OPENAI_API_KEY\`

### 3. When Switching Providers

If you change providers:
\`\`\`bash
# 1. Delete old config
rm levrouter.yaml

# 2. Update .env with new key
# 3. Regenerate config
./.venv/bin/levrouter setup
\`\`\`

---

## Summary

| Error | Cause | Fix |
|-------|-------|-----|
| 401 with levrouter.yaml present | Missing \`levrouter.load_config()\` | Add to top of code |
| 401 with wrong API key type | Config format doesn't match key | Regenerate yaml with \`levrouter setup\` |
| 401 with old setup wizard config | Setup bug generated wrong format | Delete yaml + regenerate with latest setup |

**The universal fix:**
1. \`import levrouter; levrouter.load_config()\` at top of code
2. Delete \`levrouter.yaml\`
3. Run \`levrouter setup\` to regenerate
4. Verify API key in \`.env\` matches provider type
5. Test with simple query

---

**Need help?** Check the main README.md or configuration guide at docs/configuration.md
