# Llama Index Integration with levrouter

**Recommended Approach: Indirect Routing via Underlying SDKs**

levrouter routes Llama Index calls by routing the underlying Anthropic SDK and OpenAI SDK that Llama Index uses internally. No Llama Index-specific integration needed.

## How It Works

Llama Index wraps other LLM SDKs (Anthropic, OpenAI, Gemini, etc.) when you create LlamaIndex LLM instances. levrouter intercepts at the SDK level, so:

```python
# Your code
from llama_index.llms.anthropic import Anthropic as LIAnthropic
llm = LIAnthropic(model="claude-opus-5-5")
response = llm.complete("What's the capital of France?")

# What happens:
# 1. LIAnthropic wraps anthropic.Anthropic SDK
# 2. Your question routes through levrouter
# 3. Cheap model (claude-haiku-4-5) handles simple task
# 4. Same answer, lower cost
```

## Setup (3 Steps)

### Step 1: Install

```bash
pip install levrouter
# OR for Anthropic+OpenAI
pip install "levrouter[anthropic,openai]"
```

### Step 2: Enable in Your App

```python
import levrouter

# Route both LangChain AND Llama Index
levrouter.enable(frameworks=["langchain", "anthropic", "openai"])
```

### Step 3: Use Llama Index Normally

```python
from llama_index.llms.anthropic import Anthropic as LIAnthropic

llm = LIAnthropic(model="claude-opus-5-5")

# Automatic routing ✨
response = llm.complete("What's the capital of France?")
# → Routed to claude-haiku-4-5 (simple task)

response = llm.complete("Design a database schema for...")  
# → Routed to claude-opus-5-5 (complex task)
```

## Supported Llama Index LLM Classes

Works with any Llama Index LLM that uses Anthropic or OpenAI:

| Llama Index Class | Underlying SDK | Supported |
|---|---|---|
| `llama_index.llms.anthropic.Anthropic` | anthropic SDK | ✅ |
| `llama_index.llms.openai.OpenAI` | openai SDK | ✅ |
| LangChain wrappers | LangChain | ✅ |
| Others (Gemini, Groq, DeepSeek) | Respective SDKs | Partial |

## Example: Full Llama Index Setup

```python
import levrouter
from llama_index.llms.anthropic import Anthropic as LIAnthropic
from llama_index.core import Document, VectorStoreIndex, Settings

# Enable routing
levrouter.enable(frameworks=["anthropic"])

# Configure LLM (routing applies automatically)
Settings.llm = LIAnthropic(model="claude-opus-5-5")

# Use Llama Index normally
docs = [
    Document(text="The capital of France is Paris."),
    Document(text="The capital of Spain is Madrid."),
]

index = VectorStoreIndex.from_documents(docs)

# Queries route based on complexity
response = index.as_query_engine().query("What's the capital of France?")
# → Simple query, routed to haiku

response = index.as_query_engine().query("Compare the geography and culture of France and Spain")
# → Complex query, routed to Opus
```

## Why This Approach?

### Advantages
✅ No Llama Index-specific code to maintain  
✅ Zero extra lines needed  
✅ Works for future Llama Index versions  
✅ Covers all Llama Index LLMs using supported SDKs  
✅ Users get routing transparently  

### Trade-offs
⚠️ Indirect (routing happens at SDK level, not Llama Index level)  
⚠️ Doesn't see Llama Index context (index type, retrieval settings, etc.)  
⚠️ No Llama Index-specific logging  

## Troubleshooting

**Llama Index calls not being routed?**
- Verify `enable(frameworks=["anthropic", "openai"])` is called before creating LLM
- Check that you're using the native SDK-based Llama Index LLMs, not custom wrappers
- Enable debug logging: `LEVROUTER_LOG_LEVEL=DEBUG`

---

This approach scales: any SDK that levrouter supports automatically works with its Llama Index wrapper.
