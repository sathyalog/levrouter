# Conversations: one model per thread, and moving up on re-asked questions

## Why keep a conversation on one model

Providers keep a **prompt cache per model**. When an app marks a long system prompt, its tools or the chat history as cacheable, later turns re-read that part for about 10% of the normal input price; writing it the first time costs about 125%. A different model has no cache, so switching models mid-conversation pays full price again.

Example with a 40,000-token cached prefix (Sonnet 5 $2/M input, Haiku 4.5 $1/M):

| Turn | Same model every turn | Routed turn by turn |
|---|---|---|
| 1 | Sonnet writes the cache: **$0.100** | Sonnet: **$0.100** |
| 2 ("ok, list them") | Sonnet cache hit: **$0.008** | Haiku, no cache: **$0.050** |

The "cheaper" model cost about 6× more for that turn's input. So with `sticky_threads: true` (the default), a conversation keeps its model. It may move **up** a tier when a harder question arrives, never down. The log reason shows `sticky thread: kept complex`.

## How levrouter knows the conversation

| Source | How |
|---|---|
| LangGraph | `config={"configurable": {"thread_id": ...}}` is used automatically |
| Plain LangChain | `llm.invoke(question, config={"metadata": {"levrouter_thread": conversation_id}})` |
| Anthropic / OpenAI SDK | No thread id: every call is routed on its own (re-asked questions still move up, see below) |

A conversation keeps its model until it has been quiet for `thread_ttl_s` seconds (default 300, Anthropic's default cache lifetime; use 3600 with the 1-hour cache). Up to 10,000 conversations are remembered in memory per process.

## Moving up when the user re-asks

A user asking the same thing again usually means the answer wasn't good enough. levrouter compares the latest user message with the earlier user messages **in the same request**. Chat apps send the history with each call, so this needs no thread id and no stored state.

- **A repeat** is an earlier message sharing at least half of the meaningful words (3+ letters, common words ignored). The latest message must have at least 3 such words, so "thanks" or "ok" never count.
- **Retry phrases** also count once: "try again", "that's wrong", "not what I asked", "still wrong", "doesn't work", "wrong answer", "you didn't answer".
- **When it moves up:** with `escalate_after_repeats: 2` (the default), the 3rd time a question is asked moves up one tier, and each further repeat moves up again, up to the top tier. `0` turns this off. In a thread, the higher tier is then kept.

| Turn | User | Model |
|---|---|---|
| 1 | "How do I configure retries in the payment client?" | Haiku |
| 2 | "How do I set up retries for the payment client?" | Haiku |
| 3 | "That's wrong. How do I configure payment client retries?" | **Sonnet**, `asked 3 times: moved up to standard` |
| 4 | "thanks" | Sonnet (sticky) |

**Limit:** detection compares words, not meaning. A rephrasing with entirely different words ("why does my checkout keep failing?") isn't recognised; use the manual options below.

## Switching by hand

| Need | How |
|---|---|
| A specific tier for one call (the conversation then stays on it) | `config={"metadata": {"levrouter_tier": "simple"}}` (or `standard` / `complex`) |
| The model written in your code, for one call | `config={"metadata": {"levrouter_disable": True}}` |
| Start the conversation over, e.g. after trimming or summarising history | `levrouter.reset_thread(conversation_id)`, or use a new thread id |
| Turn the behaviours off | `sticky_threads: false`, `escalate_after_repeats: 0` |
