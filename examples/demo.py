"""Show which model each prompt is routed to.

    python examples/demo.py --backend heuristic            # decisions only, no API calls
    python examples/demo.py --backend lev-local            # decisions from lev (downloads weights on first run)
    python examples/demo.py --backend lev-local --live     # also call the API (needs ANTHROPIC_API_KEY / OPENAI_API_KEY)
"""

from __future__ import annotations

import argparse
import logging

import levrouter
from levrouter import Router, RouterConfig
from levrouter.features import extract

PROMPTS = [
    "What's the capital of Australia?",
    "Translate 'see you tomorrow' into French.",
    "Write a short, friendly email telling my team the office is closed on Friday.",
    "Summarize the key differences between REST and GraphQL for a junior developer.",
    "Design a multi-region rate limiter for a public API. Compare token bucket and sliding window, "
    "handle clock skew, and prove the global limit is never exceeded.",
    "Debug this: my asyncio worker pool deadlocks under load when a task raises.\n"
    "```python\nasync def worker(q):\n    while True:\n        item = await q.get()\n        await handle(item)\n```",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="heuristic", choices=["heuristic", "lev-local", "lev-http"])
    parser.add_argument("--provider", default="anthropic", choices=sorted(levrouter.PROVIDERS))
    parser.add_argument("--live", action="store_true", help="make real API calls")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per lev decision")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    config = RouterConfig(backend=args.backend, timeout_s=args.timeout, block_on_load=True, log_path="levrouter.jsonl")
    original = levrouter.PROVIDERS[args.provider].tiers["complex"]

    if args.live and args.provider not in ("anthropic", "openai"):
        parser.error("--live is wired for anthropic and openai; use dry-run for other providers")
    if not args.live:
        router = Router(config)
        for prompt in PROMPTS:
            from langchain_core.messages import HumanMessage

            d = router.decide(args.provider, original, extract([HumanMessage(prompt)]))
            p = d.signals.tier if d.signals else {}
            print(f"{d.model:<18} {d.tier:<9} [{d.signals.source if d.signals else '-'}] "
                  f"simple={p.get('simple', 0):.2f} std={p.get('standard', 0):.2f} cx={p.get('complex', 0):.2f}"
                  f"  | {prompt.splitlines()[0][:70]}")
        return

    levrouter.enable(config)
    if args.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=original, max_tokens=1024)
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=original, max_tokens=1024)

    for prompt in PROMPTS:
        msg = llm.invoke(prompt)
        info = msg.response_metadata["levrouter"]
        print(f"\n=== {info['model']} ({info['tier']}) <- {prompt.splitlines()[0][:70]}")
        print(str(msg.content)[:300])
    print("\n", levrouter.stats())


if __name__ == "__main__":
    main()
