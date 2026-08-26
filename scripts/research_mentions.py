"""Research helper: query Grok with X search for memecoin launch evidence.

Reads the key from the environment. Never hardcode a credential here:

    export MEMESCANNER_TAVILY_API_KEY="xai-..."
    PYTHONPATH=. python3 scripts/research_mentions.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

XAI_URL = "https://api.x.ai/v1/responses"
XAI_MODEL = "grok-4.6"


def _api_key() -> str:
    key = os.environ.get("MEMESCANNER_TAVILY_API_KEY", "").strip()
    if not key:
        print(
            "MEMESCANNER_TAVILY_API_KEY is not set.\n"
            "Export an X.ai key (xai-...) before running this script.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not key.startswith("xai-"):
        print(
            "This script requires an X.ai key (xai-... prefix); "
            "the configured key is not one.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return key


async def ask_grok(question: str, api_key: str) -> str:
    """Ask Grok a question with the X search tool enabled."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            XAI_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": XAI_MODEL,
                "tools": [
                    {"type": "x_search", "x_search": {}},
                    {"type": "web_search", "web_search": {}},
                ],
                "input": question,
            },
        )
        response.raise_for_status()
        data = response.json()

    texts = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("text"):
                    texts.append(block["text"])
        elif item.get("text"):
            texts.append(item["text"])
    return "\n".join(texts) if texts else json.dumps(data.get("output", ""), indent=2)


QUESTIONS = [
    (
        "Find the earliest tweets/X posts about pump.fun Solana tokens that later "
        "did 50x or more. For each example give: the token ticker, how many X "
        "accounts posted about it in the first hour, whether the posts included "
        "the full contract address, and how many followers the earliest callers "
        "had. Concrete examples with numbers, not general advice."
    ),
    (
        "Compare early X mentions between pump.fun tokens that SUCCEEDED (50x+) "
        "versus those that RUGGED or died within 24h. Do rugged tokens have more "
        "or fewer early X posts than winners? Are bot-farm posts distinguishable "
        "from organic early calls? What engagement metrics separate them? Give "
        "measurable thresholds."
    ),
]


async def main() -> None:
    api_key = _api_key()
    for index, question in enumerate(QUESTIONS, 1):
        print("=" * 60)
        print(f"RESEARCH QUESTION {index}")
        print("=" * 60)
        print()
        try:
            print(await ask_grok(question, api_key))
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
