"""
Thin async OpenRouter client with an on-disk response cache.

Everything the experiments do is a chat completion; caching by
(model, messages, sampling params) makes reruns free and makes the whole study
reproducible without re-spending API budget.  Cache entries also record token
usage so cost can be reported honestly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
CACHE_DB = ROOT / "results" / "llm_cache.sqlite"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, pt: int, ct: int) -> None:
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.calls += 1
        m = self.by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        m["prompt_tokens"] += pt
        m["completion_tokens"] += ct
        m["calls"] += 1


USAGE = Usage()


def _init_cache() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_DB, check_same_thread=False)
    # WAL + relaxed sync: the default journal mode fsyncs on every commit, which on a mounted
    # workspace volume costs ~1s and serialises the whole async pipeline behind the cache write.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, model TEXT, ts REAL)"
    )
    con.commit()
    return con


_CON = _init_cache()
_CACHE_LOCK = asyncio.Lock()
_PENDING = 0


def flush_cache() -> None:
    """Commit any batched cache writes.  Call at the end of every run."""
    global _PENDING
    _CON.commit()
    _PENDING = 0


def _key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLM:
    """Async OpenRouter caller.  Use as an async context manager."""

    def __init__(self, concurrency: int = 16, timeout: float = 180.0, max_retries: int = 5,
                 per_model_concurrency: int = 24):
        key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_KEY not set")
        self._key = key
        # Global cap plus a per-model cap.  Without the per-model cap a slow reasoning model
        # queued first would occupy every global slot and starve the fast models entirely.
        self._sem = asyncio.Semaphore(concurrency)
        self._per_model_concurrency = per_model_concurrency
        self._model_sems: dict[str, asyncio.Semaphore] = {}
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LLM":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
        flush_cache()

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 700,
        seed: int | None = 42,
        extra: dict | None = None,
    ) -> str | None:
        """Return assistant text, or None on unrecoverable failure."""
        params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed, **(extra or {})}
        k = _key(model, messages, params)
        async with _CACHE_LOCK:
            row = _CON.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row is not None:
            USAGE.cache_hits += 1
            return json.loads(row[0])["text"]

        payload = {"model": model, "messages": messages, **{p: v for p, v in params.items() if v is not None}}
        last_err = None
        for attempt in range(self._max_retries):
            try:
                msem = self._model_sems.setdefault(model, asyncio.Semaphore(self._per_model_concurrency))
                async with msem, self._sem:
                    r = await self._client.post(
                        BASE_URL,
                        headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                if r.status_code in (429, 500, 502, 503, 504, 520, 524):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    await asyncio.sleep(min(2 ** attempt * 2, 45))
                    continue
                r.raise_for_status()
                data = r.json()
                if "choices" not in data:
                    last_err = f"no choices: {json.dumps(data)[:300]}"
                    await asyncio.sleep(min(2 ** attempt, 20))
                    continue
                text = data["choices"][0]["message"].get("content") or ""
                # Some reasoning models put content in `reasoning` when truncated; treat empty as retry-once
                if not text.strip() and attempt < 2:
                    # Reasoning models can spend the whole budget on hidden reasoning tokens and
                    # return empty content; grant more headroom once rather than failing the item.
                    last_err = "empty content"
                    payload["max_tokens"] = min(int(payload.get("max_tokens", 700) * 2), 8000)
                    continue
                u = data.get("usage") or {}
                USAGE.add(model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                async with _CACHE_LOCK:
                    _CON.execute(
                        "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                        (k, json.dumps({"text": text, "usage": u}), model, time.time()),
                    )
                    global _PENDING
                    _PENDING += 1
                    if _PENDING >= 50:      # batch commits; flush() guarantees durability at exit
                        _CON.commit()
                        _PENDING = 0
                return text
            except Exception as e:  # network / parse failures
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(min(2 ** attempt * 2, 45))
        USAGE.errors += 1
        print(f"[llm] FAILED {model}: {last_err}")
        return None


def parse_json_block(text: str | None) -> dict | None:
    """Extract the first JSON object from a model response (handles ```json fences)."""
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                t = p
                break
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except Exception:
                    return None
    return None
