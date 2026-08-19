"""
Main experiment runner: model x elicitation-regime x item grid.

Stage 1 collects raw model outputs for every (model, regime, item) cell.
Stage 2 runs the behavioural judge over the free-text regimes (R0, R1).
Both stages are cached, so the script is safe to re-run and resume.

Usage:
    python src/carb/run_eval.py --split test        # main grid
    python src/carb/run_eval.py --split dev         # threshold-tuning split
    python src/carb/run_eval.py --split transfer    # CLAMBER out-of-source check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np

from carb.llm import LLM, USAGE, parse_json_block
from carb.prompts import FREE_TEXT_REGIMES, JUDGE_PROMPT, REGIMES

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
SEED = 42

MODELS = [
    "openai/gpt-5",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]
JUDGE_MODEL = "openai/gpt-4.1"
# Second, architecturally different judge used to measure judge reliability.
JUDGE_MODEL_ALT = "google/gemini-2.5-flash"

# Token budgets are generous because reasoning models bill hidden reasoning tokens against
# max_tokens; too tight a budget returns empty content and silently biases the sample.
MAX_TOKENS = {"R0_DIRECT": 1600, "R1_AFFORDANCE": 1600}
DEFAULT_MAX_TOKENS = 1200


def load_items(split: str) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "carb_v1.jsonl").read_text().splitlines()]
    return [r for r in rows if r["split"] == split]


async def run_cell(llm: LLM, model: str, regime: str, items: list[dict], out_path: Path) -> None:
    tmpl = REGIMES[regime]
    max_tok = MAX_TOKENS.get(regime, DEFAULT_MAX_TOKENS)

    async def one(it: dict) -> dict:
        msgs = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(prompt=it["prompt"])},
        ]
        text = await llm.chat(model, msgs, temperature=0.0, max_tokens=max_tok)
        return {"item_id": it["item_id"], "model": model, "regime": regime, "raw": text}

    results = await asyncio.gather(*[one(it) for it in items])
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    n_fail = sum(1 for r in results if not r["raw"])
    print(f"  {model:38s} {regime:15s} n={len(results):4d} failures={n_fail}", flush=True)


async def stage1(split: str, concurrency: int) -> None:
    items = load_items(split)
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"\n=== STAGE 1: raw outputs | split={split} n={len(items)} ===", flush=True)
    # All (model, regime) cells run concurrently: providers differ in per-call latency by
    # more than an order of magnitude (a reasoning model vs. a small instruct model), so
    # running cells sequentially would leave most of the throughput budget idle.
    async with LLM(concurrency=concurrency) as llm:
        jobs = []
        for model in MODELS:
            for regime in REGIMES:
                out = RAW / f"{split}__{model.replace('/','_')}__{regime}.jsonl"
                if out.exists() and sum(1 for _ in out.open()) == len(items):
                    print(f"  {model:38s} {regime:15s} [cached]", flush=True)
                    continue
                jobs.append(run_cell(llm, model, regime, items, out))
        await asyncio.gather(*jobs)


async def stage2(split: str, concurrency: int, judge_model: str, tag: str) -> None:
    """Judge the free-text regimes into the 4-way behaviour space."""
    items = {it["item_id"]: it for it in load_items(split)}
    print(f"\n=== STAGE 2: behavioural judging ({judge_model}) | split={split} ===", flush=True)
    async def judge_cell(model: str, regime: str, llm: LLM) -> None:
        src = RAW / f"{split}__{model.replace('/','_')}__{regime}.jsonl"
        dst = RAW / f"judged{tag}__{split}__{model.replace('/','_')}__{regime}.jsonl"
        rows = [json.loads(l) for l in src.read_text().splitlines()]

        async def one(r: dict) -> dict:
            if not r["raw"]:
                return {**r, "behaviour": None}
            p = JUDGE_PROMPT.format(prompt=items[r["item_id"]]["prompt"][:2000], response=r["raw"][:3000])
            txt = await llm.chat(judge_model, [{"role": "user", "content": p}], temperature=0.0, max_tokens=200)
            j = parse_json_block(txt) or {}
            b = str(j.get("behaviour", "")).strip().upper()
            return {**r, "behaviour": b if b in {"ACT", "ASK", "REFUSE", "DEFER"} else None}

        out = await asyncio.gather(*[one(r) for r in rows])
        with dst.open("w") as f:
            for r in out:
                f.write(json.dumps({k: v for k, v in r.items() if k != "raw"}) + "\n")
        bad = sum(1 for r in out if r["behaviour"] is None)
        print(f"  {model:38s} {regime:15s} judged={len(out)} unparsed={bad}", flush=True)

    async with LLM(concurrency=concurrency) as llm:
        jobs = []
        for model in MODELS:
            for regime in sorted(FREE_TEXT_REGIMES):
                src = RAW / f"{split}__{model.replace('/','_')}__{regime}.jsonl"
                if not src.exists():
                    continue
                dst = RAW / f"judged{tag}__{split}__{model.replace('/','_')}__{regime}.jsonl"
                n_src = sum(1 for _ in src.open())
                if dst.exists() and sum(1 for _ in dst.open()) == n_src:
                    print(f"  {model:38s} {regime:15s} [cached]", flush=True)
                    continue
                jobs.append(judge_cell(model, regime, llm))
        await asyncio.gather(*jobs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--alt-judge", action="store_true", help="also run the second judge model")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    asyncio.run(stage1(args.split, args.concurrency))
    asyncio.run(stage2(args.split, args.concurrency, JUDGE_MODEL, ""))
    if args.alt_judge:
        asyncio.run(stage2(args.split, args.concurrency, JUDGE_MODEL_ALT, "_alt"))

    print(f"\nUsage: calls={USAGE.calls} cache_hits={USAGE.cache_hits} errors={USAGE.errors}")
    print(f"  prompt_tokens={USAGE.prompt_tokens:,} completion_tokens={USAGE.completion_tokens:,}")
    (ROOT / "results" / f"usage_{args.split}.json").write_text(
        json.dumps({"by_model": USAGE.by_model, "calls": USAGE.calls, "cache_hits": USAGE.cache_hits}, indent=2)
    )


if __name__ == "__main__":
    main()
