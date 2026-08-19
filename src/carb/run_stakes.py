"""
Experiment 3: does the ask/refuse threshold move with announced stakes?

A 3 (announced stakes: NONE / LOW / HIGH) x 4 (gold action) within-item factorial.
Stakes are manipulated purely in the system prompt (I-CALM style, arXiv 2604.03904),
so every model sees the *identical* user request under all three frames and each item is
its own control.

Two regimes are crossed with stakes:
  R1_AFFORDANCE  - free-text behaviour (what the model actually does)
  R2_TYPED       - explicit typed routing (what it decides when the ontology is surfaced)

The NONE frame produces byte-identical prompts to the main run, so those cells are served
from the shared response cache rather than re-queried.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from carb.llm import LLM, USAGE, parse_json_block
from carb.prompts import JUDGE_PROMPT, REGIMES, STAKES_FRAMES

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw_stakes"
SEED = 42
N_PER_ACTION = 60
STAKES_MODELS = ["openai/gpt-5", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash"]
STAKES_REGIMES = ["R1_AFFORDANCE", "R2_TYPED"]
JUDGE_MODEL = "openai/gpt-4.1"


def sample_items() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "carb_v1.jsonl").read_text().splitlines()]
    test = [r for r in rows if r["split"] == "test"]
    rng = random.Random(SEED)
    by_action: dict[str, list] = {}
    for r in test:
        by_action.setdefault(r["gold_action"], []).append(r)
    out = []
    for a in sorted(by_action):
        pool = sorted(by_action[a], key=lambda r: r["item_id"])
        rng.shuffle(pool)
        out.extend(pool[:N_PER_ACTION])
    return out


async def main_async(concurrency: int) -> None:
    items = sample_items()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Stakes factorial: {len(items)} items x {len(STAKES_MODELS)} models x "
          f"{len(STAKES_REGIMES)} regimes x {len(STAKES_FRAMES)} frames", flush=True)

    async def cell(llm: LLM, model: str, regime: str, frame: str) -> None:
        out = RAW / f"{model.replace('/','_')}__{regime}__{frame}.jsonl"
        if out.exists() and sum(1 for _ in out.open()) == len(items):
            print(f"  [cached] {model} {regime} {frame}", flush=True)
            return
        tmpl = REGIMES[regime]
        sys_prompt = tmpl["system"] + STAKES_FRAMES[frame]

        async def one(it):
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": tmpl["user"].format(prompt=it["prompt"])},
            ]
            txt = await llm.chat(model, msgs, temperature=0.0,
                                 max_tokens=800 if regime == "R1_AFFORDANCE" else 500)
            return {"item_id": it["item_id"], "model": model, "regime": regime,
                    "frame": frame, "raw": txt}

        res = await asyncio.gather(*[one(it) for it in items])
        with out.open("w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        print(f"  done {model} {regime} {frame} fails={sum(1 for r in res if not r['raw'])}", flush=True)

    async with LLM(concurrency=concurrency) as llm:
        await asyncio.gather(*[
            cell(llm, m, rg, fr)
            for m in STAKES_MODELS for rg in STAKES_REGIMES for fr in STAKES_FRAMES
        ])

    # judge the free-text regime
    lookup = {it["item_id"]: it for it in items}
    print("Judging free-text stakes cells...", flush=True)

    async def judge_cell(llm: LLM, model: str, frame: str) -> None:
        src = RAW / f"{model.replace('/','_')}__R1_AFFORDANCE__{frame}.jsonl"
        dst = RAW / f"judged__{model.replace('/','_')}__R1_AFFORDANCE__{frame}.jsonl"
        rows = [json.loads(l) for l in src.read_text().splitlines()]
        if dst.exists() and sum(1 for _ in dst.open()) == len(rows):
            print(f"  [cached] judge {model} {frame}", flush=True)
            return

        async def one(r):
            if not r["raw"]:
                return {**r, "behaviour": None}
            p = JUDGE_PROMPT.format(prompt=lookup[r["item_id"]]["prompt"][:2000], response=r["raw"][:3000])
            t = await llm.chat(JUDGE_MODEL, [{"role": "user", "content": p}], temperature=0.0, max_tokens=200)
            j = parse_json_block(t) or {}
            b = str(j.get("behaviour", "")).strip().upper()
            return {**r, "behaviour": b if b in {"ACT", "ASK", "REFUSE", "DEFER"} else None}

        res = await asyncio.gather(*[one(r) for r in rows])
        with dst.open("w") as f:
            for r in res:
                f.write(json.dumps({k: v for k, v in r.items() if k != "raw"}) + "\n")
        print(f"  judged {model} {frame}", flush=True)

    async with LLM(concurrency=concurrency) as llm:
        await asyncio.gather(*[judge_cell(llm, m, fr) for m in STAKES_MODELS for fr in STAKES_FRAMES])

    print(f"calls={USAGE.calls} cache_hits={USAGE.cache_hits} errors={USAGE.errors}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=96)
    a = ap.parse_args()
    asyncio.run(main_async(a.concurrency))
