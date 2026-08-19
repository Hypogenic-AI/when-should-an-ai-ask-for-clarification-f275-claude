"""
Behavioural judge running locally on the GPU.

Why local: the OpenRouter budget for this study was exhausted during data collection, after
the raw model outputs were captured but before the free-text regimes (R0, R1) could be judged.
Rather than drop Experiment 1, the judge was re-implemented on the same open-weight model that
is available locally, and its labels are validated against a stratified sample of 80 responses
hand-annotated by the author (see src/carb/judge_validation.py); the agreement statistic is
reported with the results.

The judge is scored by *constrained choice*: instead of free generation, we compare the
model's log-likelihood of each of the four action words in a fixed answer slot.  This removes
format errors entirely and makes the judgment deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", "/home/neurico/hfcache")

from carb.local_model import MODEL_ID, build_chat, load_items, load_model, set_seed  # noqa: E402
from carb.prompts import JUDGE_PROMPT  # noqa: E402

ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]


@torch.no_grad()
def judge_batch(tok, model, prompts: list[str], batch_size: int = 8) -> list[str]:
    """For each prompt, return the action whose continuation is most likely.

    The judge prompt ends by asking for a JSON object; we append the fixed prefix
    '{"behaviour": "' and compare the summed log-probability of each action's tokens.
    """
    prefix = '{"behaviour": "'
    opt_ids = [tok(a, add_special_tokens=False)["input_ids"] for a in ACTIONS]
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        best = [None] * len(chunk)
        best_lp = [-1e9] * len(chunk)
        for ai, ids in enumerate(opt_ids):
            texts = [c + prefix + ACTIONS[ai] for c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            logits = model(**enc).logits.float().log_softmax(-1)
            n = len(ids)
            for b in range(len(chunk)):
                # last n tokens are the action's tokens (right padding is off; left padding
                # means the sequence ends at the final position)
                lp = 0.0
                for k in range(n):
                    pos = enc["input_ids"].shape[1] - n + k
                    tgt = enc["input_ids"][b, pos]
                    lp += float(logits[b, pos - 1, tgt])
                lp /= n
                if lp > best_lp[b]:
                    best_lp[b] = lp
                    best[b] = ACTIONS[ai]
        out.extend(best)
        if (i // batch_size) % 10 == 0:
            print(f"    judged {i+len(chunk)}/{len(prompts)}", flush=True)
    return out


def judge_file(tok, model, src: Path, dst: Path, items: dict) -> None:
    if dst.exists() and sum(1 for _ in dst.open()) == sum(1 for _ in src.open()):
        print(f"  [cached] {dst.name}", flush=True)
        return
    rows = [json.loads(l) for l in src.read_text().splitlines()]
    live = [r for r in rows if r.get("raw")]
    prompts = [
        build_chat(tok, "You are a precise classifier. Answer only with the requested JSON.",
                   JUDGE_PROMPT.format(prompt=items[r["item_id"]]["prompt"][:1500],
                                       response=r["raw"][:2500]))
        for r in live
    ]
    print(f"  judging {src.name} ({len(prompts)} live of {len(rows)})", flush=True)
    labels = judge_batch(tok, model, prompts)
    lab = {r["item_id"]: l for r, l in zip(live, labels)}
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps({"item_id": r["item_id"], "model": r.get("model"),
                                "regime": r.get("regime"), "frame": r.get("frame"),
                                "behaviour": lab.get(r["item_id"])}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["main", "stakes", "both"], default="both")
    a = ap.parse_args()
    set_seed()
    tok, model = load_model(for_generation=True)
    print(f"Judge model: {MODEL_ID}", flush=True)

    if a.what in ("main", "both"):
        items = {i["item_id"]: i for i in load_items("test")}
        for src in sorted((ROOT / "results" / "raw").glob("test__*__R0_DIRECT.jsonl")) + \
                   sorted((ROOT / "results" / "raw").glob("test__*__R1_AFFORDANCE.jsonl")):
            dst = src.parent / f"judged__{src.name}"
            judge_file(tok, model, src, dst, items)

    if a.what in ("stakes", "both"):
        from carb.run_stakes import sample_items
        items = {i["item_id"]: i for i in sample_items()}
        for src in sorted((ROOT / "results" / "raw_stakes").glob("*__R1_AFFORDANCE__*.jsonl")):
            if src.name.startswith("judged__"):
                continue
            dst = src.parent / f"judged__{src.name}"
            judge_file(tok, model, src, dst, items)


if __name__ == "__main__":
    main()
