"""
Export a stratified sample of free-text responses for hand annotation, and score the local
judge against those annotations.

Stage 1 (`--export`) writes results/judge_validation_sample.json with the sample and empty
`human` fields.  The author fills them in.  Stage 2 (`--score`) computes raw agreement and
Cohen's kappa between the human labels and the judge labels and writes
results/judge_validation.json, which the main analysis reports alongside every judged metric.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "judge_validation_sample.json"
ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]
SEED = 17
N = 80


def export() -> None:
    from carb.derive import load_items

    items = load_items("test")
    pool = []
    for f in sorted(RAW.glob("judged__test__*.jsonl")):
        src = RAW / f.name.replace("judged__", "")
        raws = {json.loads(l)["item_id"]: json.loads(l)["raw"] for l in src.read_text().splitlines()}
        for l in f.read_text().splitlines():
            d = json.loads(l)
            if d["behaviour"] and raws.get(d["item_id"]):
                pool.append({"file": f.name, "item_id": d["item_id"],
                             "prompt": items[d["item_id"]]["prompt"],
                             "response": raws[d["item_id"]][:1200],
                             "judge": d["behaviour"], "human": ""})
    rng = random.Random(SEED)
    # stratify by the judge's label so every class is represented
    by_lab: dict[str, list] = {}
    for r in pool:
        by_lab.setdefault(r["judge"], []).append(r)
    sample = []
    per = max(1, N // max(len(by_lab), 1))
    for lab in sorted(by_lab):
        rows = by_lab[lab][:]
        rng.shuffle(rows)
        sample.extend(rows[:per])
    rng.shuffle(sample)
    OUT.write_text(json.dumps(sample, indent=2))
    print(f"Wrote {len(sample)} rows to {OUT} (judge label distribution: "
          f"{dict(Counter(r['judge'] for r in sample))})")


def score() -> None:
    rows = [r for r in json.loads(OUT.read_text()) if r.get("human") in ACTIONS]
    if not rows:
        print("No human labels filled in yet.")
        return
    x = [r["human"] for r in rows]
    y = [r["judge"] for r in rows]
    po = float(np.mean([a == b for a, b in zip(x, y)]))
    cx, cy = Counter(x), Counter(y)
    pe = sum((cx[k] / len(rows)) * (cy[k] / len(rows)) for k in ACTIONS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    cm = {f"human={a}|judge={b}": sum(1 for u, v in zip(x, y) if u == a and v == b)
          for a in ACTIONS for b in ACTIONS}
    out = {"n": len(rows), "raw_agreement": po, "cohens_kappa": float(kappa),
           "confusion": {k: v for k, v in cm.items() if v}, "annotator": "study author"}
    (ROOT / "results" / "judge_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.export:
        export()
    if a.score:
        score()
