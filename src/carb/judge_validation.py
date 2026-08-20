"""
Export a stratified sample of free-text responses for hand annotation, and score the local
judge against those annotations.

Stage 1 (`--export`) writes results/judge_validation_sample.json with the sample and empty
`human` fields, plus results/judge_validation_blind.json containing only the request and the
response.  The `human` column in this study was filled in from the *blind* file by the
orchestrating research agent (Claude Opus 5), not by an independent human annotator; the
agreement statistic should be read as "a third model, shown only the text, agrees with the
judge", which is weaker evidence than human validation and is reported as such.  Stage 2 (`--score`) computes raw agreement and
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
BLIND = ROOT / "results" / "judge_validation_blind.json"
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
    # Blind copy: the annotator sees only the request and the response, never the judge's
    # label, so the annotation cannot anchor on it.  Keyed by position in OUT.
    BLIND.write_text(json.dumps(
        [{"i": k, "prompt": r["prompt"], "response": r["response"][:900]}
         for k, r in enumerate(sample)], indent=2))
    print(f"Wrote {len(sample)} rows to {OUT} (judge label distribution: "
          f"{dict(Counter(r['judge'] for r in sample))})")


def ingest(path: str) -> None:
    """Merge blind annotations (a JSON list of {"i": int, "label": str}) into OUT."""
    ann = {int(d["i"]): d["label"].strip().upper() for d in json.loads(Path(path).read_text())}
    rows = json.loads(OUT.read_text())
    n = 0
    for k, r in enumerate(rows):
        if k in ann and ann[k] in ACTIONS:
            r["human"] = ann[k]
            n += 1
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"ingested {n} annotations into {OUT}")


def cross_judge() -> None:
    """Agreement between the two independent local judges over every judged item."""
    prim, alt = {}, {}
    for f in sorted(RAW.glob("judged__test__*.jsonl")):
        for l in f.read_text().splitlines():
            d = json.loads(l)
            prim[(f.name, d["item_id"])] = d["behaviour"]
    for f in sorted(RAW.glob("judged_alt4b__test__*.jsonl")):
        key = f.name.replace("judged_alt4b__", "judged__")
        for l in f.read_text().splitlines():
            d = json.loads(l)
            alt[(key, d["item_id"])] = d["behaviour"]
    both = [(prim[k], alt[k]) for k in prim if k in alt and prim[k] and alt[k]]
    if not both:
        print("no overlapping judged items yet")
        return
    x = [a for a, _ in both]
    y = [b for _, b in both]
    po = float(np.mean([a == b for a, b in both]))
    cx, cy = Counter(x), Counter(y)
    pe = sum((cx[k] / len(both)) * (cy[k] / len(both)) for k in ACTIONS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    out = {"n": len(both), "raw_agreement": po, "cohens_kappa": float(kappa),
           "primary_judge": "Qwen/Qwen3-14B", "alt_judge": "Qwen/Qwen3-4B",
           "primary_label_dist": dict(cx), "alt_label_dist": dict(cy),
           "confusion": {f"primary={a}|alt={b}": sum(1 for u, v in both if u == a and v == b)
                         for a in ACTIONS for b in ACTIONS}}
    (ROOT / "results" / "judge_cross_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=2))


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
           "confusion": {k: v for k, v in cm.items() if v}, "annotator": "the orchestrating research agent (Claude Opus 5), annotating blind from "
                        "request+response only, with no access to the judge's label, the gold "
                        "label, or the item's source; NOT an independent human annotator"}
    (ROOT / "results" / "judge_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cross-judge", action="store_true")
    ap.add_argument("--ingest", default=None, help="path to a blind-annotation JSON file")
    a = ap.parse_args()
    if a.export:
        export()
    if a.ingest:
        ingest(a.ingest)
    if a.cross_judge:
        cross_judge()
    if a.score:
        score()
