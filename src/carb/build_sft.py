"""
Build the supervised training set for the local routing model (Experiment 4).

Strict separation from evaluation:
  * ASK / REFUSE / DEFER examples come from CoCoNot's **train** split; CARB's CoCoNot items
    all come from the **test** split, so no item is shared.
  * ACT examples come from (a) the 239 CoCoNot contrast items *not* sampled into CARB and
    (b) AmbigQA questions annotated by its own annotators as having a single answer, i.e.
    unambiguous, answerable, safe requests.  AmbigQA appears nowhere in CARB.
  * CLAMBER and IN3 are never trained on, so both are genuine out-of-source transfer.

The ACT class is the scarce one (CoCoNot is by construction a non-compliance dataset), which
is why a second ACT source is needed; the trade-off is that a model could learn a
"short factoid question -> ACT" shortcut. The CARB ACT items are contrast-set and IN3-style,
not factoid, so such a shortcut would not be rewarded at evaluation time.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from carb.build_benchmark import _map_action

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "carb"
SEED = 42
N_PER_CLASS = 700


def main() -> None:
    rng = random.Random(SEED)
    used = {json.loads(l)["item_id"] for l in (OUT / "carb_v1.jsonl").read_text().splitlines()}

    pool: dict[str, list[str]] = {"ACT": [], "ASK": [], "REFUSE": [], "DEFER": []}

    tr = load_from_disk(str(ROOT / "datasets" / "coconot_original"))["train"]
    for r in tr:
        a = _map_action("coconot", r["category"], r["subcategory"])
        if a and f"coconot:{r['id']}" not in used:
            pool[a].append(r["prompt"].strip())

    ctr = load_from_disk(str(ROOT / "datasets" / "coconot_contrast"))["test"]
    n_contrast = 0
    for r in ctr:
        if f"coconot_contrast:{r['id']}" not in used:
            pool["ACT"].append(r["prompt"].strip())
            n_contrast += 1

    amb = load_from_disk(str(ROOT / "datasets" / "ambigqa_light"))["train"]
    n_amb = 0
    for r in amb:
        anns = r.get("annotations") or {}
        types = anns.get("type") if isinstance(anns, dict) else None
        if types and all(t == "singleAnswer" for t in types):
            pool["ACT"].append(r["question"].strip())
            n_amb += 1
        if n_amb >= 4000:
            break

    print("Raw pool:", {k: len(v) for k, v in pool.items()},
          f"(ACT = {n_contrast} contrast + {n_amb} ambigqa)")

    rows = []
    for action, prompts in pool.items():
        seen, uniq = set(), []
        for p in prompts:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        rng.shuffle(uniq)
        for p in uniq[:N_PER_CLASS]:
            rows.append({"prompt": p, "action": action})
    rng.shuffle(rows)

    n_val = int(0.08 * len(rows))
    for i, r in enumerate(rows):
        r["sft_split"] = "val" if i < n_val else "train"

    with (OUT / "sft_train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} SFT rows -> {OUT/'sft_train.jsonl'}")
    print("  by action:", Counter(r["action"] for r in rows))
    print("  by split :", Counter(r["sft_split"] for r in rows))


if __name__ == "__main__":
    main()
