"""
Benchmark-validity check: how much of CARB is solvable from surface lexical cues alone?

If a bag-of-words classifier fit on CoCoNot's *training* split can route CARB test items,
then the benchmark rewards style-matching rather than judgment, and any LLM result on it is
suspect.  This is the dataset-artifact check recommended for new benchmarks.

Also reports the degenerate policy baselines (always-ACT, always-ASK, majority, uniform
random) that every headline number must beat.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from carb.build_benchmark import _map_action
from carb.metrics import all_metrics

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def coconot_train_pool() -> list[tuple[str, str]]:
    """(prompt, action) from CoCoNot TRAIN + the contrast items not used in CARB."""
    used = {
        json.loads(l)["item_id"]
        for l in (ROOT / "datasets" / "carb" / "carb_v1.jsonl").read_text().splitlines()
    }
    pool = []
    tr = load_from_disk(str(ROOT / "datasets" / "coconot_original"))["train"]
    for r in tr:
        a = _map_action("coconot", r["category"], r["subcategory"])
        if a:
            pool.append((r["prompt"], a))
    ctr = load_from_disk(str(ROOT / "datasets" / "coconot_contrast"))["test"]
    for r in ctr:
        if f"coconot_contrast:{r['id']}" in used:
            continue  # never train on an item that appears in the benchmark
        pool.append((r["prompt"], "ACT"))
    return pool


def balanced(pool: list[tuple[str, str]], n_per_class: int, rng: random.Random) -> list[tuple[str, str]]:
    by: dict[str, list] = {}
    for p, a in pool:
        by.setdefault(a, []).append((p, a))
    out = []
    for a in sorted(by):
        rows = by[a][:]
        rng.shuffle(rows)
        out.extend(rows[: min(n_per_class, len(rows))])
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(SEED)
    items = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "carb_v1.jsonl").read_text().splitlines()]
    test = [i for i in items if i["split"] == "test"]
    transfer = [i for i in items if i["split"] == "transfer"]

    pool = coconot_train_pool()
    print("CoCoNot train pool:", Counter(a for _, a in pool))
    train = balanced(pool, 239, rng)  # 239 = size of the held-out contrast (ACT) pool
    print("Balanced TF-IDF train set:", Counter(a for _, a in train), f"n={len(train)}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
    X = vec.fit_transform([p for p, _ in train])
    y = [a for _, a in train]
    clf = LogisticRegression(max_iter=2000, C=2.0, random_state=SEED).fit(X, y)

    out = {}
    for name, split in (("test", test), ("transfer", transfer)):
        preds = list(clf.predict(vec.transform([i["prompt"] for i in split])))
        golds = [i["gold_action"] for i in split]
        out[f"tfidf_{name}"] = all_metrics(golds, preds)
        print(f"\nTF-IDF logistic regression on {name}: acc={out[f'tfidf_{name}']['accuracy']:.3f} "
              f"macroF1={out[f'tfidf_{name}']['macro_f1']:.3f}")

    # Degenerate policy baselines on the test split
    golds = [i["gold_action"] for i in test]
    maj = Counter(a for _, a in train).most_common(1)[0][0]
    rng2 = np.random.default_rng(SEED)
    policies = {
        "always_ACT": ["ACT"] * len(golds),
        "always_ASK": ["ASK"] * len(golds),
        "always_REFUSE": ["REFUSE"] * len(golds),
        "majority_class": [maj] * len(golds),
        "uniform_random": list(rng2.choice(["ACT", "ASK", "REFUSE", "DEFER"], len(golds))),
    }
    print("\nDegenerate baselines (test split):")
    for name, preds in policies.items():
        m = all_metrics(golds, preds)
        out[f"baseline_{name}"] = m
        print(f"  {name:16s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"askF1={m['ask_f1']:.3f} overcommit={m['overcommitment']:.3f} "
              f"contrast={m['contrast_compliance']:.3f}")

    (ROOT / "results" / "artifact_check.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results/artifact_check.json")


if __name__ == "__main__":
    main()
