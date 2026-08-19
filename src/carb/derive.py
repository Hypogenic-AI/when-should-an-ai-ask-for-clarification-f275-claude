"""Turn raw model outputs into predicted actions for every regime.

R0/R1  -> the behavioural judge's label (loaded from judged__*.jsonl)
R2     -> parsed "action" field
R3     -> scalar confidence, routed by thresholds FIT ON THE DEV SPLIT
R4     -> four property judgments, mapped by the fixed rule in routing.py
"""
from __future__ import annotations

import json
from pathlib import Path

from carb.llm import parse_json_block
from carb.routing import apply_scalar_router, fit_binary_scalar_router, fit_scalar_router, recognition_to_action

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
ACTIONS = {"ACT", "ASK", "REFUSE", "DEFER"}


_AUDIT = json.loads((ROOT / "datasets" / "carb" / "in3_capability_audit.json").read_text())
IN3_RELABEL: dict[str, str] = {k: "DEFER" for k in _AUDIT["relabel_to_DEFER"]}


def load_items(split: str) -> dict[str, dict]:
    """Benchmark items for a split.

    Each item carries both the pre-registered gold label (`gold_action`) and the v1.1
    capability-audited label (`gold_action_v11`); see datasets/carb/in3_capability_audit.json.
    Headline results use `gold_action`; `gold_action_v11` is reported as a sensitivity check.
    """
    rows = [json.loads(l) for l in (ROOT / "datasets" / "carb" / "carb_v1.jsonl").read_text().splitlines()]
    out = {}
    for r in rows:
        if r["split"] != split:
            continue
        r["gold_action_v11"] = IN3_RELABEL.get(r["item_id"], r["gold_action"])
        out[r["item_id"]] = r
    return out


def _raw_rows(split: str, model: str, regime: str) -> list[dict] | None:
    p = RAW / f"{split}__{model.replace('/','_')}__{regime}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def _judged_rows(split: str, model: str, regime: str, tag: str = "") -> dict[str, str | None]:
    p = RAW / f"judged{tag}__{split}__{model.replace('/','_')}__{regime}.jsonl"
    if not p.exists():
        return {}
    return {r["item_id"]: r["behaviour"] for r in (json.loads(l) for l in p.read_text().splitlines())}


def scalar_confidences(split: str, model: str) -> dict[str, float | None]:
    rows = _raw_rows(split, model, "R3_SCALAR") or []
    out = {}
    for r in rows:
        j = parse_json_block(r["raw"])
        c = None
        if j is not None and "confidence" in j:
            try:
                c = float(j["confidence"])
                c = min(100.0, max(0.0, c))
            except (TypeError, ValueError):
                c = None
        out[r["item_id"]] = c
    return out


def fit_routers(model: str, split: str = "test") -> dict:
    """Fit the scalar router on `split`.  Returns both the 4-way and binary routers.

    NOTE: the in-sample router reported here is an *upper bound* on what a single scalar can
    do; it is used only for reporting the fitted thresholds. All accuracy numbers for R3 come
    from `cv_scalar_preds`, which is cross-validated and therefore leakage-free.
    (The study originally reserved a separate dev split for this; the API budget was exhausted
    before dev could be collected, so k-fold cross-validation on the test split replaced it.
    Cross-validation is if anything the stricter choice, since every item's prediction comes
    from a router that never saw it.)
    """
    items = load_items(split)
    confs = scalar_confidences(split, model)
    pairs = [(c, items[i]["gold_action"]) for i, c in confs.items() if c is not None and i in items]
    if len(pairs) < 20:
        return {}
    cs, gs = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "four_way": fit_scalar_router(cs, gs),
        "binary": fit_binary_scalar_router(cs, gs),
        "n_fit": len(pairs),
    }


def cv_scalar_preds(split: str, model: str, k: int = 5, seed: int = 42) -> dict[str, str | None]:
    """Leakage-free R3 predictions: k-fold CV in which each fold's thresholds are fitted on
    the other k-1 folds.  Gives the scalar-uncertainty baseline the best router that can be
    fitted without seeing the item being predicted."""
    import random as _random

    items = load_items(split)
    confs = scalar_confidences(split, model)
    ids = sorted([i for i, c in confs.items() if c is not None and i in items])
    if len(ids) < 40:
        return {}
    rng = _random.Random(seed)
    order = ids[:]
    rng.shuffle(order)
    folds = [order[j::k] for j in range(k)]
    out: dict[str, str | None] = {i: None for i in confs}
    for j in range(k):
        train_ids = [i for f, fold in enumerate(folds) if f != j for i in fold]
        cs = [confs[i] for i in train_ids]
        gs = [items[i]["gold_action"] for i in train_ids]
        router = fit_scalar_router(cs, gs)
        for i in folds[j]:
            out[i] = apply_scalar_router(confs[i], router)
    return out


def derive(split: str, model: str, regime: str, routers: dict | None = None, judge_tag: str = "") -> dict[str, str | None]:
    """item_id -> predicted action (or None if unparseable / call failed)."""
    if regime in ("R0_DIRECT", "R1_AFFORDANCE"):
        return _judged_rows(split, model, regime, judge_tag)

    rows = _raw_rows(split, model, regime)
    if rows is None:
        return {}

    out: dict[str, str | None] = {}
    if regime == "R2_TYPED":
        for r in rows:
            j = parse_json_block(r["raw"]) or {}
            a = str(j.get("action", "")).strip().upper()
            if a not in ACTIONS and r["raw"]:  # fall back to a bare mention of one action word
                hits = [x for x in ACTIONS if x in r["raw"].upper()]
                a = hits[0] if len(hits) == 1 else ""
            out[r["item_id"]] = a if a in ACTIONS else None
    elif regime == "R3_SCALAR":
        # cross-validated thresholds; see cv_scalar_preds
        out = cv_scalar_preds(split, model)
    elif regime == "R4_RECOGNITION":
        for r in rows:
            j = parse_json_block(r["raw"])
            out[r["item_id"]] = recognition_to_action(j) if j else None
    return out
