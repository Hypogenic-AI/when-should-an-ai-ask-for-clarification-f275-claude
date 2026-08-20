"""
Experiment 4 analysis: prompting vs. LoRA training vs. frozen-representation probe,
all on the same open-weight model (Qwen3-4B).

Answers two questions the API experiments cannot:
  * Is the routing decision *trainable* on a small open model with a modest LoRA budget?
  * Is the information already linearly decodable from the base model's representation,
    i.e. is the failure one of access rather than of knowledge?
"""
from __future__ import annotations

import json
from pathlib import Path

from carb.derive import load_items
from carb.llm import parse_json_block
from carb.metrics import ACTIONS, all_metrics, bootstrap_ci, accuracy, macro_f1, mcnemar
from carb.routing import (apply_scalar_router, fit_binary_scalar_router, fit_scalar_router,
                          recognition_to_action)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw_local"


def _rows(tag: str, split: str, regime: str) -> list[dict] | None:
    p = RAW / f"{tag}__{split}__{regime}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def judged_from(tag: str, split: str, regime: str) -> dict[str, str | None]:
    """Behaviour labels for the free-text regimes, from the same local judge used for the
    API models (see src/carb/local_judge.py --what local)."""
    p = RAW / f"judged__{tag}__{split}__{regime}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["item_id"]: json.loads(l)["behaviour"] for l in p.read_text().splitlines()}


def preds_from(tag: str, split: str, regime: str, router: dict | None = None) -> dict[str, str | None]:
    if regime in ("R0_DIRECT", "R1_AFFORDANCE"):
        return judged_from(tag, split, regime)
    rows = _rows(tag, split, regime)
    if rows is None:
        return {}
    out = {}
    for r in rows:
        j = parse_json_block(r["raw"])
        if regime == "R2_TYPED":
            a = str((j or {}).get("action", "")).strip().upper()
            out[r["item_id"]] = a if a in ACTIONS else None
        elif regime == "R4_RECOGNITION":
            out[r["item_id"]] = recognition_to_action(j) if j else None
        elif regime == "R3_SCALAR":
            try:
                c = min(100.0, max(0.0, float(j["confidence"])))
                out[r["item_id"]] = apply_scalar_router(c, router) if router else None
            except (KeyError, TypeError, ValueError):
                out[r["item_id"]] = None
    return out


def main() -> None:
    base_tag = next((p.name.split("__")[0] for p in sorted(RAW.glob("*_base__test__R2_TYPED.jsonl"))), None)
    sft_tag = next((p.name.split("__")[0] for p in sorted(RAW.glob("*_sft__test__R2_TYPED.jsonl"))), None)
    if base_tag is None:
        print("No local results found.")
        return

    # scalar router fitted on the dev split, exactly as for the API models
    dev_items = load_items("dev")
    dev_rows = _rows(base_tag, "dev", "R3_SCALAR") or []
    cs, gs = [], []
    for r in dev_rows:
        j = parse_json_block(r["raw"]) or {}
        try:
            cs.append(min(100.0, max(0.0, float(j["confidence"]))))
            gs.append(dev_items[r["item_id"]]["gold_action"])
        except (KeyError, TypeError, ValueError):
            pass
    router = fit_scalar_router(cs, gs) if len(cs) > 20 else None
    binrouter = fit_binary_scalar_router(cs, gs) if len(cs) > 20 else None

    probe = json.loads((ROOT / "results" / "probe_preds.json").read_text()) if (
        ROOT / "results" / "probe_preds.json").exists() else {}

    report: dict = {"router": router, "binary_router": binrouter, "base_tag": base_tag, "sft_tag": sft_tag}
    for split in ("test", "transfer"):
        items = load_items(split)
        conds: dict[str, dict[str, str | None]] = {
            "behaviour: plain prompt (R0)": preds_from(base_tag, split, "R0_DIRECT"),
            "behaviour: ask-affordance (R1)": preds_from(base_tag, split, "R1_AFFORDANCE"),
            "prompted: typed ontology (R2)": preds_from(base_tag, split, "R2_TYPED"),
            "prompted: scalar confidence (R3)": preds_from(base_tag, split, "R3_SCALAR", router),
            "prompted: recognition + rule (R4)": preds_from(base_tag, split, "R4_RECOGNITION"),
        }
        if sft_tag:
            conds["trained: LoRA SFT (R2 prompt)"] = preds_from(sft_tag, split, "R2_TYPED")
        if split in probe:
            conds["frozen-representation linear probe"] = dict(zip(probe[split]["item_id"], probe[split]["pred"]))

        out = {}
        for name, pred in conds.items():
            ids = [i for i in items if i in pred]
            if not ids:
                continue
            g = [items[i]["gold_action"] for i in ids]
            p = [pred[i] for i in ids]
            m = all_metrics(g, p)
            m["accuracy_ci"] = list(bootstrap_ci(g, p, accuracy))
            m["macro_f1_ci"] = list(bootstrap_ci(g, p, macro_f1))
            out[name] = m
        report[split] = out

        # paired significance: SFT vs prompted, probe vs prompted
        tests = {}
        base = conds.get("prompted: typed ontology (R2)", {})
        for name in ("trained: LoRA SFT (R2 prompt)", "frozen-representation linear probe"):
            if name not in conds:
                continue
            ids = [i for i in items if i in base and i in conds[name]]
            g = [items[i]["gold_action"] for i in ids]
            tests[name] = mcnemar(g, [conds[name][i] for i in ids], [base[i] for i in ids])
        report[f"{split}_tests_vs_prompted_typed"] = tests

    (ROOT / "results" / "local_analysis.json").write_text(json.dumps(report, indent=2, default=float))

    for split in ("test", "transfer"):
        print(f"\n=== Qwen3-4B on {split} split ===")
        print(f"{'condition':38s} {'acc':>6s} {'95% CI':>16s} {'mF1':>6s} {'askF1':>6s} {'overcom':>8s} {'contrast':>9s}")
        for name, m in report.get(split, {}).items():
            ci = m["accuracy_ci"]
            print(f"{name:38s} {m['accuracy']:6.3f} [{ci[0]:.3f},{ci[1]:.3f}] {m['macro_f1']:6.3f} "
                  f"{m['ask_f1']:6.3f} {m['overcommitment']:8.3f} {m['contrast_compliance']:9.3f}")
        for name, t in report.get(f"{split}_tests_vs_prompted_typed", {}).items():
            print(f"  McNemar vs prompted-typed | {name:40s} b={t['b']:3d} c={t['c']:3d} p={t['p']:.2e}")
    print("\nWrote results/local_analysis.json")


if __name__ == "__main__":
    main()
