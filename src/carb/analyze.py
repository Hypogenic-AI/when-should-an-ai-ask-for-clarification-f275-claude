"""
Analysis for Experiments 1, 2 and 4: metrics, statistical tests, figures.

Experiment 1 (recognition vs. behaviour, D3): compare the action implied by the model's own
property judgments (R4, applied through a fixed rule) against what it actually does under a
plain prompt (R0) and under a minimal affordance (R1).

Experiment 2 (typed ontology vs. scalar uncertainty, D1): compare R2 (typed categorical
routing) against R3 (a single scalar confidence routed by thresholds fitted on dev).

All model comparisons are within-item, so McNemar's exact test is the appropriate test;
p-values across the pre-registered comparison family are Holm-corrected.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from carb.derive import derive, fit_routers, load_items, scalar_confidences
from carb.metrics import (ACTIONS, accuracy, all_metrics, ask_f1, bootstrap_ci, cohens_h,
                          confusion, contrast_compliance, holm, macro_f1, mcnemar,
                          overcommitment, typed_deferral_accuracy)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ["openai/gpt-5", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
          "meta-llama/llama-3.3-70b-instruct"]
REGIME_ORDER = ["R0_DIRECT", "R1_AFFORDANCE", "R2_TYPED", "R3_SCALAR", "R4_RECOGNITION"]
SHORT = {"openai/gpt-5": "GPT-5", "anthropic/claude-sonnet-4.5": "Claude-Sonnet-4.5",
         "google/gemini-2.5-flash": "Gemini-2.5-Flash",
         "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B",
         "qwen3-8b_base": "Qwen3-8B (base)", "qwen3-8b_sft": "Qwen3-8B (LoRA SFT)"}


def collect(split: str) -> dict:
    """(model, regime) -> {item_id: pred}. Also returns fitted scalar routers per model."""
    routers = {m: fit_routers(m) for m in MODELS}
    preds = {}
    for m in MODELS:
        for rg in REGIME_ORDER:
            p = derive(split, m, rg, routers.get(m))
            if p:
                preds[(m, rg)] = p
    return {"preds": preds, "routers": routers}


def aligned(items: dict, pred: dict) -> tuple[list[str], list[str | None], list[str]]:
    ids = [i for i in items if i in pred]
    return ([items[i]["gold_action"] for i in ids], [pred[i] for i in ids], ids)


def main() -> None:
    RES.mkdir(exist_ok=True)
    report: dict = {}

    for split in ("test",):
        items = load_items(split)
        c = collect(split)
        report.setdefault("routers", {k: v for k, v in c["routers"].items()})
        rows = []
        for (m, rg), pred in sorted(c["preds"].items()):
            golds, preds_, ids = aligned(items, pred)
            if not golds:
                continue
            met = all_metrics(golds, preds_)
            met.update({"model": m, "regime": rg, "split": split})
            for name, fn in (("accuracy", accuracy), ("macro_f1", macro_f1),
                             ("ask_f1", lambda g, p: ask_f1(g, p)[0]),
                             ("overcommitment", overcommitment),
                             ("contrast_compliance", contrast_compliance),
                             ("typed_deferral_acc", typed_deferral_accuracy)):
                lo, hi = bootstrap_ci(golds, preds_, fn)
                met[f"{name}_ci"] = [lo, hi]
            met["confusion"] = confusion(golds, preds_).tolist()
            met["pred_dist"] = dict(Counter([p if p in ACTIONS else "NA" for p in preds_]))
            # per-source breakdown: separates in-source (CoCoNot) from out-of-source arms
            met["by_source"] = {}
            for src in sorted({items[i]["source"] for i in ids}):
                sel = [k for k, i in enumerate(ids) if items[ids[k]]["source"] == src]
                g2 = [golds[k] for k in sel]
                p2 = [preds_[k] for k in sel]
                met["by_source"][src] = {"n": len(sel), "accuracy": accuracy(g2, p2),
                                         "macro_f1": macro_f1(g2, p2)}
            rows.append(met)
        report[f"metrics_{split}"] = rows

    # ----------------------------------------------- label-mapping sensitivity (robustness)
    # Two cells of the pre-registered mapping are genuinely contestable:
    #   Incomplete/false-presuppositions -> ASK   (a model that corrects the premise instead
    #       reads as REFUSE to the judge)
    #   Indeterminate/subjective-questions -> REFUSE  (arguably ASK-for-preference)
    # We therefore recompute the headline accuracy (a) with those items dropped and
    # (b) under an alternative mapping that sends false-presupposition items to REFUSE.
    items_all = load_items("test")
    c_test = collect("test")
    sens = {}
    for (m, rg), pred in sorted(c_test["preds"].items()):
        ids = [i for i in items_all if i in pred]
        g = [items_all[i]["gold_action"] for i in ids]
        p_ = [pred[i] for i in ids]
        keep = [k for k, i in enumerate(ids) if not items_all[i]["contested"]]
        g_alt = [
            "REFUSE" if items_all[i]["src_subcategory"] == "false presuppositions" else items_all[i]["gold_action"]
            for i in ids
        ]
        sens[f"{m}|{rg}"] = {
            "acc_preregistered": accuracy(g, p_),
            "acc_drop_contested": accuracy([g[k] for k in keep], [p_[k] for k in keep]),
            "n_drop_contested": len(keep),
            "acc_altmap_falsepresup_refuse": accuracy(g_alt, p_),
            "acc_v11_in3_capability_audit": accuracy(
                [items_all[i]["gold_action_v11"] for i in ids], p_),
        }
    report["label_sensitivity"] = sens

    # ------------------------------------------------------------------ statistical tests
    items = load_items("test")
    c = collect("test")
    tests: dict[str, dict] = {}
    for m in MODELS:
        for a, b, name in (("R4_RECOGNITION", "R0_DIRECT", "E1_recognition_vs_direct"),
                           ("R4_RECOGNITION", "R1_AFFORDANCE", "E1_recognition_vs_affordance"),
                           ("R1_AFFORDANCE", "R0_DIRECT", "E1_affordance_vs_direct"),
                           ("R2_TYPED", "R3_SCALAR", "E2_typed_vs_scalar"),
                           ("R2_TYPED", "R1_AFFORDANCE", "E2_typed_vs_affordance"),
                           ("R2_TYPED", "R4_RECOGNITION", "E2_typed_vs_recognition")):
            pa, pb = c["preds"].get((m, a)), c["preds"].get((m, b))
            if not pa or not pb:
                continue
            ids = [i for i in items if i in pa and i in pb]
            g = [items[i]["gold_action"] for i in ids]
            xa = [pa[i] for i in ids]
            xb = [pb[i] for i in ids]
            r = mcnemar(g, xa, xb)
            r.update({"acc_a": accuracy(g, xa), "acc_b": accuracy(g, xb), "n": len(ids),
                      "cohens_h": cohens_h(accuracy(g, xa), accuracy(g, xb))})
            tests[f"{name}|{m}"] = r
    adj = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    report["mcnemar"] = tests

    # ---------------------------------------------------- scalar-uncertainty diagnostics (D1)
    diag = {}
    for m in MODELS:
        confs_test = scalar_confidences("test", m)
        pairs = [(c_, items[i]["gold_action"]) for i, c_ in confs_test.items() if c_ is not None and i in items]
        if not pairs:
            continue
        by_action = {a: [c_ for c_, g in pairs if g == a] for a in ACTIONS}
        # AUROC of "should not act" vs the scalar
        from sklearn.metrics import roc_auc_score
        y = [0 if g == "ACT" else 1 for _, g in pairs]
        s = [-c_ for c_, _ in pairs]
        auroc_act = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
        # pairwise AUROC among the three deferral types: can the scalar tell them apart?
        pair_auroc = {}
        for a in ("ASK", "REFUSE", "DEFER"):
            for b in ("ASK", "REFUSE", "DEFER"):
                if a >= b:
                    continue
                sub = [(c_, g) for c_, g in pairs if g in (a, b)]
                if len({g for _, g in sub}) < 2:
                    continue
                pair_auroc[f"{a}_vs_{b}"] = float(
                    roc_auc_score([1 if g == a else 0 for _, g in sub], [c_ for c_, _ in sub]))
        diag[m] = {
            "mean_conf_by_action": {a: (float(np.mean(v)) if v else None) for a, v in by_action.items()},
            "std_conf_by_action": {a: (float(np.std(v)) if v else None) for a, v in by_action.items()},
            "auroc_act_vs_notact": auroc_act,
            "pairwise_auroc_deferral": pair_auroc,
            "router": c["routers"].get(m),
            "n": len(pairs),
        }
    report["scalar_diagnostics"] = diag

    # ------------------------------------------------- judge reliability vs. human annotation
    hv = RES / "judge_validation.json"
    report["judge_agreement"] = json.loads(hv.read_text()) if hv.exists() else {}

    (RES / "main_analysis.json").write_text(json.dumps(report, indent=2, default=float))
    print("Wrote results/main_analysis.json")

    # ----------------------------------------------------------------------------- printout
    print("\n=== TEST SPLIT: 4-way routing ===")
    hdr = f"{'model':22s} {'regime':16s} {'acc':>6s} {'mF1':>6s} {'askF1':>6s} {'overcom':>8s} {'contrast':>9s} {'typedDef':>9s}"
    print(hdr)
    for r in sorted(report["metrics_test"], key=lambda r: (r["model"], REGIME_ORDER.index(r["regime"]))):
        print(f"{SHORT.get(r['model'], r['model']):22s} {r['regime']:16s} {r['accuracy']:6.3f} "
              f"{r['macro_f1']:6.3f} {r['ask_f1']:6.3f} {r['overcommitment']:8.3f} "
              f"{r['contrast_compliance']:9.3f} {r['typed_deferral_acc']:9.3f}")

    print("\n=== McNemar (Holm-corrected) ===")
    for k, v in sorted(report["mcnemar"].items()):
        print(f"  {k:60s} acc {v['acc_a']:.3f} vs {v['acc_b']:.3f}  b={v['b']:3d} c={v['c']:3d}  "
              f"p={v['p']:.2e} p_holm={v['p_holm']:.2e} h={v['cohens_h']:+.2f}")

    print("\n=== Scalar-uncertainty diagnostics ===")
    for m, d in report["scalar_diagnostics"].items():
        print(f"  {SHORT.get(m,m)}: AUROC(act vs not-act)={d['auroc_act_vs_notact']:.3f}")
        print(f"     mean conf: {({k: (round(v,1) if v is not None else None) for k,v in d['mean_conf_by_action'].items()})}")
        print(f"     pairwise AUROC among deferral types: "
              f"{({k: round(v,3) for k,v in d['pairwise_auroc_deferral'].items()})}")

    print("\n=== Label-mapping sensitivity (test split accuracy) ===")
    print(f"  {'model|regime':52s} {'pre-reg':>8s} {'-contested':>11s} {'alt-map':>8s} {'v1.1':>7s}")
    for k, v in sorted(report["label_sensitivity"].items()):
        print(f"  {k:52s} {v['acc_preregistered']:8.3f} {v['acc_drop_contested']:11.3f} "
              f"{v['acc_altmap_falsepresup_refuse']:8.3f} {v['acc_v11_in3_capability_audit']:7.3f}")

    ja = report.get("judge_agreement", {})
    if ja:
        print("\n=== Judge reliability vs. hand annotation ===")
        print(f"  n={ja.get('n')} agreement={ja.get('raw_agreement'):.3f} kappa={ja.get('cohens_kappa'):.3f}")


if __name__ == "__main__":
    main()
