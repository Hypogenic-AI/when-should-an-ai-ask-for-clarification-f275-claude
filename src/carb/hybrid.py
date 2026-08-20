"""
Is the scalar confidence *complementary* to the typed judgments, or redundant with them?

The main analysis shows a single scalar separates ACT from not-ACT reasonably well (AUROC
0.79-0.87) but barely separates the three kinds of withholding from one another
(DEFER-vs-REFUSE AUROC 0.44-0.56, i.e. chance).  That suggests an obvious hybrid:

    stage 1  use the scalar, with the best binary threshold fitted by cross-validation,
             to decide ACT vs withhold  -- the thing the scalar is good at;
    stage 2  if withholding, use the model's own typed judgments to decide WHICH kind --
             the thing the scalar is bad at.

Two versions of stage 2 are evaluated: the typed router's own choice (R2) and the fixed rule
applied to the recognition judgments (R4).  If either hybrid beats both of its parents, the
scalar carries information the typed regime is missing; if it merely tracks its stage-2 parent,
the scalar is redundant and "estimate your uncertainty" adds nothing to "choose a type".
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from carb.derive import derive, load_items, scalar_confidences
from carb.metrics import ACTIONS, accuracy, all_metrics, bootstrap_ci, holm, mcnemar
from carb.routing import fit_binary_scalar_router

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ["openai/gpt-5", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
          "meta-llama/llama-3.3-70b-instruct"]
SEED = 42
K_FOLDS = 5


def cv_binary_act(split: str, model: str) -> dict[str, bool | None]:
    """Cross-validated ACT / not-ACT decision from the scalar alone (no label leakage)."""
    items = load_items(split)
    confs = scalar_confidences(split, model)
    ids = sorted([i for i, c in confs.items() if c is not None and i in items])
    if len(ids) < 40:
        return {}
    rng = random.Random(SEED)
    order = ids[:]
    rng.shuffle(order)
    folds = [order[j::K_FOLDS] for j in range(K_FOLDS)]
    out: dict[str, bool | None] = {i: None for i in confs}
    for j in range(K_FOLDS):
        tr = [i for f, fold in enumerate(folds) if f != j for i in fold]
        r = fit_binary_scalar_router([confs[i] for i in tr],
                                     [items[i]["gold_action"] for i in tr])
        for i in folds[j]:
            out[i] = (confs[i] > r["cut"]) if r["hi_is_act"] else (confs[i] <= r["cut"])
    return out


def main() -> None:
    items = load_items("test")
    ids = sorted(items)
    golds = [items[i]["gold_action"] for i in ids]
    report: dict = {"models": {}}
    pvals: dict[str, float] = {}

    for m in MODELS:
        act = cv_binary_act("test", m)
        if not act:
            continue
        typed = derive("test", m, "R2_TYPED")
        recog = derive("test", m, "R4_RECOGNITION")
        scalar4 = derive("test", m, "R3_SCALAR")

        def hybrid(stage2: dict) -> list[str | None]:
            out = []
            for i in ids:
                a = act.get(i)
                if a is None:
                    out.append(None)
                elif a:
                    out.append("ACT")
                else:
                    s = stage2.get(i)
                    # stage 2 must name a kind of withholding; if it says ACT (contradicting
                    # stage 1) or is unparseable, fall back to ASK, the least destructive
                    # withholding action.
                    out.append(s if s in ("ASK", "REFUSE", "DEFER") else "ASK")
            return out

        cands = {
            "R3_SCALAR (4-way, CV router)": [scalar4.get(i) for i in ids],
            "R2_TYPED": [typed.get(i) for i in ids],
            "R4_RECOGNITION": [recog.get(i) for i in ids],
            "hybrid: scalar gate + typed kind": hybrid(typed),
            "hybrid: scalar gate + recognition kind": hybrid(recog),
        }
        d = {}
        for name, preds in cands.items():
            met = all_metrics(golds, preds)
            lo, hi = bootstrap_ci(golds, preds, accuracy)
            met["accuracy_ci"] = [lo, hi]
            d[name] = met
        # is the hybrid better than the typed policy it is built on?
        for h in ("hybrid: scalar gate + typed kind", "hybrid: scalar gate + recognition kind"):
            parent = "R2_TYPED" if "typed" in h else "R4_RECOGNITION"
            t = mcnemar(golds, cands[h], cands[parent])
            d[h]["vs_parent"] = {"parent": parent, **t}
            pvals[f"{m}|{h}"] = t["p"]
        report["models"][m] = d

    adj = holm(pvals)
    for m, d in report["models"].items():
        for h in list(d):
            if "vs_parent" in d[h]:
                d[h]["vs_parent"]["p_holm"] = adj[f"{m}|{h}"]

    (RES / "hybrid_analysis.json").write_text(json.dumps(report, indent=2, default=float))
    print("Wrote results/hybrid_analysis.json\n")
    print(f"{'model':26s} {'policy':42s} {'acc':>6s} {'typedDef':>9s} {'vs parent p_holm':>17s}")
    for m, d in report["models"].items():
        for name, met in d.items():
            vp = met.get("vs_parent")
            p = f"{vp['p_holm']:.2e}" if vp else ""
            print(f"{m[:26]:26s} {name:42s} {met['accuracy']:6.3f} "
                  f"{met['typed_deferral_acc']:9.3f} {p:>17s}")


if __name__ == "__main__":
    main()
