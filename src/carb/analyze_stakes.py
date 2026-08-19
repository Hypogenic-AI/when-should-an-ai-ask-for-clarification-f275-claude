"""
Experiment 3 analysis: does announced stakes move the ask/refuse threshold?

Within-item design (each item seen under NONE / LOW / HIGH), so the tests are paired.
Primary contrast is HIGH vs LOW on:
  * ask-rate among items whose gold action is ASK  (does higher stakes buy more asking
    where asking is warranted?)
  * act-rate among items whose gold action is ACT  (does it cost over-caution where acting
    is warranted?)
A stakes-sensitive policy moves the first without destroying the second.

Secondary: a cluster-robust logistic regression of P(ask) on stakes x ambiguity, with
standard errors clustered by item, pooled across models.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from carb.llm import parse_json_block
from carb.run_stakes import STAKES_MODELS, STAKES_REGIMES, sample_items

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw_stakes"
FRAMES = ["NONE", "LOW", "HIGH"]
ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]


def load_preds() -> pd.DataFrame:
    items = {it["item_id"]: it for it in sample_items()}
    rows = []
    for m in STAKES_MODELS:
        for rg in STAKES_REGIMES:
            for fr in FRAMES:
                if rg == "R1_AFFORDANCE":
                    p = RAW / f"judged__{m.replace('/','_')}__{rg}__{fr}.jsonl"
                    if not p.exists():
                        continue
                    for l in p.read_text().splitlines():
                        d = json.loads(l)
                        rows.append({"model": m, "regime": rg, "frame": fr,
                                     "item_id": d["item_id"], "pred": d.get("behaviour")})
                else:
                    p = RAW / f"{m.replace('/','_')}__{rg}__{fr}.jsonl"
                    if not p.exists():
                        continue
                    for l in p.read_text().splitlines():
                        d = json.loads(l)
                        j = parse_json_block(d["raw"]) or {}
                        a = str(j.get("action", "")).strip().upper()
                        rows.append({"model": m, "regime": rg, "frame": fr,
                                     "item_id": d["item_id"], "pred": a if a in ACTIONS else None})
    df = pd.DataFrame(rows)
    df["gold"] = df["item_id"].map(lambda i: items[i]["gold_action"])
    return df


def paired_test(df: pd.DataFrame, gold: str, target: str) -> dict:
    """McNemar-style exact test of frame HIGH vs LOW on P(pred == target) for gold-`gold` items."""
    sub = df[df.gold == gold]
    piv = sub.pivot_table(index=["model", "item_id"], columns="frame", values="pred",
                          aggfunc="first")
    if "HIGH" not in piv or "LOW" not in piv:
        return {}
    hi = piv["HIGH"] == target
    lo = piv["LOW"] == target
    b = int((hi & ~lo).sum())   # switched INTO target when stakes rose
    c = int((~hi & lo).sum())   # switched OUT of target when stakes rose
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"n_pairs": int(len(piv)), "rate_low": float(lo.mean()), "rate_high": float(hi.mean()),
            "delta": float(hi.mean() - lo.mean()), "b_low_to_high": b, "c_high_to_low": c,
            "p": float(p)}


def main() -> None:
    df = load_preds()
    if df.empty:
        print("No stakes data found; run src/carb/run_stakes.py first.")
        return
    out: dict = {"n_rows": int(len(df)), "unparsed": int(df.pred.isna().sum())}

    print("=== Ask-rate / act-rate by frame (pooled over models) ===")
    tbl = {}
    for rg in STAKES_REGIMES:
        for gold in ACTIONS:
            for fr in FRAMES:
                sel = df[(df.regime == rg) & (df.gold == gold) & (df.frame == fr)]
                if sel.empty:
                    continue
                tbl[f"{rg}|{gold}|{fr}"] = {
                    "n": int(len(sel)),
                    "p_ask": float((sel.pred == "ASK").mean()),
                    "p_act": float((sel.pred == "ACT").mean()),
                    "p_refuse": float((sel.pred == "REFUSE").mean()),
                    "p_defer": float((sel.pred == "DEFER").mean()),
                    "accuracy": float((sel.pred == sel.gold).mean()),
                }
    out["cells"] = tbl
    for rg in STAKES_REGIMES:
        print(f"\n  {rg}")
        print(f"    {'gold':8s} {'frame':6s} {'P(ask)':>8s} {'P(act)':>8s} {'P(ref)':>8s} {'P(def)':>8s} {'acc':>7s}")
        for gold in ACTIONS:
            for fr in FRAMES:
                k = f"{rg}|{gold}|{fr}"
                if k in tbl:
                    v = tbl[k]
                    print(f"    {gold:8s} {fr:6s} {v['p_ask']:8.3f} {v['p_act']:8.3f} "
                          f"{v['p_refuse']:8.3f} {v['p_defer']:8.3f} {v['accuracy']:7.3f}")

    print("\n=== Paired HIGH vs LOW contrasts ===")
    contrasts = {}
    for rg in STAKES_REGIMES:
        d = df[df.regime == rg]
        contrasts[f"{rg}|ASKgold_pAsk"] = paired_test(d, "ASK", "ASK")
        contrasts[f"{rg}|ACTgold_pAct"] = paired_test(d, "ACT", "ACT")
        contrasts[f"{rg}|ACTgold_pAsk"] = paired_test(d, "ACT", "ASK")
        contrasts[f"{rg}|REFUSEgold_pRefuse"] = paired_test(d, "REFUSE", "REFUSE")
        for m in STAKES_MODELS:
            dm = d[d.model == m]
            contrasts[f"{rg}|{m}|ASKgold_pAsk"] = paired_test(dm, "ASK", "ASK")
            contrasts[f"{rg}|{m}|ACTgold_pAct"] = paired_test(dm, "ACT", "ACT")
    from carb.metrics import holm
    valid = {k: v["p"] for k, v in contrasts.items() if v}
    adj = holm(valid)
    for k, v in contrasts.items():
        if v:
            v["p_holm"] = adj[k]
    out["contrasts"] = contrasts
    for k, v in contrasts.items():
        if v:
            print(f"  {k:60s} low={v['rate_low']:.3f} high={v['rate_high']:.3f} "
                  f"d={v['delta']:+.3f} p={v['p']:.2e} p_holm={v['p_holm']:.2e}")

    # cluster-robust logistic regression of P(ASK) on stakes x ambiguity
    try:
        import statsmodels.formula.api as smf
        d = df[(df.regime == "R2_TYPED") & df.pred.notna()].copy()
        d["y"] = (d.pred == "ASK").astype(int)
        d["ambiguous"] = (d.gold == "ASK").astype(int)
        d["high"] = (d.frame == "HIGH").astype(int)
        d["low"] = (d.frame == "LOW").astype(int)
        mres = smf.logit("y ~ ambiguous * (high + low) + C(model)", data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["item_id"]})
        out["logit_R2"] = {"params": mres.params.to_dict(), "pvalues": mres.pvalues.to_dict(),
                           "n": int(mres.nobs)}
        print("\n=== Logistic regression: P(ASK) ~ ambiguous * stakes (item-clustered SE, R2_TYPED) ===")
        print(mres.summary2().tables[1].to_string())
    except Exception as e:
        print(f"[logit skipped] {e}")

    (ROOT / "results" / "stakes_analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nWrote results/stakes_analysis.json")


if __name__ == "__main__":
    main()
