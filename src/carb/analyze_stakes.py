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

# The OpenRouter budget was exhausted partway through the stakes run: 240/240 of GPT-5's
# R2_TYPED HIGH and LOW cells and >50% of its R1 cells came back empty (HTTP 403).  A model
# whose cells are missing asymmetrically across the manipulated factor cannot enter a
# within-item factorial, so GPT-5 is excluded from Experiment 3 and this is stated in the
# report rather than silently pooled.  Missingness per cell is written to the results file.
MAX_MISSING_FRAC = 0.05


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

    miss = (df.assign(na=df.pred.isna())
              .groupby(["model", "regime", "frame"])["na"].mean().to_dict())
    drop = {m for (m, rg, fr), v in miss.items() if v > MAX_MISSING_FRAC}
    if drop:
        print(f"[stakes] excluding models with >{MAX_MISSING_FRAC:.0%} missing responses "
              f"in at least one cell: {sorted(drop)}")
        for (m, rg, fr), v in sorted(miss.items()):
            if m in drop:
                print(f"          {m:34s} {rg:15s} {fr:5s} missing={v:.1%}")
    df.attrs["missing_by_cell"] = {f"{m}|{rg}|{fr}": float(v) for (m, rg, fr), v in miss.items()}
    df.attrs["excluded_models"] = sorted(drop)
    return df[~df.model.isin(drop)].copy()


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


def sdt(df: pd.DataFrame, regime: str) -> dict:
    """Signal-detection decomposition of the stakes effect.

    Treat "the model asked" as a detection response and "the item really needs a question"
    (gold == ASK) as the signal.  Then
        d'  = z(hit rate) - z(false-alarm rate)   how well ask-worthy items are DISCRIMINATED
        c   = -0.5 * (z(hit) + z(fa))             where the ask THRESHOLD sits
    The hypothesis "the model raises its asking threshold when stakes rise" predicts a change
    in c.  The stronger reading -- "it asks more where asking is warranted, without asking
    more elsewhere" -- predicts a change in d'.  Reporting both separates them.

    Rates are corrected by the standard 1/(2N) rule so that 0 and 1 remain finite.
    """
    from scipy.stats import norm

    out = {}
    d = df[(df.regime == regime) & df.pred.notna()]
    for model in [None] + list(STAKES_MODELS):
        dm = d if model is None else d[d.model == model]
        for fr in FRAMES:
            sub = dm[dm.frame == fr]
            sig = sub[sub.gold == "ASK"]
            noi = sub[sub.gold == "ACT"]
            if len(sig) == 0 or len(noi) == 0:
                continue
            h = (sig.pred == "ASK").mean()
            f = (noi.pred == "ASK").mean()
            h = min(max(h, 1 / (2 * len(sig))), 1 - 1 / (2 * len(sig)))
            f = min(max(f, 1 / (2 * len(noi))), 1 - 1 / (2 * len(noi)))
            zh, zf = float(norm.ppf(h)), float(norm.ppf(f))

            # Percentile bootstrap over items, so the figure can carry honest error bars.
            rng = np.random.default_rng(42)
            sig_y = (sig.pred == "ASK").to_numpy().astype(int)
            noi_y = (noi.pred == "ASK").to_numpy().astype(int)
            ds, cs = [], []
            for _ in range(2000):
                hb = sig_y[rng.integers(0, len(sig_y), len(sig_y))].mean()
                fb = noi_y[rng.integers(0, len(noi_y), len(noi_y))].mean()
                hb = min(max(hb, 1 / (2 * len(sig_y))), 1 - 1 / (2 * len(sig_y)))
                fb = min(max(fb, 1 / (2 * len(noi_y))), 1 - 1 / (2 * len(noi_y)))
                zhb, zfb = float(norm.ppf(hb)), float(norm.ppf(fb))
                ds.append(zhb - zfb)
                cs.append(-0.5 * (zhb + zfb))
            out[f"{model or 'pooled'}|{fr}"] = {
                "hit_rate": float(h), "false_alarm_rate": float(f),
                "d_prime": zh - zf, "criterion_c": -0.5 * (zh + zf),
                "d_prime_ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                "criterion_c_ci": [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))],
                "n_signal": int(len(sig)), "n_noise": int(len(noi))}
    return out


def in3_importance() -> dict:
    """Item-level (rather than announced) stakes: IN3 annotates, for every missing detail of a
    vague request, how important that detail is (1-3).  Among IN3 items whose gold action is
    ASK, does the models' ask-rate rise with the annotated importance of what is missing?

    This is the *natural* stakes axis -- nobody told the model the stakes -- and it is the
    weaker test, because the sample is small (26 items at importance 2, 14 at importance 3;
    IN3's sampled subset contains no importance-1 items).  Reported as a descriptive
    contrast with a Fisher exact test, not as a powered comparison.
    """
    from scipy.stats import fisher_exact, mannwhitneyu

    from carb.derive import derive, load_items

    items = load_items("test")
    sel = {i: it for i, it in items.items()
           if it["source"] == "in3" and it["gold_action"] == "ASK" and it.get("in3_importance")}
    out = {"n_items": len(sel),
           "n_by_importance": {str(k): sum(1 for it in sel.values() if it["in3_importance"] == k)
                               for k in (1, 2, 3)}}
    from carb.analyze import MODELS  # same four API models
    for rg in ("R2_TYPED", "R1_AFFORDANCE"):
        tab = {2: [0, 0], 3: [0, 0]}   # importance -> [asked, total]
        for m in MODELS:
            pr = derive("test", m, rg)
            for i, it in sel.items():
                k = it["in3_importance"]
                if k not in tab or i not in pr or pr[i] is None:
                    continue
                tab[k][1] += 1
                tab[k][0] += int(pr[i] == "ASK")
        # item-level aggregation: fraction of models that asked, one value per item.
        # The pooled Fisher test below treats the 4 models' answers on the same item as
        # independent, which they are not; the Mann-Whitney on item-level rates does not.
        per_item = {i: [] for i in sel}
        for m in MODELS:
            pr = derive("test", m, rg)
            for i in sel:
                if pr.get(i) is not None:
                    per_item[i].append(int(pr[i] == "ASK"))
        a2 = [sum(v) / len(v) for i, v in per_item.items() if v and sel[i]["in3_importance"] == 2]
        a3 = [sum(v) / len(v) for i, v in per_item.items() if v and sel[i]["in3_importance"] == 3]
        if len(a2) > 2 and len(a3) > 2:
            u, pu = mannwhitneyu(a3, a2, alternative="two-sided")
            out.setdefault(rg, {}).update({
                "item_level_mean_askrate_importance2": float(sum(a2) / len(a2)),
                "item_level_mean_askrate_importance3": float(sum(a3) / len(a3)),
                "item_level_n2": len(a2), "item_level_n3": len(a3),
                "mannwhitney_u": float(u), "mannwhitney_p": float(pu)})
        if tab[2][1] and tab[3][1]:
            odds, p = fisher_exact([[tab[3][0], tab[3][1] - tab[3][0]],
                                    [tab[2][0], tab[2][1] - tab[2][0]]])
            out.setdefault(rg, {}).update({"ask_rate_importance2": tab[2][0] / tab[2][1],
                       "ask_rate_importance3": tab[3][0] / tab[3][1],
                       "n2": tab[2][1], "n3": tab[3][1],
                       "odds_ratio": float(odds), "fisher_p": float(p)})
    return out


def sdt_delta(df: pd.DataFrame, regime: str, a: str = "HIGH", b: str = "NONE",
              n_boot: int = 4000) -> dict:
    """Paired bootstrap of the CHANGE in d' and criterion between two stakes frames.

    Comparing two separately-bootstrapped intervals is not a test; the quantity of interest is
    the difference, resampled on the same items.  Items are the resampling unit and each
    resampled item carries its responses under BOTH frames, so the pairing is preserved.

    Fully vectorised: all `n_boot` resamples are drawn as one index matrix and `norm.ppf` is
    applied to the whole vector of resampled rates.  (A scalar loop costs ~0.1 s per iteration.)
    """
    from scipy.stats import norm

    d = df[(df.regime == regime) & df.pred.notna()]
    piv = d.pivot_table(index=["model", "item_id"], columns="frame", values="pred",
                        aggfunc="first").dropna(subset=[a, b])
    gold = {i: g for i, g in zip(d.item_id, d.gold)}
    sig = [k for k in piv.index if gold[k[1]] == "ASK"]
    noi = [k for k in piv.index if gold[k[1]] == "ACT"]
    if len(sig) < 20 or len(noi) < 20:
        return {}

    sig_a = (piv.loc[sig, a] == "ASK").to_numpy(dtype=float)
    sig_b = (piv.loc[sig, b] == "ASK").to_numpy(dtype=float)
    noi_a = (piv.loc[noi, a] == "ASK").to_numpy(dtype=float)
    noi_b = (piv.loc[noi, b] == "ASK").to_numpy(dtype=float)

    ns, nn = len(sig), len(noi)
    lo_s, hi_s = 1 / (2 * ns), 1 - 1 / (2 * ns)
    lo_n, hi_n = 1 / (2 * nn), 1 - 1 / (2 * nn)

    def z(rate_s, rate_n):
        return (norm.ppf(np.clip(rate_s, lo_s, hi_s)), norm.ppf(np.clip(rate_n, lo_n, hi_n)))

    zha, zfa = z(sig_a.mean(), noi_a.mean())
    zhb, zfb = z(sig_b.mean(), noi_b.mean())
    da, ca = zha - zfa, -0.5 * (zha + zfa)
    db, cb = zhb - zfb, -0.5 * (zhb + zfb)

    rng = np.random.default_rng(42)
    si = rng.integers(0, ns, (n_boot, ns))
    ni = rng.integers(0, nn, (n_boot, nn))
    ZHA, ZFA = z(sig_a[si].mean(1), noi_a[ni].mean(1))
    ZHB, ZFB = z(sig_b[si].mean(1), noi_b[ni].mean(1))
    dd = (ZHA - ZFA) - (ZHB - ZFB)
    dc = (-0.5 * (ZHA + ZFA)) - (-0.5 * (ZHB + ZFB))

    return {"frames": [a, b], "n_signal": ns, "n_noise": nn, "n_boot": n_boot,
            "delta_d_prime": float(da - db),
            "delta_d_prime_ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
            "delta_criterion": float(ca - cb),
            "delta_criterion_ci": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
            "p_two_sided_d_prime": float(2 * min(np.mean(dd >= 0), np.mean(dd <= 0))),
            "p_two_sided_criterion": float(2 * min(np.mean(dc >= 0), np.mean(dc <= 0)))}


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

    print("\n=== Signal-detection decomposition of the stakes effect (ASK vs ACT items) ===")
    for rg in STAKES_REGIMES:
        sd = sdt(df, rg)
        if not sd:
            continue
        out[f"sdt_{rg}"] = sd
        print(f"\n  {rg}")
        print(f"    {'model':30s} {'frame':6s} {'hit':>7s} {'FA':>7s} {chr(100)+chr(39):>7s} {'c':>7s}")
        for k, v in sd.items():
            mdl, fr = k.split("|")
            print(f"    {mdl:30s} {fr:6s} {v['hit_rate']:7.3f} {v['false_alarm_rate']:7.3f} "
                  f"{v['d_prime']:7.3f} {v['criterion_c']:7.3f}")

    print("\n=== Paired bootstrap of the CHANGE in d' and c (HIGH vs NONE) ===")
    for rg in STAKES_REGIMES:
        dd = sdt_delta(df, rg)
        if not dd:
            continue
        out[f"sdt_delta_{rg}"] = dd
        print(f"  {rg:15s} delta d' = {dd['delta_d_prime']:+.3f} "
              f"[{dd['delta_d_prime_ci'][0]:+.3f}, {dd['delta_d_prime_ci'][1]:+.3f}] "
              f"p={dd['p_two_sided_d_prime']:.3f}   "
              f"delta c = {dd['delta_criterion']:+.3f} "
              f"[{dd['delta_criterion_ci'][0]:+.3f}, {dd['delta_criterion_ci'][1]:+.3f}] "
              f"p={dd['p_two_sided_criterion']:.3f}")

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

    try:
        ii = in3_importance()
        out["in3_importance"] = ii
        print("\n=== Item-level stakes: IN3 annotated importance of the missing detail ===")
        print(f"  n={ii['n_items']} ASK items, by importance {ii['n_by_importance']}")
        for rg in ("R2_TYPED", "R1_AFFORDANCE"):
            if rg in ii:
                v = ii[rg]
                print(f"  {rg:15s} ask-rate imp2={v['ask_rate_importance2']:.3f} (n={v['n2']})  "
                      f"imp3={v['ask_rate_importance3']:.3f} (n={v['n3']})  "
                      f"OR={v['odds_ratio']:.2f} p_fisher={v['fisher_p']:.3f} "
                      f"(anti-conservative); item-level Mann-Whitney p={v.get('mannwhitney_p', float('nan')):.3f}")
    except Exception as e:
        print(f"[in3 importance skipped] {e}")

    (ROOT / "results" / "stakes_analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nWrote results/stakes_analysis.json")


if __name__ == "__main__":
    main()
