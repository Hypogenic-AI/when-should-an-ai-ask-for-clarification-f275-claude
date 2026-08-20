"""
Collect every results/*.json into one compact markdown file (results/SUMMARY.md).

The report quotes these tables verbatim, so there is a single place where a number can be
checked against the artefact that produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
REGIME_ORDER = ["R0_DIRECT", "R1_AFFORDANCE", "R2_TYPED", "R3_SCALAR", "R4_RECOGNITION"]
SHORT = {"openai/gpt-5": "GPT-5", "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
         "google/gemini-2.5-flash": "Gemini 2.5 Flash",
         "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"}
ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]


def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def main() -> None:
    out: list[str] = ["# CARB results summary", ""]

    main_a = load("main_analysis.json")
    if main_a:
        out += ["## Table 1 — Test split (n=480), 4-way routing by elicitation regime", "",
                "| model | regime | acc [95% CI] | macro-F1 | ASK-F1 | over-commit | contrast compl. | typed-deferral | unparsed |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(main_a["metrics_test"],
                        key=lambda r: (r["model"], REGIME_ORDER.index(r["regime"]))):
            ci = r.get("accuracy_ci", [None, None])
            out.append(f"| {SHORT.get(r['model'], r['model'])} | {r['regime']} | "
                       f"{f(r['accuracy'])} [{f(ci[0])}, {f(ci[1])}] | {f(r['macro_f1'])} | "
                       f"{f(r['ask_f1'])} | {f(r['overcommitment'])} | "
                       f"{f(r['contrast_compliance'])} | {f(r['typed_deferral_acc'])} | "
                       f"{r['unparsed']} |")
        out.append("")

        out += ["## Table 2 — Paired within-item comparisons (McNemar exact, Holm-corrected)", "",
                "| comparison | model | acc A | acc B | b | c | p | p (Holm) | Cohen's h |",
                "|---|---|---|---|---|---|---|---|---|"]
        for k, v in sorted(main_a.get("mcnemar", {}).items()):
            name, m = k.split("|", 1)
            out.append(f"| {name} | {SHORT.get(m, m)} | {f(v['acc_a'])} | {f(v['acc_b'])} | "
                       f"{v['b']} | {v['c']} | {v['p']:.2e} | {v['p_holm']:.2e} | "
                       f"{v['cohens_h']:+.2f} |")
        out.append("")

        out += ["## Table 3 — Scalar-confidence diagnostics (R3)", "",
                "| model | AUROC ACT vs not-ACT | mean conf ACT / ASK / REFUSE / DEFER | "
                "AUROC ASK-REFUSE | AUROC ASK-DEFER | AUROC DEFER-REFUSE | best 4-way router acc |",
                "|---|---|---|---|---|---|---|"]
        for m, d in main_a.get("scalar_diagnostics", {}).items():
            mc = d["mean_conf_by_action"]
            pa = d["pairwise_auroc_deferral"]
            r4 = (d.get("router") or {}).get("four_way", {}).get("acc")
            out.append(f"| {SHORT.get(m, m)} | {f(d['auroc_act_vs_notact'])} | "
                       f"{f(mc['ACT'],1)} / {f(mc['ASK'],1)} / {f(mc['REFUSE'],1)} / {f(mc['DEFER'],1)} | "
                       f"{f(pa.get('ASK_vs_REFUSE'))} | {f(pa.get('ASK_vs_DEFER'))} | "
                       f"{f(pa.get('DEFER_vs_REFUSE'))} | {f(r4)} |")
        out.append("")

        out += ["## Table 4 — Label-mapping sensitivity (test accuracy)", "",
                "| model \\| regime | pre-registered | contested dropped | alt map | v1.1 capability audit |",
                "|---|---|---|---|---|"]
        for k, v in sorted(main_a.get("label_sensitivity", {}).items()):
            out.append(f"| {k.replace('|', ' / ')} | {f(v['acc_preregistered'])} | {f(v['acc_drop_contested'])} | "
                       f"{f(v['acc_altmap_falsepresup_refuse'])} | "
                       f"{f(v['acc_v11_in3_capability_audit'])} |")
        out.append("")

        out += ["## Table 5 — Confusion matrices (rows = gold, cols = ACT/ASK/REFUSE/DEFER/unparsed)", ""]
        for r in sorted(main_a["metrics_test"],
                        key=lambda r: (r["model"], REGIME_ORDER.index(r["regime"]))):
            if r["regime"] not in ("R2_TYPED", "R1_AFFORDANCE", "R4_RECOGNITION"):
                continue
            out.append(f"**{SHORT.get(r['model'], r['model'])} — {r['regime']}**")
            out.append("")
            out.append("| gold | ACT | ASK | REFUSE | DEFER | NA |")
            out.append("|---|---|---|---|---|---|")
            for gi, g in enumerate(ACTIONS):
                out.append("| " + g + " | " + " | ".join(str(x) for x in r["confusion"][gi]) + " |")
            out.append("")

    art = load("artifact_check.json")
    if art:
        out += ["## Table 6 — Baselines and the lexical-shortcut check", "",
                "| baseline | split | acc | macro-F1 | ASK-F1 | over-commit | contrast compl. |",
                "|---|---|---|---|---|---|---|"]
        for k, v in art.items():
            split = "transfer" if k.endswith("transfer") else "test"
            out.append(f"| {k} | {split} | {f(v['accuracy'])} | {f(v['macro_f1'])} | "
                       f"{f(v['ask_f1'])} | {f(v['overcommitment'])} | {f(v['contrast_compliance'])} |")
        out.append("")

    st = load("stakes_analysis.json")
    if st:
        out += ["## Table 7 — Experiment 3: announced stakes", "",
                "| regime | gold | frame | P(ask) | P(act) | P(refuse) | P(defer) | acc | n |",
                "|---|---|---|---|---|---|---|---|---|"]
        for k, v in st["cells"].items():
            rg, gold, fr = k.split("|")
            out.append(f"| {rg} | {gold} | {fr} | {f(v['p_ask'])} | {f(v['p_act'])} | "
                       f"{f(v['p_refuse'])} | {f(v['p_defer'])} | {f(v['accuracy'])} | {v['n']} |")
        out.append("")
        out += ["### Signal-detection decomposition (ASK vs ACT items)", "",
                "| regime | model | frame | hit | false alarm | d' [95% CI] | c [95% CI] |",
                "|---|---|---|---|---|---|---|"]
        for key in [k for k in st if k.startswith("sdt_") and not k.startswith("sdt_delta_")]:
            for k, v in st[key].items():
                m, fr = k.split("|")
                dci = v.get("d_prime_ci", [None, None])
                cci = v.get("criterion_c_ci", [None, None])
                out.append(f"| {key[4:]} | {SHORT.get(m, m)} | {fr} | {f(v['hit_rate'])} | "
                           f"{f(v['false_alarm_rate'])} | {f(v['d_prime'])} [{f(dci[0])}, {f(dci[1])}] | "
                           f"{f(v['criterion_c'])} [{f(cci[0])}, {f(cci[1])}] |")
        out.append("")
        deltas = [k for k in st if k.startswith("sdt_delta_")]
        if deltas:
            out += ["### Paired bootstrap of the CHANGE between frames (HIGH vs NONE)", "",
                    "| regime | delta d' [95% CI] | p | delta c [95% CI] | p |",
                    "|---|---|---|---|---|"]
            for k in sorted(deltas):
                v = st[k]
                out.append(f"| {k[10:]} | {v['delta_d_prime']:+.3f} "
                           f"[{v['delta_d_prime_ci'][0]:+.3f}, {v['delta_d_prime_ci'][1]:+.3f}] | "
                           f"{v['p_two_sided_d_prime']:.3f} | {v['delta_criterion']:+.3f} "
                           f"[{v['delta_criterion_ci'][0]:+.3f}, {v['delta_criterion_ci'][1]:+.3f}] | "
                           f"{v['p_two_sided_criterion']:.4f} |")
            out.append("")
        out += ["### Paired HIGH vs LOW contrasts", "",
                "| contrast | rate LOW | rate HIGH | delta | p | p (Holm) |", "|---|---|---|---|---|---|"]
        for k, v in st["contrasts"].items():
            if v:
                out.append(f"| {k.replace('|', ' / ')} | {f(v['rate_low'])} | {f(v['rate_high'])} | {v['delta']:+.3f} | "
                           f"{v['p']:.2e} | {v.get('p_holm', float('nan')):.2e} |")
        out.append("")
        if "logit_R2" in st:
            out += ["### Logistic regression P(ASK) ~ ambiguous * stakes (item-clustered SE)", "",
                    "| term | coef | p |", "|---|---|---|"]
            for t, c in st["logit_R2"]["params"].items():
                out.append(f"| {t} | {c:+.3f} | {st['logit_R2']['pvalues'][t]:.2e} |")
            out.append("")
        if "in3_importance" in st:
            ii = st["in3_importance"]
            out += ["### Item-level stakes (IN3 annotated importance of the missing detail)", "",
                    f"n={ii['n_items']} ASK items, by importance {ii['n_by_importance']}", "",
                    "| regime | ask-rate imp2 | ask-rate imp3 | OR | Fisher p | item-level MW p |",
                    "|---|---|---|---|---|---|"]
            for rg in ("R2_TYPED", "R1_AFFORDANCE"):
                if rg in ii:
                    v = ii[rg]
                    out.append(f"| {rg} | {f(v['ask_rate_importance2'])} | {f(v['ask_rate_importance3'])} | "
                               f"{f(v['odds_ratio'],2)} | {v['fisher_p']:.3g} | "
                               f"{f(v.get('mannwhitney_p'),3)} |")
            out.append("")

    loc = load("local_analysis.json")
    if loc:
        out += ["## Table 8 — Experiment 4: local open-weight model (Qwen3-4B)", "",
                "| split | condition | acc [95% CI] | macro-F1 | ASK-F1 | over-commit | contrast compl. | typed-deferral |",
                "|---|---|---|---|---|---|---|---|"]
        for split in ("test", "transfer"):
            for k, v in loc.get(split, {}).items():
                ci = v.get("accuracy_ci", [None, None])
                out.append(f"| {split} | {k} | {f(v['accuracy'])} [{f(ci[0])}, {f(ci[1])}] | "
                           f"{f(v['macro_f1'])} | {f(v['ask_f1'])} | {f(v['overcommitment'])} | "
                           f"{f(v['contrast_compliance'])} | {f(v['typed_deferral_acc'])} |")
        out.append("")

    ut = load("utility_analysis.json")
    if ut:
        out += ["## Table 9 — Decision-theoretic value (mean utility per request)", "",
                "K = cost of acting on an item that should have been withheld; "
                f"c_ask={ut['c_ask']}, partial credit={ut['partial_credit']}", "",
                "| policy | " + " | ".join(f"K={k}" for k in ut["K_grid"]) + " |",
                "|---" * (len(ut["K_grid"]) + 1) + "|"]
        for name, vals in ut["utility"].items():
            out.append(f"| {name.replace('|', ' / ')} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
        out.append("")

    bc = load("budget_calibration.json")
    if bc:
        out += ["## Table 10 — Calibration and interaction-budget curves (R3 scalar)", "",
                "| model | ECE | mean stated conf | base rate ACT | " +
                " | ".join(f"ask-recall @{int(b*100)}%" for b in bc["budgets"]) + " |",
                "|---" * (4 + len(bc["budgets"])) + "|"]
        for m, d in bc["models"].items():
            row = " | ".join(f(d["budget_curve_scalar"][f"{b:.2f}"]["ask_recall"], 2)
                             for b in bc["budgets"])
            out.append(f"| {SHORT.get(m, m)} | {f(d['ece_act'])} | {f(d['mean_confidence'])} | "
                       f"{f(d['base_rate_act'])} | {row} |")
        out.append("")
        out += ["Typed (R2) operating points: " + "; ".join(
            f"{SHORT.get(m, m)} ask-rate {d['typed_operating_point']['ask_rate']:.2f} → "
            f"recall {d['typed_operating_point']['ask_recall']:.2f}, "
            f"precision {d['typed_operating_point']['ask_precision']:.2f}"
            for m, d in bc["models"].items() if d.get("typed_operating_point")), ""]

    hy = load("hybrid_analysis.json")
    if hy:
        out += ["## Table 11 — Is the scalar complementary to the typed judgments?", "",
                "| model | policy | acc [95% CI] | typed-deferral | vs parent p (Holm) |",
                "|---|---|---|---|---|"]
        for m, d in hy["models"].items():
            for name, met in d.items():
                ci = met.get("accuracy_ci", [None, None])
                vp = met.get("vs_parent")
                out.append(f"| {SHORT.get(m, m)} | {name} | {f(met['accuracy'])} "
                           f"[{f(ci[0])}, {f(ci[1])}] | {f(met['typed_deferral_acc'])} | "
                           + (f"{vp['p_holm']:.2e} (vs {vp['parent']})" if vp else "—") + " |")
        out.append("")

    rp = load("recognition_properties.json")
    if rp:
        out += ["## Table 12 — R4 property-judgment accuracy (scored where the rule reads it)", "",
                "| gold action \\| property | accuracy | n |", "|---|---|---|"]
        for k, v in sorted(rp["pooled_by_gold"].items()):
            out.append(f"| {k.replace('|', ' → ')} | {f(v['accuracy'])} | {v['n']} |")
        out.append("")

    for name, title in (("judge_cross_agreement.json", "Judge cross-agreement (Qwen3-14B vs Qwen3-4B)"),
                        ("judge_validation.json", "Judge vs. hand annotation")):
        d = load(name)
        if d:
            out += [f"## {title}", "",
                    f"n={d['n']}, raw agreement={f(d['raw_agreement'])}, "
                    f"Cohen's kappa={f(d['cohens_kappa'])}", ""]

    (RES / "SUMMARY.md").write_text("\n".join(out) + "\n")
    print(f"Wrote results/SUMMARY.md ({len(out)} lines)")


if __name__ == "__main__":
    main()
