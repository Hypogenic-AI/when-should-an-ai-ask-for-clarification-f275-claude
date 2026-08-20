"""
Re-check every quantitative claim in REPORT.md against the artefact that produced it.

A report is only as trustworthy as the link between its prose and its JSON.  This module makes
that link executable: each `chk(...)` restates one number from REPORT.md and asserts it against
`results/*.json`.  Run it after any re-analysis; if a claim drifts, this fails loudly instead of
leaving a stale number in the prose.

    python -m carb.verify_report      # exits non-zero if any claim no longer holds
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ["openai/gpt-5", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash",
          "meta-llama/llama-3.3-70b-instruct"]


def load(name):
    return json.loads((RES / name).read_text())


def main() -> int:
    m = load("main_analysis.json")
    loc = load("local_analysis.json")
    st = load("stakes_analysis.json")
    art = load("artifact_check.json")
    ut = load("utility_analysis.json")
    bc = load("budget_calibration.json")
    rp = load("recognition_properties.json")
    jv = load("judge_validation.json")
    jc = load("judge_cross_agreement.json")

    def cell(model, regime, key):
        for r in m["metrics_test"]:
            if r["model"] == model and r["regime"] == regime:
                return r[key]

    checks: list[tuple[str, bool, object]] = []

    def chk(name, cond, val=None):
        checks.append((name, bool(cond), val))

    r2 = [cell(x, "R2_TYPED", "accuracy") for x in MODELS]
    r3 = [cell(x, "R3_SCALAR", "accuracy") for x in MODELS]
    r4a = [cell(x, "R4_RECOGNITION", "accuracy") for x in MODELS]
    r0oc = [cell(x, "R0_DIRECT", "overcommitment") for x in MODELS]
    r2oc = [cell(x, "R2_TYPED", "overcommitment") for x in MODELS]
    r4oc = [cell(x, "R4_RECOGNITION", "overcommitment") for x in MODELS]

    chk("§1/§5.2 typed accuracy 0.60-0.72", round(min(r2), 2) == 0.60 and round(max(r2), 2) == 0.72,
        (min(r2), max(r2)))
    chk("§1 scalar accuracy 0.42-0.45", round(min(r3), 2) == 0.42 and round(max(r3), 2) == 0.45,
        (min(r3), max(r3)))
    chk("§1 recognition accuracy 0.55-0.61", round(min(r4a), 2) == 0.55 and round(max(r4a), 2) == 0.61,
        (min(r4a), max(r4a)))
    chk("§5.1 R0 over-commitment 0.45-0.58", 0.44 < min(r0oc) < 0.46 and 0.57 < max(r0oc) < 0.59,
        (min(r0oc), max(r0oc)))
    chk("§5.1 R2 over-commitment 0.09-0.25", 0.09 <= min(r2oc) < 0.10 and 0.25 <= max(r2oc) < 0.26,
        (min(r2oc), max(r2oc)))
    chk("§5.1 R4 over-commitment 0.11-0.28", 0.11 <= min(r4oc) < 0.12 and 0.28 <= max(r4oc) < 0.29,
        (min(r4oc), max(r4oc)))
    diffs = [a - b for a, b in zip(r2, r3)]
    chk("§5.2 typed-scalar gap 0.158-0.275",
        abs(min(diffs) - 0.158) < 0.002 and abs(max(diffs) - 0.275) < 0.002, (min(diffs), max(diffs)))
    chk("§5.2 all typed-vs-scalar p_holm < 1e-8",
        all(m["mcnemar"][f"E2_typed_vs_scalar|{x}"]["p_holm"] < 1e-8 for x in MODELS))

    d = m["scalar_diagnostics"]
    au = [d[x]["auroc_act_vs_notact"] for x in MODELS]
    dr = [d[x]["pairwise_auroc_deferral"]["DEFER_vs_REFUSE"] for x in MODELS]
    chk("§5.2 AUROC ACT-vs-not 0.79-0.87", 0.78 < min(au) < 0.80 and 0.87 < max(au) < 0.88,
        (min(au), max(au)))
    chk("§5.2 DEFER-vs-REFUSE AUROC 0.44-0.56", 0.43 < min(dr) < 0.45 and 0.54 < max(dr) < 0.57,
        (min(dr), max(dr)))

    pg = rp["pooled_by_gold"]
    chk("§5.1 information_sufficient on ASK = 0.565",
        abs(pg["ASK|information_sufficient"]["accuracy"] - 0.565) < 0.001)
    chk("§5.1 safe_and_determinate on REFUSE = 0.893",
        abs(pg["REFUSE|safe_and_determinate"]["accuracy"] - 0.893) < 0.001)
    chk("§5.1 within_capability on ACT = 0.943",
        abs(pg["ACT|within_capability"]["accuracy"] - 0.943) < 0.001)

    ls = m["label_sensitivity"]
    variants = ["acc_preregistered", "acc_drop_contested", "acc_altmap_falsepresup_refuse",
                "acc_v11_in3_capability_audit"]
    shift = max(abs(v[x] - v["acc_preregistered"]) for v in ls.values() for x in variants)
    chk("§7.4 label-mapping shift never exceeds 0.068", shift <= 0.0685, shift)
    per = {}
    for k, v in ls.items():
        mdl, rg = k.split("|")
        per.setdefault(mdl, {})[rg] = v
    chk("§7.4 typed>scalar and recognition>direct in all 16 model x variant cells",
        all(d["R2_TYPED"][x] > d["R3_SCALAR"][x] and d["R4_RECOGNITION"][x] > d["R0_DIRECT"][x]
            for d in per.values() for x in variants))

    chk("§5.0 TF-IDF 0.702 test / 0.355 transfer",
        abs(art["tfidf_test"]["accuracy"] - 0.702) < 0.001
        and abs(art["tfidf_transfer"]["accuracy"] - 0.355) < 0.001)

    chk("§6 always-ASK utility flat at +0.29",
        all(abs(v - 0.29) < 0.005 for v in ut["utility"]["always_ASK"]))
    chk("§6 LoRA SFT utility row +0.70/+0.68/+0.66/+0.61/+0.40",
        [round(ut["utility"]["qwen3-4b|lora_sft"][i], 2) for i in (1, 3, 4, 5, 7)]
        == [0.70, 0.68, 0.66, 0.61, 0.40])
    chk("§6 both supervised policies stay above always-ASK across the sweep",
        all(a > b for a, b in zip(ut["utility"]["qwen3-4b|lora_sft"], ut["utility"]["always_ASK"]))
        and all(a > b for a, b in zip(ut["utility"]["qwen3-4b|frozen_probe"],
                                      ut["utility"]["always_ASK"])))
    chk("§6 best model typed policy positive at K=5, negative at K=10",
        ut["utility"]["anthropic/claude-sonnet-4.5|R2_TYPED"][4] > 0
        > ut["utility"]["anthropic/claude-sonnet-4.5|R2_TYPED"][5])

    ece = [bc["models"][x]["ece_act"] for x in MODELS]
    chk("§5.2 ECE 0.16-0.21", 0.16 <= min(ece) < 0.17 and 0.21 <= max(ece) < 0.215,
        (min(ece), max(ece)))

    sd, sd1 = st["sdt_R2_TYPED"], st["sdt_R1_AFFORDANCE"]
    chk("§5.3 typed criterion 1.152 -> 0.712",
        abs(sd["pooled|NONE"]["criterion_c"] - 1.152) < 0.002
        and abs(sd["pooled|HIGH"]["criterion_c"] - 0.712) < 0.002)
    dl2, dl1 = st["sdt_delta_R2_TYPED"], st["sdt_delta_R1_AFFORDANCE"]
    chk("§5.3 typed delta-c = -0.44 [-0.69,-0.27], p<0.001",
        abs(dl2["delta_criterion"] + 0.440) < 0.005 and dl2["p_two_sided_criterion"] < 0.001)
    chk("§5.3 typed delta-d' = -0.11, n.s. (p=0.53)",
        abs(dl2["delta_d_prime"] + 0.112) < 0.005 and 0.4 < dl2["p_two_sided_d_prime"] < 0.7)
    chk("§5.3 free-text delta-d' = -0.58, p=0.013",
        abs(dl1["delta_d_prime"] + 0.581) < 0.005 and dl1["p_two_sided_d_prime"] < 0.05)
    chk("§5.3 free-text d' 1.301 -> 0.720",
        abs(sd1["pooled|NONE"]["d_prime"] - 1.301) < 0.002
        and abs(sd1["pooled|HIGH"]["d_prime"] - 0.720) < 0.002)
    chk("§5.3 ambiguous x high interaction ~0, p>0.95",
        abs(st["logit_R2"]["params"]["ambiguous:high"]) < 0.02
        and st["logit_R2"]["pvalues"]["ambiguous:high"] > 0.95)
    chk("§5.3 GPT-5 excluded from E3", "openai/gpt-5" not in {k.split("|")[0] for k in sd})

    chk("§4 judge kappa 0.75 (blind) / 0.78 (cross)",
        abs(jv["cohens_kappa"] - 0.75) < 0.005 and abs(jc["cohens_kappa"] - 0.781) < 0.005)
    chk("§4 cross-judge n = 3768", jc["n"] == 3768)

    lt, ltr = loc["test"], loc["transfer"]
    chk("§5.4 LoRA SFT 0.794 test / 0.470 transfer",
        abs(lt["trained: LoRA SFT (R2 prompt)"]["accuracy"] - 0.794) < 0.002
        and abs(ltr["trained: LoRA SFT (R2 prompt)"]["accuracy"] - 0.470) < 0.002)
    chk("§5.4 frozen probe 0.779 test / 0.465 transfer",
        abs(lt["frozen-representation linear probe"]["accuracy"] - 0.779) < 0.002
        and abs(ltr["frozen-representation linear probe"]["accuracy"] - 0.465) < 0.002)
    chk("§5.4 local typed 0.577 and local R0 over-commitment 0.581",
        abs(lt["prompted: typed ontology (R2)"]["accuracy"] - 0.577) < 0.002
        and abs(lt["behaviour: plain prompt (R0)"]["overcommitment"] - 0.581) < 0.002)
    chk("§5.4 local SFT over-commitment 0.014",
        abs(lt["trained: LoRA SFT (R2 prompt)"]["overcommitment"] - 0.014) < 0.002)
    chk("§5.4 every transfer condition within +/-0.09 of always-ACT (0.50)",
        all(abs(v["accuracy"] - 0.50) <= 0.091 for v in ltr.values()),
        {k: round(v["accuracy"], 3) for k, v in ltr.items()})

    # §5.5 confusion-matrix claims, pooled over the four models
    import numpy as np

    def pooled(reg):
        t = np.zeros((4, 5), int)
        for r in m["metrics_test"]:
            if r["regime"] == reg:
                t += np.array(r["confusion"])
        return t

    r0, r2t = pooled("R0_DIRECT"), pooled("R2_TYPED")
    chk("§5.5 pooled row totals 516/548/448/408 and 1,920 pairs",
        list(r0.sum(1)) == [516, 548, 448, 408] == list(r2t.sum(1)) and r0.sum() == 1920)
    chk("§5.5 ASK answered outright: 390/548 (71%) R0 -> 170/548 (31%) R2",
        r0[1, 0] == 390 and r2t[1, 0] == 170)
    chk("§5.5 contrast compliance 477/516 (92%) -> 393/516 (76%)",
        r0[0, 0] == 477 and r2t[0, 0] == 393)
    chk("§5.5 ASK routed to REFUSE under R2: 144/548 vs 178 correct",
        r2t[1, 2] == 144 and r2t[1, 1] == 178)
    chk("§5.5 DEFER routed to REFUSE under R2: 77/408", r2t[3, 2] == 77)
    chk("§3 test split composition 137 ASK / 129 ACT / 112 REFUSE / 102 DEFER",
        list(r0.sum(1) // 4) == [129, 137, 112, 102])
    chk("§3 uniform-random 0.231 and always-ASK 0.285 baselines",
        abs(art["baseline_uniform_random"]["accuracy"] - 0.231) < 0.001
        and abs(art["baseline_always_ASK"]["accuracy"] - 0.285) < 0.001)

    bad = [c for c in checks if not c[1]]
    for name, ok, val in checks:
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   -> {val}"))
    print(f"\n{len(checks) - len(bad)}/{len(checks)} report claims verified against artefacts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
