"""Figures for the report.  All plots read from results/*.json produced by the analysis scripts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"
ACTIONS = ["ACT", "ASK", "REFUSE", "DEFER"]
REGIME_ORDER = ["R0_DIRECT", "R1_AFFORDANCE", "R2_TYPED", "R3_SCALAR", "R4_RECOGNITION"]
REGIME_LABEL = {"R0_DIRECT": "R0 direct\n(default behaviour)",
                "R1_AFFORDANCE": "R1 affordance\n(options named)",
                "R2_TYPED": "R2 typed ontology\n(structured)",
                "R3_SCALAR": "R3 scalar confidence\n(tuned thresholds)",
                "R4_RECOGNITION": "R4 recognition\n(properties + fixed rule)"}
SHORT = {"openai/gpt-5": "GPT-5", "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
         "google/gemini-2.5-flash": "Gemini 2.5 Flash",
         "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B"}
PALETTE = {"R0_DIRECT": "#B0B7C3", "R1_AFFORDANCE": "#7C93B8", "R2_TYPED": "#2D6A9F",
           "R3_SCALAR": "#E08B4F", "R4_RECOGNITION": "#5C9E76"}


def _load() -> dict:
    return json.loads((RES / "main_analysis.json").read_text())


def fig_regime_accuracy(rep: dict) -> None:
    rows = rep["metrics_test"]
    models = [m for m in SHORT if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, title, ylab in (
        (axes[0], "accuracy", "4-way routing accuracy", "accuracy"),
        (axes[1], "typed_deferral_acc", "Typed deferral accuracy\n(right KIND of deferral, given deferral)", "accuracy"),
    ):
        w = 0.16
        x = np.arange(len(models))
        for k, rg in enumerate(REGIME_ORDER):
            vals, los, his = [], [], []
            for m in models:
                r = next((r for r in rows if r["model"] == m and r["regime"] == rg), None)
                v = r[metric] if r else np.nan
                ci = r.get(f"{metric}_ci", [np.nan, np.nan]) if r else [np.nan, np.nan]
                vals.append(v)
                los.append(max(0, v - ci[0]) if r else 0)
                his.append(max(0, ci[1] - v) if r else 0)
            ax.bar(x + (k - 2) * w, vals, w, yerr=[los, his], capsize=2,
                   label=REGIME_LABEL[rg].replace("\n", " "), color=PALETTE[rg], ecolor="#444")
        # references differ per metric: 4-way accuracy has uniform-random 0.231 and the best
        # single-action policy 0.285; typed-deferral accuracy has always-ASK at 0.390
        if metric == "accuracy":
            ax.axhline(0.231, ls=":", c="k", lw=1)
            ax.axhline(0.285, ls="--", c="#8A6D3B", lw=1)
            ax.text(len(models) - 0.5, 0.185, "uniform random (0.23)", fontsize=7.5, ha="right")
            ax.text(len(models) - 0.5, 0.295, "best single action, always-ASK (0.29)",
                    fontsize=7.5, ha="right", color="#8A6D3B")
        else:
            ax.axhline(0.390, ls="--", c="#8A6D3B", lw=1)
            ax.text(len(models) - 0.5, 0.40, "always-ASK (0.39)", fontsize=7.5, ha="right",
                    color="#8A6D3B")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_regime_accuracy.png", dpi=170)
    plt.close(fig)


def fig_recognition_behaviour_gap(rep: dict) -> None:
    """Per-model: what the model *judges* (R4) vs what it *does* (R0, R1).

    Left panel is 4-way accuracy; right panel is over-commitment, the quantity the gap is
    really about -- how often the model went ahead and acted on a request it should have
    withheld action on.
    """
    rows = rep["metrics_test"]
    models = [m for m in SHORT if any(r["model"] == m for r in rows)]
    regimes = ["R0_DIRECT", "R1_AFFORDANCE", "R4_RECOGNITION", "R2_TYPED"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    for ax, metric, ylab, title in (
            (axes[0], "accuracy", "4-way routing accuracy",
             "What it judges vs. what it does"),
            (axes[1], "overcommitment", "P(acted | should not have acted)",
             "Over-commitment: acting anyway")):
        x = np.arange(len(models))
        for k, rg in enumerate(regimes):
            vals = [next((r[metric] for r in rows if r["model"] == m and r["regime"] == rg), np.nan)
                    for m in models]
            ax.bar(x + (k - 1.5) * 0.2, vals, 0.2, color=PALETTE[rg],
                   label=REGIME_LABEL[rg].replace("\n", " "))
            for xi, v in zip(x + (k - 1.5) * 0.2, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.suptitle("Experiment 1: recognition vs. behaviour on identical items", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig2_recognition_behaviour_gap.png", dpi=170)
    plt.close(fig)


def fig_confusions(rep: dict) -> None:
    rows = rep["metrics_test"]
    models = [m for m in SHORT if any(r["model"] == m for r in rows)]
    regs = ["R0_DIRECT", "R2_TYPED", "R3_SCALAR", "R4_RECOGNITION"]
    fig, axes = plt.subplots(len(models), len(regs), figsize=(3.0 * len(regs), 2.9 * len(models)))
    for i, m in enumerate(models):
        for j, rg in enumerate(regs):
            ax = axes[i, j]
            r = next((r for r in rows if r["model"] == m and r["regime"] == rg), None)
            if r is None:
                ax.axis("off")
                continue
            cm = np.array(r["confusion"], dtype=float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for a in range(4):
                for b in range(5):
                    if cmn[a, b] > 0.005:
                        ax.text(b, a, f"{cmn[a,b]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if cmn[a, b] > 0.55 else "black")
            ax.set_xticks(range(5))
            ax.set_xticklabels(ACTIONS + ["n/a"], fontsize=7, rotation=45)
            ax.set_yticks(range(4))
            ax.set_yticklabels(ACTIONS, fontsize=7)
            if i == 0:
                ax.set_title(REGIME_LABEL[rg].split("\n")[0], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{SHORT[m]}\ngold", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("predicted", fontsize=8)
    fig.suptitle("Row-normalised confusion matrices (test split, n=480)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig3_confusion_matrices.png", dpi=160)
    plt.close(fig)


def fig_scalar_diagnostics(rep: dict) -> None:
    diag = rep["scalar_diagnostics"]
    models = [m for m in SHORT if m in diag]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    ax = axes[0]
    w = 0.2
    x = np.arange(len(models))
    colors = {"ACT": "#5C9E76", "ASK": "#2D6A9F", "REFUSE": "#C0504D", "DEFER": "#E08B4F"}
    for k, a in enumerate(ACTIONS):
        means = np.array([diag[m]["mean_conf_by_action"].get(a) or np.nan for m in models])
        stds = np.array([diag[m]["std_conf_by_action"].get(a) or 0 for m in models])
        # confidence is bounded [0, 100]; clip the +/-1 SD whiskers so they stay on the scale
        lo = means - np.clip(means - stds, 0, 100)
        hi = np.clip(means + stds, 0, 100) - means
        ax.bar(x + (k - 1.5) * w, means, w, yerr=[lo, hi], capsize=2, color=colors[a], label=a,
               ecolor="#555")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=8.5)
    ax.set_ylabel("stated confidence that it can act now (0-100)\nbars = mean, whiskers = +/-1 SD clipped to the scale")
    ax.set_title("Scalar confidence separates ACT from the rest,\nbut not the deferral types from each other", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    keys = ["ASK_vs_DEFER", "ASK_vs_REFUSE", "DEFER_vs_REFUSE"]
    x = np.arange(len(models))
    ax.bar(x - 0.3, [diag[m]["auroc_act_vs_notact"] for m in models], 0.22,
           color="#2D6A9F", label="ACT vs. not-ACT")
    for k, key in enumerate(keys):
        vals = [abs(diag[m]["pairwise_auroc_deferral"].get(key, np.nan) - 0.5) + 0.5 for m in models]
        ax.bar(x + (k - 0.5) * 0.22, vals, 0.22, label=key.replace("_", " "),
               color=["#E08B4F", "#C0504D", "#9C7BB5"][k])
    ax.axhline(0.5, ls=":", c="k", lw=1)
    ax.text(len(models) - 0.5, 0.515, "chance", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=8.5)
    ax.set_ylabel("AUROC (deferral pairs shown as |AUROC-0.5|+0.5)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Discriminative power of the single scalar", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_scalar_diagnostics.png", dpi=170)
    plt.close(fig)


def fig_askf1_vs_contrast(rep: dict) -> None:
    """The two metrics that degenerate policies split: asking well vs. not over-refusing."""
    rows = rep["metrics_test"]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    marker = {"openai/gpt-5": "o", "anthropic/claude-sonnet-4.5": "s",
              "google/gemini-2.5-flash": "^", "meta-llama/llama-3.3-70b-instruct": "D"}
    for r in rows:
        if r["model"] not in marker:
            continue
        ax.scatter(r["contrast_compliance"], r["ask_f1"], s=95, marker=marker[r["model"]],
                   color=PALETTE[r["regime"]], edgecolor="k", linewidth=0.6, zorder=3)
    art = json.loads((RES / "artifact_check.json").read_text())
    for name, mk in (("always_ACT", "x"), ("always_ASK", "+")):
        b = art[f"baseline_{name}"]
        ax.scatter(b["contrast_compliance"], b["ask_f1"], s=140, marker=mk, color="k", zorder=4)
        ax.annotate(name, (b["contrast_compliance"], b["ask_f1"]), fontsize=8,
                    xytext=(4, 5), textcoords="offset points")
    t = art["tfidf_test"]
    ax.scatter(t["contrast_compliance"], t["ask_f1"], s=120, marker="*", color="#666", zorder=4)
    ax.annotate("TF-IDF lexical baseline", (t["contrast_compliance"], t["ask_f1"]), fontsize=8,
                xytext=(4, -11), textcoords="offset points")

    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[rg], markeredgecolor="k",
                          label=REGIME_LABEL[rg].replace("\n", " ")) for rg in REGIME_ORDER]
    handles += [plt.Line2D([], [], marker=marker[m], ls="", color="w", markeredgecolor="k",
                           label=SHORT[m]) for m in marker]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left", ncol=1, framealpha=0.92)
    ax.set_xlabel("contrast-set compliance  (P(act) on genuinely answerable items)")
    ax.set_ylabel("ASK-F1")
    ax.set_title("Asking well vs. not over-refusing\n(degenerate policies sit in the corners)", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_askf1_vs_contrast.png", dpi=170)
    plt.close(fig)


def fig_stakes() -> None:
    p = RES / "stakes_analysis.json"
    if not p.exists():
        return
    st = json.loads(p.read_text())
    cells = st["cells"]
    regimes = sorted({k.split("|")[0] for k in cells})
    frames = ["LOW", "NONE", "HIGH"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(6.0 * len(regimes), 4.4), squeeze=False)
    for ai, rg in enumerate(regimes):
        ax = axes[0][ai]
        x = np.arange(len(ACTIONS))
        for k, fr in enumerate(frames):
            vals = [cells.get(f"{rg}|{g}|{fr}", {}).get("p_ask", np.nan) for g in ACTIONS]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=f"stakes: {fr}",
                   color=["#9EC6E0", "#B0B7C3", "#C0504D"][k])
        ax.set_xticks(x)
        ax.set_xticklabels(ACTIONS)
        ax.set_xlabel("gold action of the item")
        ax.set_ylabel("P(model asks a clarifying question)")
        ax.set_title(f"{rg}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1)
    fig.suptitle("Experiment 3: announced stakes vs. ask-rate (within-item, pooled over 3 models)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "fig6_stakes.png", dpi=170)
    plt.close(fig)


def fig_local(rep: dict) -> None:
    p = RES / "local_analysis.json"
    if not p.exists():
        return
    loc = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, split in zip(axes, ["test", "transfer"]):
        names, vals = [], []
        for k, v in loc.get(split, {}).items():
            names.append(k)
            vals.append(v["accuracy"])
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order], color="#2D6A9F")
        for i, oi in enumerate(order):
            ax.text(vals[oi] + 0.008, i, f"{vals[oi]:.3f}", va="center", fontsize=8)
        # the transfer split is CLAMBER, which is binary ACT/ASK at 100/100 -- so the
        # meaningful reference there is the 0.50 always-ACT policy, not 4-way chance
        ax.axvline(0.231, ls=":", c="k", lw=1)
        if split == "transfer":
            ax.axvline(0.50, ls="--", c="#C0504D", lw=1.2)
            ax.text(0.505, -0.6, "always-ACT (0.50)", fontsize=8, color="#C0504D")
        else:
            ax.text(0.236, -0.6, "uniform random (0.23)", fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("4-way routing accuracy")
        ax.set_title(f"{split} split" + (" — CLAMBER, binary ACT/ASK" if split == "transfer" else ""),
                     fontsize=11)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Experiment 4: prompting vs. training vs. frozen-representation probe (Qwen3-4B)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig7_local_training.png", dpi=170)
    plt.close(fig)


def fig_utility() -> None:
    """Mean utility per item as the price of an unwarranted action rises."""
    p = RES / "utility_analysis.json"
    if not p.exists():
        return
    u = json.loads(p.read_text())
    K = u["K_grid"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    styles = {"oracle": ("k", "-", 2.0), "always_ASK": ("#7a7a7a", "--", 1.8),
              "always_ACT": ("#C0504D", "--", 1.8), "tfidf_lexical": ("#8E6FB0", ":", 1.8),
              "qwen3-4b|frozen_probe": ("#5C9E76", "-.", 1.8),
              "qwen3-4b|lora_sft": ("#2E7D4F", (0, (3, 1, 1, 1)), 1.8)}
    for name, vals in u["utility"].items():
        if name in styles:
            c, ls, lw = styles[name]
            ax.plot(K, vals, linestyle=ls, color=c, lw=lw, label=name.replace("|", " "))
        elif name.endswith("R2_TYPED"):
            m = name.split("|")[0]
            ax.plot(K, vals, "-", lw=1.4, alpha=0.9,
                    label=f"{SHORT.get(m, m)} (typed R2)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_ylim(-4, 1.1)
    ax.set_xlabel("K = cost of acting on an item that should have been withheld\n"
                  "(1 = as bad as a needless refusal; 30 = catastrophic)")
    ax.set_ylabel("mean utility per request")
    ax.set_title("Decision-theoretic value of each routing policy as stakes rise", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_utility_vs_stakes.png", dpi=170)
    plt.close(fig)


def fig_budget_calibration() -> None:
    """Left: reliability of the verbalised confidence.  Right: ask-recall under a budget."""
    p = RES / "budget_calibration.json"
    if not p.exists():
        return
    b = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
    for m, d in b["models"].items():
        bins = [t for t in d["reliability"] if t["n"]]
        xs = [t["mean_confidence"] for t in bins]
        ys = [t["empirical_rate"] for t in bins]
        ns = [t["n"] for t in bins]
        (ln,) = ax.plot(xs, ys, "-", lw=1.3, label=f"{SHORT.get(m, m)} (ECE {d['ece_act']:.2f})")
        # marker area proportional to bin count: the jagged mid-range bins hold very few items
        ax.scatter(xs, ys, s=[6 + 0.6 * n for n in ns], color=ln.get_color(), alpha=0.75,
                   edgecolors="none")
    ax.set_xlabel("stated confidence that it can act now\n(marker area = items in the bin)")
    ax.set_ylabel("P(the item really was ACT)")
    ax.set_title("Is the verbalised confidence calibrated?", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    budgets = b["budgets"]
    colours = {}
    for m, d in b["models"].items():
        ys = [d["budget_curve_scalar"][f"{x:.2f}"]["ask_recall"] for x in budgets]
        (ln,) = ax.plot([x * 100 for x in budgets], ys, "o-", ms=4, lw=1.4, label=SHORT.get(m, m))
        colours[m] = ln.get_color()
    rnd = [b["models"][list(b["models"])[0]]["budget_curve_random"][f"{x:.2f}"]["ask_recall"]
           for x in budgets]
    ax.plot([x * 100 for x in budgets], rnd, "k:", lw=1.2, label="random ranking")
    for m, d in b["models"].items():
        op = d.get("typed_operating_point")
        if op:
            ax.plot(op["ask_rate"] * 100, op["ask_recall"], "*", ms=14,
                    color=colours.get(m, "k"), mec="k", mew=0.7)
    ax.set_xlabel("interaction budget: % of requests allowed to become a question")
    ax.set_ylabel("ask-recall (of items that needed a question)")
    ax.set_title("Spending a fixed asking budget\n(stars = where the typed policy sits)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig9_budget_calibration.png", dpi=170)
    plt.close(fig)


def fig_stakes_sdt() -> None:
    """Signal-detection view: does announced stakes move the threshold or the discrimination?"""
    p = RES / "stakes_analysis.json"
    if not p.exists():
        return
    st = json.loads(p.read_text())
    sd = st.get("sdt_R2_TYPED")
    if not sd:
        return
    frames = ["NONE", "LOW", "HIGH"]
    models = sorted({k.split("|")[0] for k in sd})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, title, ylab in (
            (axes[0], "d_prime", "Discrimination d'\n(can it tell ask-worthy from answerable?)", "d'"),
            (axes[1], "criterion_c", "Criterion c\n(how reluctant is it to ask?)", "c")):
        for m in models:
            ys = [sd.get(f"{m}|{fr}", {}).get(key, np.nan) for fr in frames]
            cis = [sd.get(f"{m}|{fr}", {}).get(f"{key}_ci") for fr in frames]
            err = None
            if all(cis):
                err = [[y - c[0] for y, c in zip(ys, cis)], [c[1] - y for y, c in zip(ys, cis)]]
            ax.errorbar(frames, ys, yerr=err, fmt="o-", capsize=3,
                        lw=2 if m == "pooled" else 1.3,
                        color="k" if m == "pooled" else None,
                        ms=6 if m == "pooled" else 4,
                        label=SHORT.get(m, m))
        # y-axis anchored at 0 so a small change is not visually inflated
        ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))
        ax.set_ylabel(ylab)
        ax.set_xlabel("announced stakes")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Experiment 3: announced stakes move the asking threshold, not the discrimination",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / "fig10_stakes_sdt.png", dpi=170)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rep = _load()
    fig_regime_accuracy(rep)
    fig_recognition_behaviour_gap(rep)
    fig_confusions(rep)
    fig_scalar_diagnostics(rep)
    fig_askf1_vs_contrast(rep)
    fig_stakes()
    fig_local(rep)
    fig_utility()
    fig_budget_calibration()
    fig_stakes_sdt()
    print("Figures written to figures/")


if __name__ == "__main__":
    main()
