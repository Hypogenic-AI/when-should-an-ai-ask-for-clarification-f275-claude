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
        ax.axhline(0.25, ls=":", c="k", lw=1)
        ax.text(len(models) - 0.5, 0.26, "chance", fontsize=8, ha="right")
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
    """Per-model: what the model *judges* (R4) vs what it *does* (R0, R1)."""
    rows = rep["metrics_test"]
    models = [m for m in SHORT if any(r["model"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(models))
    for k, rg in enumerate(["R0_DIRECT", "R1_AFFORDANCE", "R4_RECOGNITION"]):
        vals = [next((r["accuracy"] for r in rows if r["model"] == m and r["regime"] == rg), np.nan)
                for m in models]
        ax.bar(x + (k - 1) * 0.26, vals, 0.26, color=PALETTE[rg],
               label=REGIME_LABEL[rg].replace("\n", " "))
        for xi, v in zip(x + (k - 1) * 0.26, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=9)
    ax.set_ylabel("4-way routing accuracy")
    ax.set_title("Recognition-behaviour gap: what the model judges vs. what it does", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
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
        means = [diag[m]["mean_conf_by_action"].get(a) or np.nan for m in models]
        stds = [diag[m]["std_conf_by_action"].get(a) or 0 for m in models]
        ax.bar(x + (k - 1.5) * w, means, w, yerr=stds, capsize=2, color=colors[a], label=a, ecolor="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=8.5)
    ax.set_ylabel("stated confidence that it can act now (0-100)")
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
        ax.axvline(0.25, ls=":", c="k", lw=1)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("4-way routing accuracy")
        ax.set_title(f"{split} split", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Experiment 4: prompting vs. training vs. frozen-representation probe (Qwen3-4B)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig7_local_training.png", dpi=170)
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
    print("Figures written to figures/")


if __name__ == "__main__":
    main()
