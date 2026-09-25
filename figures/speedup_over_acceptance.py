"""Speedup over AR as a function of forced acceptance rate alpha.

Batch 1, 1k-token prefill, 512 decode steps, A6000, KV share off.
One panel per model; the vertical dashed line marks the model's measured alpha
and the horizontal dashed grey line marks AR parity (speedup 1).

Usage: python figures/speedup_over_acceptance.py
Output: figures/speedup_over_acceptance.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    # (panel title, results json, measured alpha)
    ("Ouro-2.6B", "results/exp4/ouro-2.6b_p1024d512_0714_0511.json", 0.92),
    ("Huginn-3.5B", "results/exp4/huginn-3.5b_p1024d512_0714_1744.json", 0.94),
]

DRAFT_LENGTHS = [2, 4, 8, 16, 32]

BATCH_SIZE = 1
KV_BUDGET_S = 0  # KV share off

WFD_COLOR = "#1668d9"
# light -> dark reds for dtv draft_len 2 -> 32
DTV_COLORS = plt.cm.Reds(np.linspace(0.35, 0.95, len(DRAFT_LENGTHS)))

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def select(results, scheduler, draft_length=None):
    """Return (alphas, speedups) for one series, sorted by alpha."""
    pts = [
        (e["alpha"], e["speedup_vs_ar"])
        for e in results
        if e["scheduler"] == scheduler
        and e["batch_size"] == BATCH_SIZE
        and e["kv_budget_s"] == KV_BUDGET_S
        and (draft_length is None or e.get("draft_length") == draft_length)
        and e["alpha"] <= 0.95
    ]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

    for ax, (title, path, measured_alpha) in zip(axes, MODELS):
        results = json.load(open(ROOT / path))["results"]

        ax.axhline(1.0, color="0.6", ls="--", lw=1.0, zorder=1)
        # x in axes fraction so the label sticks to an edge whatever the data
        # limits turn out to be — right edge here, the left one is crowded by
        # the low-alpha ends of the DtV curves
        ax.annotate("AR", (0.98, 1.0), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 3),
                    ha="right", va="bottom", fontsize=8, color="0.45")

        x, y = select(results, "wfd")
        ax.plot(x, y, "-o", color=WFD_COLOR, lw=1.8, ms=3.5, label="WFD", zorder=3)

        for dl, color in zip(DRAFT_LENGTHS, DTV_COLORS):
            x, y = select(results, "dtv", draft_length=dl)
            ax.plot(x, y, "--o", color=color, lw=1.4, ms=3,
                    label=f"DtV ($\gamma$={dl})", zorder=2)

        ax.set_title(title)
        ax.set_xlabel(r"Acceptance rate $\alpha$")
        # pinned: tight_layout shrinks the axes enough that the automatic
        # locator would fall back to 1.0 steps
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Speedup over AR")

    # measured-alpha markers, drawn after all series so the shared ylim is final
    for ax, (_, _, measured_alpha) in zip(axes, MODELS):
        ax.axvline(measured_alpha, color="0.35", ls="--", lw=1.2, zorder=1)
        ax.text(measured_alpha, ax.get_ylim()[1], f" $\\alpha$={measured_alpha}",
                ha="left", va="top", fontsize=8, color="0.35")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06), columnspacing=1.2, handlelength=1.8)

    fig.tight_layout()
    out = ROOT / "figures" / "speedup_over_acceptance.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
