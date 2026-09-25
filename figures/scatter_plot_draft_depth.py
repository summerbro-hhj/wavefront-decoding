"""Acceptance and Speed up over AR as a function of draft depth T_d.

Batch 1, 512 decode steps, A6000. One point per T_d: x is the measured
acceptance rate, y is the AR-baseline walltime divided by the run's walltime
(every run emits the same token count, so the ratio is a throughput speedup).
Points come from Default_points plus every results/exp3 file matching
Search_points.

Usage: python figures/scatter_plot_draft_depth.py
Output: figures/scatter_plot_draft_depth.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIR = ROOT / "results" / "exp3"

MODELS = [
    ("Huginn-3.5B",),
]

Baseline_walltime = [
    ("Huginn-3.5B", 129992.94320833462)
]

"""
Default_points = list of (Model, T_d, walltime, acceptance rate)
"""
Default_points = [
    ("Huginn-3.5B", 4, 51144.635588467005,0.8806128701545528)
]

"""
Search
"""
Search_points = ["huginn-3.5b_gsm8k_T32_d*_s0_tauoff_epsoff_ssd_n0.json",]

# per-T_d label placement: (dx pt, dy pt, ha) — keeps text off neighbouring
# markers in the crowded high-acceptance corner
LABEL_OFFSETS = {
    4: (-8, 0, "right"),
    6: (0, 10, "center"),
    8: (0, -13, "center"),
    16: (-8, 0, "right"),
}
DEFAULT_OFFSET = (8, -1, "left")

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def collect(model):
    """Return [(T_d, acceptance, speedup)] for one model, sorted by T_d."""
    base = dict(Baseline_walltime)[model]
    pts = {}
    for m, td, walltime, acceptance in Default_points:
        if m == model:
            pts[td] = (acceptance, base / walltime)
    for pattern in Search_points:
        for path in sorted(SEARCH_DIR.glob(pattern)):
            d = json.load(open(path))
            cfg, agg = d["config"], d["aggregate"]
            if cfg["model"].lower() != model.lower():
                continue
            pts[cfg["T_draft"]] = (agg["acceptance_rate"],
                                   base / agg["ssd_walltime_s"])
    return sorted((td, acc, sp) for td, (acc, sp) in pts.items())


def color_for(td, all_tds):
    """Sequential blue, light -> dark with T_d. Keyed on the value (not the
    rank) so a point's colour does not move when another point is added."""
    lo, hi = np.log2(min(all_tds)), np.log2(max(all_tds))
    t = (np.log2(td) - lo) / (hi - lo) if hi > lo else 0.5
    return plt.cm.Blues(0.40 + 0.55 * t)


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.6, 3.0), squeeze=False)

    for ax, (title,) in zip(axes[0], MODELS):
        points = collect(title)
        tds = [p[0] for p in points]

        ax.axhline(1.0, color="0.6", ls="--", lw=1.0, zorder=1)
        # x in axes fraction so the label sticks to the left edge whatever the
        # data limits turn out to be
        ax.annotate("AR", (0.02, 1.0), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 3),
                    ha="left", va="bottom", fontsize=8, color="0.45")

        for td, acc, speedup in points:
            ax.plot(acc, speedup, marker="o", ms=6, ls="none",
                    color=color_for(td, tds), mec="white", mew=0.8, zorder=3)
            dx, dy, ha = LABEL_OFFSETS.get(td, DEFAULT_OFFSET)
            ax.annotate(f"$T_d$ = {td}", (acc, speedup),
                        textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va="center", fontsize=8, color="0.25")

        ax.set_title(title)
        ax.set_xlabel("Acceptance rate")
        ax.set_ylabel("Speedup over AR")
        ax.margins(x=0.16, y=0.14)
        ax.grid(color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = ROOT / "figures" / "scatter_plot_draft_depth.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    for ax, (title,) in zip(axes[0], MODELS):
        for td, acc, speedup in collect(title):
            print(f"  {title}  T_d={td:>2}  acc={acc:.4f}  speedup={speedup:.3f}")


if __name__ == "__main__":
    main()
