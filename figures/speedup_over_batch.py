"""Speedup over AR as a function of batch size.

1k-token prefill, 512 decode steps, measured alpha (Ouro 0.92 / Huginn 0.94),
A6000. Solid: WFD/DtV with KV sharing. Dashed: the same schedulers without KV
sharing, drawn up to their last feasible batch, with dotted stubs into one x at
the first OOM batch. The horizontal dashed grey line marks AR parity
(speedup 1). Each panel draws only the series listed in MODELS.

Usage: python figures/speedup_over_batch.py
Output: figures/speedup_over_batch.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

DRAFT_LENGTH = 8

WFD_COLOR = "#1668d9"
DTV_COLOR = plt.cm.Reds(0.65)  # same red as gamma=8 in speedup_over_acceptance

# series key -> (scheduler, kv_budget_s, label, color); declaration order is
# also the legend order
SERIES = {
    "wfd_on": ("wfd", 1, "WFD w/ KV sharing", WFD_COLOR),
    "dtv_on": ("dtv", 1, f"DtV w/ KV sharing ($\\gamma$={DRAFT_LENGTH})", DTV_COLOR),
    "wfd_off": ("wfd", 0, "WFD w/o KV sharing", WFD_COLOR),
    "dtv_off": ("dtv", 0, "DtV w/o KV sharing", DTV_COLOR),
}

MODELS = [
    # (panel title, results json, series keys drawn in this panel)
    ("Ouro-2.6B", "results/exp4/ouro-2.6b_p1024d512_0826_0835.json",
     ["wfd_on", "dtv_on", "wfd_off", "dtv_off"]),
    ("Huginn-3.5B", "results/exp4/huginn-3.5b_p1024d512_0828_0618.json",
     ["wfd_on", "dtv_on", "wfd_off", "dtv_off"]),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def select(results, scheduler, kv_budget_s):
    """Return (batch_sizes, speedups) for one series, sorted by batch size."""
    pts = [
        (e["batch_size"], e["speedup_vs_ar"])
        for e in results
        if e["scheduler"] == scheduler
        and e["kv_budget_s"] == kv_budget_s
        and (scheduler != "dtv" or e.get("draft_length") == DRAFT_LENGTH)
    ]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def first_oom(results, kv_budget_s):
    ooms = sorted(e["batch_size"] for e in results
                  if e["scheduler"] == "OOM" and e["kv_budget_s"] == kv_budget_s)
    return ooms[0] if ooms else None


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    handles = {}

    for ax, (title, path, keys) in zip(axes, MODELS):
        results = json.load(open(ROOT / path))["results"]
        xticks, oom_x, oom_y = set(), None, None

        ax.axhline(1.0, color="0.6", ls="--", lw=1.0, zorder=1)
        # x in axes fraction so the label sticks to the left edge whatever the
        # data limits turn out to be
        ax.annotate("AR", (0.02, 1.0), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 3),
                    ha="left", va="bottom", fontsize=8, color="0.45")

        for key in keys:
            sched, kv, label, color = SERIES[key]
            x, y = select(results, sched, kv)
            style = (dict(ls="-", lw=1.8, mfc=color) if kv
                     else dict(ls="--", lw=1.4, mfc="white"))
            line, = ax.plot(x, y, marker="o", color=color, ms=3.5, label=label,
                            zorder=3, **style)
            handles.setdefault(label, line)
            xticks.update(x)

            # no sharing: dotted stub from the last feasible batch into ONE gray
            # x at the (shared) first OOM batch, placed at the first stub's height
            oom = first_oom(results, kv) if not kv else None
            if oom is not None:
                if oom_y is None:
                    oom_x, oom_y = oom, y[-1]
                ax.plot([x[-1], oom], [y[-1], oom_y], ls=":", color=color,
                        lw=1.0, zorder=2)
                xticks.add(oom)

        if oom_x is not None:
            ax.plot(oom_x, oom_y, marker="x", color="0.35", ms=6, mew=1.8,
                    ls="none", zorder=3)
            ax.annotate("OOM", (oom_x, oom_y), textcoords="offset points",
                        xytext=(7, -3), ha="left", va="center", fontsize=8,
                        color="0.35")

        xticks = sorted(xticks)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks, [str(b) for b in xticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("Batch size")
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Speedup over AR")

    ordered = [SERIES[k][2] for k in SERIES if SERIES[k][2] in handles]
    fig.legend([handles[lb] for lb in ordered], ordered,
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), columnspacing=1.6, handlelength=1.8)

    fig.tight_layout()
    out = ROOT / "figures" / "speedup_over_batch.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
