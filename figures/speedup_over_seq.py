"""Speedup over AR as a function of prefill sequence length.

Batch 1, 512 decode steps, measured alpha (Ouro 0.92 / Huginn 0.94), A6000.
Solid: WFD/DtV with KV sharing. Dashed: the same schedulers without KV
sharing, drawn up to their last feasible length, with an x at the first
OOM point. The dashed grey line marks AR parity (speedup 1).

Usage: python figures/speedup_over_seq.py
Output: figures/speedup_over_seq.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    # (panel title, kv-share-on json, kv-share-off json)
    # 0915_13xx/14xx runs: pad_q1 kernel-parity routing — single-query reads padded to the causal tile so
    # AR / per-token / grouped reads share the same per-byte cost. Earlier
    # same-day runs (1137/1200/1133/1209) are the pre-parity measurements.
    ("Ouro-2.6B",
     "results/exp4/ouro-2.6b_p1024-2048-4096-8192-16384-32768-65536-131072d512_0915_1433.json",
     "results/exp4/ouro-2.6b_p1024-2048-4096-8192-16384-32768-65536-131072d512_0915_1355.json"),
    ("Huginn-3.5B",
     "results/exp4/huginn-3.5b_p1024-2048-4096-8192-16384-32768-65536-131072d512_0915_1446.json",
     "results/exp4/huginn-3.5b_p1024-2048-4096-8192-16384-32768-65536-131072d512_0915_1353.json"),
]

DRAFT_LENGTH = 8
BATCH_SIZE = 1

WFD_COLOR = "#1668d9"
DTV_COLOR = plt.cm.Reds(0.65)  # same red as gamma=8 in speedup_over_acceptance

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def select(results, scheduler, draft_length=None):
    """Return (prefill_lens, speedups) for one series, sorted by prefill_len."""
    pts = [
        (e["prefill_len"], e["speedup_decode_vs_ar"])
        for e in results
        if e["scheduler"] == scheduler
        and e["batch_size"] == BATCH_SIZE
        and (draft_length is None or e.get("draft_length") == draft_length)
    ]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def first_oom(results):
    ooms = sorted(e["prefill_len"] for e in results if e["scheduler"] == "OOM")
    return ooms[0] if ooms else None


def klabel(n):
    return f"{n // 1024}k"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

    for ax, (title, kv1_path, kv0_path) in zip(axes, MODELS):
        kv1 = json.load(open(ROOT / kv1_path))["results"]
        kv0 = json.load(open(ROOT / kv0_path))["results"]

        ax.axhline(1.0, color="0.6", ls="--", lw=1.0, zorder=1)
        # x in axes fraction so the label sticks to the left edge whatever the
        # data limits turn out to be
        ax.annotate("AR", (0.02, 1.0), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 3),
                    ha="left", va="bottom", fontsize=8, color="0.45")

        x, y = select(kv1, "wfd")
        ax.plot(x, y, "-o", color=WFD_COLOR, lw=1.8, ms=3.5, label="WFD w/ KV sharing", zorder=3)
        xticks = x

        x, y = select(kv1, "dtv", draft_length=DRAFT_LENGTH)
        ax.plot(x, y, "-o", color=DTV_COLOR, lw=1.8, ms=3.5,
                label=f"DtV w/ KV sharing ($\\gamma$={DRAFT_LENGTH})", zorder=2)

        # WFD/DtV without KV sharing: drawn up to the last feasible length,
        # then dotted stubs from both last points into ONE gray x at the
        # (shared) first OOM point, placed at the WFD stub's height
        oom = first_oom(kv0)
        oom_y = None
        for sched, dl, color, label in [
            ("wfd", None, WFD_COLOR, "WFD w/o KV sharing"),
            ("dtv", DRAFT_LENGTH, DTV_COLOR, "DtV w/o KV sharing"),
        ]:
            x, y = select(kv0, sched, draft_length=dl)
            ax.plot(x, y, "--o", color=color, lw=1.4, ms=3.5, mfc="white",
                    label=label, zorder=3)
            if oom is not None:
                if oom_y is None:
                    oom_y = y[-1]
                ax.plot([x[-1], oom], [y[-1], oom_y], ls=":", color=color,
                        lw=1.0, zorder=2)
        if oom is not None and oom_y is not None:
            ax.plot(oom, oom_y, marker="x", color="0.35", ms=6, mew=1.8,
                    ls="none", zorder=3)
            ax.annotate("OOM", (oom, oom_y), textcoords="offset points",
                        xytext=(7, -3), ha="left", va="center", fontsize=8,
                        color="0.35")

        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks, [klabel(n) for n in xticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("Prefill length (tokens)")
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Decoding speedup over AR")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), columnspacing=1.6, handlelength=1.8)

    fig.tight_layout()
    out = ROOT / "figures" / "speedup_over_seq.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
