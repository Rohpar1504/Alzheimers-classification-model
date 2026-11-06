#!/usr/bin/env python3
"""
p/q investigation - Item 3: scatter plot of each subject's (p, q), AD vs CN.

Covers the advisor to-do items:
    1. "Plot p and q values for subjects with AD and subjects without AD - for node2vec"
    4. "Replicate the scatter plot for with AD and without AD"

Reads each subject's winning (p, q) from Item 2 (data/pq_subject_fit/subject_pq.csv,
found by link prediction, labels never used) and plots them on the 5 x 5 p/q grid,
one panel per connectivity measure, AD and CN side by side in every grid cell.
Plot only - the AD vs CN tests are part of Item 4.

Paper-style figure (Rougier et al., "Ten Simple Rules for Better Figures", 2014;
Frontiers figure requirements): one row of panels A-C, text >= 8 pt at print size,
18 cm wide, colour-blind-safe Okabe-Ito colours plus filled (AD) vs hollow (CN)
markers so it also reads in black and white, no numbers or statistics on the figure,
explained by a caption (printed when the script runs).

Outputs:
    outputs/figures/pq_study/
        "Plot p and q values for subjects with AD and subjects without AD - for node2vec".png/.pdf
        (one figure covers both to-do items 1 and 4)
    outputs/results/pq_subject_ad_cn_counts.csv
        number of AD and CN subjects at each (p, q), per measure - no subject IDs

Usage:
    .venv/bin/python src/pq_ad_cn_scatter.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_grid import P_VALUES, Q_VALUES

PROJECT_ROOT = SRC_DIR.parent
SUBJECT_PQ = PROJECT_ROOT / "data" / "pq_subject_fit" / "subject_pq.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "pq_study"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

TITLE = "Plot p and q values for subjects with AD and subjects without AD - for node2vec"
FIG_NAMES = [TITLE]

MEASURES = [("pearson", "Pearson"), ("spearman", "Spearman"), ("kendall", "Kendall")]
# Okabe-Ito colour-blind-safe pair; AD filled, CN hollow (second, non-colour cue).
GROUPS = {"AD": {"color": "#D55E00", "filled": True,  "dx": -0.17},
          "CN": {"color": "#0072B2", "filled": False, "dx": +0.17}}
SIZE_PER_SUBJECT = 26       # marker area (pt^2) per subject -> area proportional to count
LEGEND_SIZES = [1, 3, 6]
TICK_LABELS = ["0.25", "0.5", "1", "2", "4"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "pdf.fonttype": 42})


def count_table(best: pd.DataFrame) -> pd.DataFrame:
    """Number of AD and CN subjects at every (p, q) cell, per measure (zeros included)."""
    rows = []
    for measure, _ in MEASURES:
        sub = best[best.measure == measure]
        for p in P_VALUES:
            for q in Q_VALUES:
                cell = sub[(sub.best_p == p) & (sub.best_q == q)]
                rows.append({"measure": measure, "p": p, "q": q,
                             "AD": int((cell.group == "AD").sum()),
                             "CN": int((cell.group == "CN").sum())})
    return pd.DataFrame(rows)


def marker_style(group: str) -> dict:
    g = GROUPS[group]
    if g["filled"]:
        return {"facecolors": g["color"], "edgecolors": "white", "linewidths": 0.6}
    return {"facecolors": "white", "edgecolors": g["color"], "linewidths": 1.3}


def main() -> None:
    best = pd.read_csv(SUBJECT_PQ)
    counts = count_table(best)
    n = best[best.measure == MEASURES[0][0]].group.value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.25), sharex=True, sharey=True)
    for k, (ax, (measure, label)) in enumerate(zip(axes, MEASURES)):
        sub = counts[counts.measure == measure]
        ax.axvline(Q_VALUES.index(1.0), color="0.6", lw=0.7, ls="--", zorder=0)
        ax.axhline(P_VALUES.index(1.0), color="0.6", lw=0.7, ls="--", zorder=0)
        for group, g in GROUPS.items():
            cells = sub[sub[group] > 0]
            ax.scatter([Q_VALUES.index(q) + g["dx"] for q in cells.q],
                       [P_VALUES.index(p) for p in cells.p],
                       s=cells[group] * SIZE_PER_SUBJECT, zorder=3, **marker_style(group))
        ax.set_xticks(range(5), TICK_LABELS)
        ax.set_yticks(range(5), TICK_LABELS)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-0.6, 4.6)
        ax.set_aspect("equal")
        ax.grid(color="0.92", lw=0.6, zorder=0)
        ax.tick_params(length=2.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("q")
        ax.set_title(label)
        ax.text(-0.02, 1.04, "ABC"[k], transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="right", va="bottom")
    axes[0].set_ylabel("p")

    group_handles = [
        plt.scatter([], [], s=3 * SIZE_PER_SUBJECT, label=f"{grp} (n = {n.get(grp, 0)})",
                    **marker_style(grp)) for grp in GROUPS]
    size_handles = [
        plt.scatter([], [], s=c * SIZE_PER_SUBJECT, facecolors="0.55", edgecolors="white",
                    label=f"{c} subject{'s' if c > 1 else ''}") for c in LEGEND_SIZES]
    spacer = Line2D([], [], color="none", label=" ")
    fig.legend(handles=group_handles + [spacer] + size_handles, loc="lower center",
               ncol=6, frameon=False, handletextpad=0.3, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(TITLE, fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95), w_pad=1.5)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in FIG_NAMES:
        fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
        fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "pq_subject_ad_cn_counts.csv"
    counts.to_csv(out_csv, index=False)

    print("Item 3: AD vs CN scatter of each subject's (p, q)")
    for measure, label in MEASURES:
        sub = counts[(counts.measure == measure) & ((counts.AD > 0) | (counts.CN > 0))]
        cells = ", ".join(f"({p:g},{q:g}) AD {a} / CN {c}"
                          for p, q, a, c in sub[["p", "q", "AD", "CN"]].itertuples(index=False))
        print(f"  {label:8s} {cells}")
    print("\nSuggested caption:")
    print(f"  Node2vec walk parameters (p, q) for subjects with AD (filled, n = {n.get('AD', 0)}) "
          f"and without AD (hollow, n = {n.get('CN', 0)}), for networks built with "
          "(A) Pearson, (B) Spearman and (C) Kendall correlation. Each subject's (p, q) is the "
          "pair that best predicted 20% held-out edges of their own network (link-prediction "
          "AUC, mean of 3 splits; diagnosis not used). Bubble area is proportional to the number "
          "of subjects. Dashed lines mark p = 1 and q = 1 (unbiased walk). p < 1 favours "
          "returning to the previous region; q < 1 favours moving outward.")
    print("\nSaved ->")
    for name in FIG_NAMES:
        print(f"  {(FIG_DIR / name).relative_to(PROJECT_ROOT)}.png / .pdf")
    print(f"  {out_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
