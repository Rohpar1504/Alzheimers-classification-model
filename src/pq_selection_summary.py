#!/usr/bin/env python3
"""
p/q investigation - Steps 4 & 8: summarise which (p, q) the nested CV selected.

Reads the per-fold output of Step 3 (outputs/results/pq_nested_cv.csv) - nothing is
retrained - and produces:

Step 4 - "represent mean and std dev as a table":
    outputs/results/pq_selection_summary.csv   one row per connectivity method:
        - selected p and q, mean +/- std over the outer folds, in TWO forms:
            arithmetic  - the plain mean +/- std, as asked
            geometric   - the mean taken on the log2 scale and converted back. The
                          grid is powers of two, so on the plain scale one pick of 4
                          outweighs eight picks of 0.25; the geometric mean treats
                          0.25 and 4 as equally far from 1. Its spread is reported as
                          a multiplicative factor ("x/ 2.1" = within a factor of 2.1).
        - median and most-selected (p, q); how often (1, 1) was chosen; how many
          distinct pairs were chosen
        - test accuracy / F1 (mean +/- std) for the selected (p, q) vs p = q = 1
    outputs/results/pq_selection_counts.csv    5 x 5 grid per method (p rows, q
        columns): how many outer folds selected each pair - the data behind the plots

Step 8 - "histogram / scatter plot":
    outputs/figures/pq_study/pq_selection_histogram.png
        how often each p value and each q value was selected, per method
    outputs/figures/pq_study/pq_selection_scatter.png
        p vs q (log2 axes), one circle per grid point sized by how many folds chose it.
        A plain scatter would stack all 100 points on the 25 grid positions, so the
        circle size (and the printed count) carries the frequency instead.

Usage:
    .venv/bin/python src/pq_selection_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_grid import P_VALUES, Q_VALUES, DEEPWALK_BASELINE

PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "pq_study"
IN_CSV = RESULTS_DIR / "pq_nested_cv.csv"

METHODS = ["pearson", "spearman", "kendall"]
# One fixed colour per connectivity method (identity), same order everywhere.
METHOD_COLOR = {"pearson": "#2a78d6", "spearman": "#eb6834", "kendall": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def fmt(x: float) -> str:
    return f"{x:g}"


def geo_mean_spread(values: pd.Series):
    """Geometric mean and multiplicative spread (2 ** std of log2 values)."""
    logs = np.log2(values.to_numpy())
    return 2 ** logs.mean(), 2 ** logs.std(ddof=1)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        s = df[df["method"] == method]
        pair_counts = s.groupby(["p", "q"]).size().sort_values(ascending=False)
        (mode_p, mode_q), mode_n = pair_counts.index[0], pair_counts.iloc[0]
        row = {"method": method, "n_folds": len(s)}
        for par in ("p", "q"):
            g_mean, g_spread = geo_mean_spread(s[par])
            row[f"{par}_mean"] = s[par].mean()
            row[f"{par}_std"] = s[par].std(ddof=1)
            row[f"{par}_geo_mean"] = g_mean
            row[f"{par}_geo_spread_factor"] = g_spread
            row[f"{par}_median"] = s[par].median()
        row.update({
            "most_selected": f"({fmt(mode_p)}, {fmt(mode_q)})",
            "most_selected_n": int(mode_n),
            "baseline_11_selected_n": int(((s["p"] == 1) & (s["q"] == 1)).sum()),
            "distinct_pairs": len(pair_counts),
        })
        for tag, col in (("tuned", "test"), ("baseline_11", "baseline")):
            for m in ("accuracy", "f1"):
                row[f"{tag}_{m}_mean"] = s[f"{col}_{m}"].mean()
                row[f"{tag}_{m}_std"] = s[f"{col}_{m}"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def count_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: method, p, then one column per q value, cells = selection counts."""
    frames = []
    for method in METHODS:
        s = df[df["method"] == method]
        grid = (s.groupby(["p", "q"]).size().unstack(fill_value=0)
                 .reindex(index=P_VALUES, columns=Q_VALUES, fill_value=0))
        grid.columns = [f"q={fmt(q)}" for q in grid.columns]
        grid.insert(0, "p", [fmt(p) for p in grid.index])
        grid.insert(0, "method", method)
        frames.append(grid.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)


def plot_histogram(df: pd.DataFrame, out: Path) -> None:
    """2 x 3 small multiples: rows = p, q; columns = method; bars = selection counts."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    ymax = max(df.groupby(["method", par]).size().max() for par in ("p", "q"))
    xs = np.arange(len(P_VALUES))
    for col, method in enumerate(METHODS):
        s = df[df["method"] == method]
        for row, (par, values, base) in enumerate((("p", P_VALUES, DEEPWALK_BASELINE[0]),
                                                   ("q", Q_VALUES, DEEPWALK_BASELINE[1]))):
            ax = axes[row, col]
            counts = s[par].value_counts().reindex(values, fill_value=0).to_numpy()
            bars = ax.bar(xs, counts, width=0.72, color=METHOD_COLOR[method],
                          edgecolor="white", linewidth=2, zorder=3)
            # Mark the unbiased (DeepWalk) value with hatching, so it is not colour-only.
            bars[values.index(base)].set_hatch("///")
            for x, c in zip(xs, counts):
                ax.text(x, c + ymax * 0.02, str(c), ha="center", va="bottom",
                        fontsize=8.5, color=INK_2)
            ax.set_xticks(xs, [fmt(v) + (" *" if v == base else "") for v in values])
            ax.set_ylim(0, ymax * 1.15)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            style_axes(ax)
            if row == 0:
                ax.set_title(method.capitalize(), fontsize=11, color=INK, pad=8)
            ax.set_xlabel(f"selected {par}", fontsize=9, color=INK_2)
            if col == 0:
                ax.set_ylabel("outer folds", fontsize=9, color=INK_2)
    n = df.groupby("method").size().iloc[0]
    fig.suptitle(f"How often each p and q was selected by nested CV ({n} outer folds per method)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "* = 1, the unbiased (DeepWalk) value; its bar is hatched.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out: Path) -> None:
    """One panel per method: p vs q on log2 axes, circle area = selection count."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    max_n = df.groupby(["method", "p", "q"]).size().max()
    ticks = np.log2(P_VALUES)
    for ax, method in zip(axes, METHODS):
        s = df[df["method"] == method]
        counts = s.groupby(["p", "q"]).size()
        for (p, q), n in counts.items():
            x, y = np.log2(q), np.log2(p)
            ax.scatter(x, y, s=170 + 1300 * n / max_n, color=METHOD_COLOR[method], alpha=0.85,
                       edgecolor="white", linewidth=2, zorder=3)
            ax.text(x, y, str(n), ha="center", va="center", fontsize=8,
                    color="white", zorder=4)
        # Baseline (1, 1) outlined so it is findable whether or not it was chosen.
        ax.scatter(0, 0, s=1700, facecolor="none", edgecolor=INK_2, linewidth=1,
                   linestyle=(0, (3, 2)), zorder=2)
        ax.set_xticks(ticks, [fmt(v) for v in Q_VALUES])
        ax.set_yticks(ticks, [fmt(v) for v in P_VALUES])
        ax.set_xlim(ticks[0] - 0.7, ticks[-1] + 0.7)
        ax.set_ylim(ticks[0] - 0.7, ticks[-1] + 0.7)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_aspect("equal")
        style_axes(ax)
        ax.set_title(method.capitalize(), fontsize=11, color=INK, pad=8)
        ax.set_xlabel("q   (< 1 global / DFS,  > 1 local / BFS)", fontsize=8.5, color=INK_2)
    axes[0].set_ylabel("p   (< 1 return-prone,  > 1 return-averse)", fontsize=8.5, color=INK_2)
    fig.suptitle("Selected (p, q) pairs across outer folds (number = folds that chose the pair)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.915, "Dashed ring = (1, 1), the unbiased DeepWalk setting. Axes are log2-spaced.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = summary_table(df)
    summary.round(3).to_csv(RESULTS_DIR / "pq_selection_summary.csv", index=False)
    counts = count_grid(df)
    counts.to_csv(RESULTS_DIR / "pq_selection_counts.csv", index=False)

    print("Step 4: selected (p, q) across outer folds\n")
    print(f"{'method':9s} {'p mean+/-std':>14s} {'p geo':>11s} {'q mean+/-std':>14s} "
          f"{'q geo':>11s} {'most chosen':>16s} {'(1,1)':>6s} {'tuned acc':>13s} {'p=q=1 acc':>13s}")
    for _, r in summary.iterrows():
        print(f"{r['method']:9s} {r['p_mean']:6.2f} +/- {r['p_std']:4.2f} "
              f"{r['p_geo_mean']:5.2f} x/{r['p_geo_spread_factor']:4.2f} "
              f"{r['q_mean']:6.2f} +/- {r['q_std']:4.2f} "
              f"{r['q_geo_mean']:5.2f} x/{r['q_geo_spread_factor']:4.2f} "
              f"{r['most_selected']:>11s} x{r['most_selected_n']:<3d} {r['baseline_11_selected_n']:>5d} "
              f"{r['tuned_accuracy_mean']:.3f}+/-{r['tuned_accuracy_std']:.3f} "
              f"{r['baseline_11_accuracy_mean']:.3f}+/-{r['baseline_11_accuracy_std']:.3f}")
    print("\nSelection counts (rows p, columns q):")
    print(counts.to_string(index=False))

    plot_histogram(df, FIG_DIR / "pq_selection_histogram.png")
    plot_scatter(df, FIG_DIR / "pq_selection_scatter.png")
    print(f"\nSaved -> {(RESULTS_DIR / 'pq_selection_summary.csv').relative_to(PROJECT_ROOT)}, "
          f"{(RESULTS_DIR / 'pq_selection_counts.csv').relative_to(PROJECT_ROOT)}, "
          f"{FIG_DIR.relative_to(PROJECT_ROOT)}/pq_selection_{{histogram,scatter}}.png")


if __name__ == "__main__":
    main()
