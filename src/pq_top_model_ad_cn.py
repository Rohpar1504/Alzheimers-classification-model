#!/usr/bin/env python3
"""
p/q investigation - Item 4: top node2vec model -> compare p, q for AD vs CN.

Covers the advisor to-do item:
    2. "Top model for Node2vec -> compare p, q distributions for healthy vs AD patients"

The top model from Item 1 is node2vec + MLP on Spearman. Each subject's (p, q) from
Item 2 comes from link prediction on their own network, so it depends only on the
connectivity measure (Spearman), not on the classifier. This script therefore compares
AD and CN subjects on their Spearman networks.

Four per-subject values, one panel and one test each:
    A  winning p        (from subject_pq.csv; compared on log2 scale)
    B  winning q        (same)
    C  outward preference         = mean AUC over q < 1  minus  mean AUC over q > 1
    D  no-backtracking preference = mean AUC over p > 1  minus  mean AUC over p < 1
    C and D use all 25 (p, q) scores of a subject (each averaged over the 3 splits),
    so they are not thrown off by near-ties between the winning pair and the runner-up.

Test: permutation test of the difference in group means (AD - CN). The AD/CN labels
are shuffled N_PERMUTATIONS times (group sizes kept); two-sided
p-value = (1 + #shuffles with |difference| >= |real difference|) / (1 + N_PERMUTATIONS),
the same "+1" rule as Step 5. Bonferroni: 4 tests -> significant if p < 0.05 / 4.
Effect size: AD/CN ratio of geometric means for p and q; Cohen's d for C and D.

Outputs (group-level only, no subject IDs):
    outputs/figures/pq_study/
        "Top model for Node2vec - compare p, q distributions for healthy vs AD patients".png/.pdf
    outputs/results/pq_top_model_ad_cn_tests.csv

Usage:
    .venv/bin/python src/pq_top_model_ad_cn.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_ad_cn_scatter import (GROUPS, TICK_LABELS, marker_style, FIG_DIR, RESULTS_DIR,
                              PROJECT_ROOT)
from pq_grid import P_VALUES

DATA_DIR = PROJECT_ROOT / "data" / "pq_subject_fit"
MEASURE = "spearman"                 # top model (Item 1): node2vec + MLP on Spearman
TITLE = "Top model for Node2vec - compare p, q distributions for healthy vs AD patients"
N_PERMUTATIONS = 10_000
ALPHA = 0.05
SEED = 0
JITTER = 0.13                        # sideways spread of dots within a group (C, D)
STACK_STEP = 0.075                   # spacing of dots sharing one value (A, B)


def stacked_offsets(v: np.ndarray) -> np.ndarray:
    """Place subjects with the same value side by side so none are hidden."""
    off = np.zeros(len(v))
    for val in np.unique(v):
        idx = np.flatnonzero(v == val)
        off[idx] = (np.arange(len(idx)) - (len(idx) - 1) / 2) * STACK_STEP
    return off


def subject_values() -> pd.DataFrame:
    """One row per Spearman subject: group, log2 p, log2 q, outward, no-backtracking."""
    best = pd.read_csv(DATA_DIR / "subject_pq.csv")
    best = best[best.measure == MEASURE]
    grid = pd.read_csv(DATA_DIR / "subject_auc_grid.csv")
    grid = grid[grid.measure == MEASURE]
    mean_auc = grid.groupby(["subject", "p", "q"], as_index=False)["auc"].mean()

    rows = []
    for r in best.itertuples():
        s = mean_auc[mean_auc.subject == r.subject]
        rows.append({"subject": r.subject, "group": r.group,
                     "log2_p": np.log2(r.best_p), "log2_q": np.log2(r.best_q),
                     "outward": s[s.q < 1].auc.mean() - s[s.q > 1].auc.mean(),
                     "no_backtrack": s[s.p > 1].auc.mean() - s[s.p < 1].auc.mean()})
    return pd.DataFrame(rows)


def perm_test(values: np.ndarray, is_ad: np.ndarray, rng) -> tuple[float, float, np.ndarray]:
    """Real AD - CN mean difference, two-sided permutation p-value, null differences."""
    real = values[is_ad].mean() - values[~is_ad].mean()
    null = np.empty(N_PERMUTATIONS)
    for k in range(N_PERMUTATIONS):
        lab = rng.permutation(is_ad)
        null[k] = values[lab].mean() - values[~lab].mean()
    pval = (1 + np.sum(np.abs(null) >= np.abs(real) - 1e-12)) / (1 + N_PERMUTATIONS)
    return real, pval, null


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


PANELS = [  # column, panel title, y label, kind
    ("log2_p", "Winning p", "p", "log"),
    ("log2_q", "Winning q", "q", "log"),
    ("outward", "Outward preference", "AUC difference", "diff"),
    ("no_backtrack", "No-backtracking preference", "AUC difference", "diff"),
]


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main() -> None:
    df = subject_values()
    is_ad = (df.group == "AD").to_numpy()
    n_ad, n_cn = int(is_ad.sum()), int((~is_ad).sum())
    cutoff = ALPHA / len(PANELS)
    rng = np.random.default_rng(SEED)

    results = []
    for col, title, _, kind in PANELS:
        v = df[col].to_numpy()
        real, pval, _ = perm_test(v, is_ad, rng)
        ad, cn = v[is_ad], v[~is_ad]
        row = {"measure": MEASURE, "test": title, "n_AD": n_ad, "n_CN": n_cn}
        if kind == "log":       # typical value = geometric mean of p (or q)
            row.update({"AD_typical": 2 ** ad.mean(), "CN_typical": 2 ** cn.mean(),
                        "summary": "geometric mean", "difference": real,
                        "difference_unit": "log2", "effect_size": 2 ** real,
                        "effect_size_type": "AD/CN ratio"})
        else:
            row.update({"AD_typical": ad.mean(), "CN_typical": cn.mean(),
                        "summary": "mean", "difference": real,
                        "difference_unit": "AUC", "effect_size": cohens_d(ad, cn),
                        "effect_size_type": "Cohen's d"})
        row.update({"p_value": pval, "bonferroni_cutoff": cutoff,
                    "significant": bool(pval < cutoff)})
        results.append(row)
    res = pd.DataFrame(results)

    # ---- figure --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(7.09, 3.1))
    jit = np.random.default_rng(SEED)
    for k, (ax, (col, title, ylab, kind), r) in enumerate(zip(axes, PANELS, results)):
        for x, (group, g) in enumerate(GROUPS.items()):
            v = df.loc[df.group == group, col].to_numpy()
            xs = x + (stacked_offsets(v) if kind == "log"
                      else jit.uniform(-JITTER, JITTER, len(v)))
            ax.scatter(xs, v, s=22, zorder=3, **marker_style(group))
            typical = v.mean()
            ax.plot([x - 0.42, x + 0.42], [typical, typical], color="0.15", lw=1.2, zorder=4)
        ax.set_xticks([0, 1], list(GROUPS))
        ax.set_xlim(-0.65, 1.65)
        if kind == "log":
            ax.set_yticks(np.log2(P_VALUES), TICK_LABELS)
            ax.set_ylim(np.log2(P_VALUES[0]) - 0.4, np.log2(P_VALUES[-1]) + 0.4)
            ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=0)
        else:
            ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=0)
        ax.set_ylabel(ylab)
        ax.set_title(title if kind == "log" else title.replace(" preference", "\npreference"),
                     fontsize=8.5)
        ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
        ax.tick_params(length=2.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0.5, -0.2, fmt_p(r["p_value"]), transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
        ax.text(-0.25, 1.02, "ABCD"[k], transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="right", va="bottom")

    handles = [plt.scatter([], [], s=22, label=f"{g} (n = {n})", **marker_style(g))
               for g, n in (("AD", n_ad), ("CN", n_cn))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(TITLE, fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), w_pad=1.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{TITLE}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{TITLE}.pdf")
    plt.close(fig)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "pq_top_model_ad_cn_tests.csv"
    res.round(4).to_csv(out_csv, index=False)

    # ---- terminal summary ----------------------------------------------------------
    print(f"Item 4: top model (node2vec + MLP on {MEASURE.title()}) - p/q, AD vs CN")
    print(f"subjects: AD={n_ad}, CN={n_cn}   {N_PERMUTATIONS} label shuffles per test   "
          f"Bonferroni cutoff = {ALPHA}/{len(PANELS)} = {cutoff:.4f}\n")
    print(f"  {'test':<28} {'AD':>8} {'CN':>8} {'effect size':>22} {'p-value':>9}  significant")
    for r in results:
        eff = f"{r['effect_size']:.2f} ({r['effect_size_type']})"
        print(f"  {r['test']:<28} {r['AD_typical']:>8.4f} {r['CN_typical']:>8.4f} "
              f"{eff:>22} {r['p_value']:>9.4f}  {'YES' if r['significant'] else 'no'}")
    print("  (p and q: geometric means; preferences: mean AUC differences)")

    print("\nSuggested caption:")
    print(f"  Node2vec walk parameters for subjects with AD (filled, n = {n_ad}) and without "
          f"AD (hollow, n = {n_cn}), Spearman networks (input of the best-performing model, "
          "node2vec + MLP). (A) p and (B) q that best predicted each subject's held-out "
          "edges (log scale). (C) Outward preference: mean link-prediction AUC over q < 1 "
          "minus q > 1. (D) No-backtracking preference: mean AUC over p > 1 minus p < 1. "
          "Each dot is one subject (spread sideways for visibility); black bars show the "
          "group geometric mean (A, B) or mean (C, D). p-values: two-sided permutation "
          f"test of the group difference ({N_PERMUTATIONS:,} label shuffles); Bonferroni "
          f"threshold {cutoff:.4f} for 4 tests.")
    print(f"\nSaved ->\n  {(FIG_DIR / TITLE).relative_to(PROJECT_ROOT)}.png / .pdf")
    print(f"  {out_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
