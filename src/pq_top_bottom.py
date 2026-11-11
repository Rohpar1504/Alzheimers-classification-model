#!/usr/bin/env python3
"""
p/q investigation - Item 5: compare p, q between the top and bottom node2vec models.

Covers the advisor to-do item:
    3. "Top and bottom models for Node2vec -> compare p, q distributions among these 2 models"

Top and bottom come from Item 1:
    top    = node2vec + MLP on Spearman
    bottom = node2vec + Logistic Regression on Kendall
They differ in BOTH the classifier and the connectivity measure, so a difference
between them cannot be pinned on either one alone.

Two kinds of p/q, one panel each:
    A  Chosen for classification - in each of the 100 outer folds of the nested CV,
       the (p, q) the model picked on its training subjects.
           top:    outputs/results/pq_nested_cv.csv         (Step 3, MLP), Spearman rows
           bottom: outputs/results/pq_nested_cv_logreg.csv  (pq_nested_cv.py --clf logreg)
       Described with numbers only (no p-value): the 100 folds reuse the same 25
       subjects, so they are not independent and a test would be overconfident.
    B  Best fit per subject - each subject's (p, q) from link prediction (Item 2),
       which depends only on the measure: Spearman vs Kendall networks, same subjects.
       4 paired permutation tests (same 4 values as Item 4): each subject's
       Spearman - Kendall difference has its sign flipped at random N_PERMUTATIONS times;
       two-sided p = (1 + #flips with |mean diff| >= |real|) / (1 + N_PERMUTATIONS).
       Bonferroni: 4 tests -> significant if p < 0.05 / 4.
       Effect size: Spearman/Kendall ratio of geometric means (p, q); paired Cohen's d
       (mean difference / SD of differences) for the two preferences.

Bubble area = share of folds (A) or of subjects (B), so the two panels share one scale.

Outputs (no subject IDs):
    outputs/figures/pq_study/
        "Top and bottom models for Node2vec - compare p, q distributions among these 2 models".png/.pdf
    outputs/results/pq_top_bottom_counts.csv   share and count per (p, q), per panel and model
    outputs/results/pq_top_bottom_tests.csv    panel A summary + panel B tests

Usage (needs pq_nested_cv.py kendall --clf logreg to have run first):
    .venv/bin/python src/pq_top_bottom.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_ad_cn_scatter import TICK_LABELS, FIG_DIR, RESULTS_DIR, PROJECT_ROOT
from pq_grid import P_VALUES, Q_VALUES

DATA_DIR = PROJECT_ROOT / "data" / "pq_subject_fit"
TITLE = "Top and bottom models for Node2vec - compare p, q distributions among these 2 models"
N_PERMUTATIONS = 10_000
ALPHA = 0.05
SEED = 0

# label, measure, nested-CV file. Okabe-Ito colours, different from the AD/CN pair.
MODELS = {
    "Top":    {"desc": "MLP, Spearman", "measure": "spearman",
               "cv_file": RESULTS_DIR / "pq_nested_cv.csv",
               "color": "#009E73", "filled": True, "dx": -0.17},
    "Bottom": {"desc": "LogReg, Kendall", "measure": "kendall",
               "cv_file": RESULTS_DIR / "pq_nested_cv_logreg.csv",
               "color": "#CC79A7", "filled": False, "dx": +0.17},
}
SIZE_PER_PCT = 9            # marker area (pt^2) per 1% of folds/subjects
LEGEND_PCTS = [4, 12, 28]   # 4% = 1 subject in panel B

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "pdf.fonttype": 42})


def marker_style(model: str) -> dict:
    m = MODELS[model]
    if m["filled"]:
        return {"facecolors": m["color"], "edgecolors": "white", "linewidths": 0.6}
    return {"facecolors": "white", "edgecolors": m["color"], "linewidths": 1.4}


# ---- data ---------------------------------------------------------------------------
def fold_choices() -> dict:
    """Panel A: each model's chosen (p, q) per outer fold."""
    out = {}
    for name, m in MODELS.items():
        df = pd.read_csv(m["cv_file"])
        out[name] = df[df.method == m["measure"]][["p", "q"]].reset_index(drop=True)
    return out


def subject_values(measure: str) -> pd.DataFrame:
    """Panel B: per subject - winning p, q (log2) and the two preferences (as Item 4)."""
    best = pd.read_csv(DATA_DIR / "subject_pq.csv")
    best = best[best.measure == measure]
    grid = pd.read_csv(DATA_DIR / "subject_auc_grid.csv")
    grid = grid[grid.measure == measure]
    mean_auc = grid.groupby(["subject", "p", "q"], as_index=False)["auc"].mean()
    rows = []
    for r in best.itertuples():
        s = mean_auc[mean_auc.subject == r.subject]
        rows.append({"subject": r.subject, "best_p": r.best_p, "best_q": r.best_q,
                     "log2_p": np.log2(r.best_p), "log2_q": np.log2(r.best_q),
                     "outward": s[s.q < 1].auc.mean() - s[s.q > 1].auc.mean(),
                     "no_backtrack": s[s.p > 1].auc.mean() - s[s.p < 1].auc.mean()})
    return pd.DataFrame(rows).set_index("subject").sort_index()


def share_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Count and % of rows at every (p, q) cell (zeros included)."""
    counts = pairs.groupby(["p", "q"]).size()
    rows = [{"p": p, "q": q, "count": int(counts.get((p, q), 0))}
            for p in P_VALUES for q in Q_VALUES]
    t = pd.DataFrame(rows)
    t["percent"] = 100 * t["count"] / len(pairs)
    return t


# ---- statistics ---------------------------------------------------------------------
def paired_perm_test(diff: np.ndarray, rng) -> float:
    """Two-sided sign-flip permutation p-value for mean(diff) != 0."""
    real = diff.mean()
    signs = rng.choice([-1.0, 1.0], size=(N_PERMUTATIONS, len(diff)))
    null = (signs * diff).mean(axis=1)
    return (1 + np.sum(np.abs(null) >= np.abs(real) - 1e-12)) / (1 + N_PERMUTATIONS)


def fold_summary(pairs: pd.DataFrame) -> dict:
    lp, lq = np.log2(pairs.p), np.log2(pairs.q)
    top3 = pairs.groupby(["p", "q"]).size().sort_values(ascending=False).head(3)
    return {"n": len(pairs),
            "pct_at_1_1": 100 * np.mean((pairs.p == 1) & (pairs.q == 1)),
            "n_distinct_pairs": int(pairs.groupby(["p", "q"]).ngroups),
            "geo_mean_p": 2 ** lp.mean(), "geo_mean_q": 2 ** lq.mean(),
            "spread_p_x": 2 ** lp.std(ddof=1), "spread_q_x": 2 ** lq.std(ddof=1),
            "most_chosen": ", ".join(f"({p:g},{q:g}) x{n}" for (p, q), n in top3.items())}


TESTS = [  # column, name, kind
    ("log2_p", "Winning p", "log"),
    ("log2_q", "Winning q", "log"),
    ("outward", "Outward preference", "diff"),
    ("no_backtrack", "No-backtracking preference", "diff"),
]


# ---- figure -------------------------------------------------------------------------
def draw_panel(ax, tables: dict, title: str, letter: str) -> None:
    ax.axvline(Q_VALUES.index(1.0), color="0.6", lw=0.7, ls="--", zorder=0)
    ax.axhline(P_VALUES.index(1.0), color="0.6", lw=0.7, ls="--", zorder=0)
    for name, m in MODELS.items():
        t = tables[name]
        t = t[t["count"] > 0]
        ax.scatter([Q_VALUES.index(q) + m["dx"] for q in t.q],
                   [P_VALUES.index(p) for p in t.p],
                   s=t.percent * SIZE_PER_PCT, zorder=3, **marker_style(name))
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
    ax.set_ylabel("p")
    ax.set_title(title)
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="right", va="bottom")


def main() -> None:
    folds = fold_choices()
    subj = {name: subject_values(m["measure"]) for name, m in MODELS.items()}
    assert subj["Top"].index.equals(subj["Bottom"].index), "subjects must match for pairing"
    n_subj = len(subj["Top"])

    tables_a = {name: share_table(folds[name]) for name in MODELS}
    tables_b = {name: share_table(subj[name].rename(columns={"best_p": "p", "best_q": "q"}))
                for name in MODELS}

    # counts table
    count_rows = []
    for panel, tables in (("A_chosen_for_classification", tables_a),
                          ("B_best_fit_per_subject", tables_b)):
        for name, t in tables.items():
            count_rows.append(t.assign(panel=panel, model=name,
                                       description=MODELS[name]["desc"]))
    counts = pd.concat(count_rows)[["panel", "model", "description", "p", "q",
                                    "count", "percent"]]

    # panel A summary
    summ_a = {name: fold_summary(folds[name]) for name in MODELS}

    # panel B tests (paired: same subjects, Spearman vs Kendall)
    rng = np.random.default_rng(SEED)
    cutoff = ALPHA / len(TESTS)
    test_rows = []
    for name in MODELS:
        s = summ_a[name]
        test_rows.append({"panel": "A", "model": name, "description": MODELS[name]["desc"],
                          "test": "chosen (p, q) over folds - described, not tested",
                          "n": s["n"], "top_typical": None, "bottom_typical": None,
                          **{k: v for k, v in s.items() if k != "n"}})
    for col, name, kind in TESTS:
        top, bot = subj["Top"][col].to_numpy(), subj["Bottom"][col].to_numpy()
        diff = top - bot
        pval = paired_perm_test(diff, rng)
        if kind == "log":
            typ_t, typ_b = 2 ** top.mean(), 2 ** bot.mean()
            eff, eff_type = 2 ** diff.mean(), "Top/Bottom ratio"
        else:
            typ_t, typ_b = top.mean(), bot.mean()
            sd = diff.std(ddof=1)
            eff, eff_type = (diff.mean() / sd if sd > 0 else 0.0), "paired Cohen's d"
        test_rows.append({"panel": "B", "model": "Top vs Bottom",
                          "description": "Spearman vs Kendall, same subjects",
                          "test": name, "n": n_subj, "top_typical": typ_t,
                          "bottom_typical": typ_b, "mean_difference": diff.mean(),
                          "effect_size": eff, "effect_size_type": eff_type,
                          "p_value": pval, "bonferroni_cutoff": cutoff,
                          "significant": bool(pval < cutoff)})
    tests = pd.DataFrame(test_rows)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 4.3))
    draw_panel(axes[0], tables_a, f"Chosen for classification\n({summ_a['Top']['n']} CV folds)", "A")
    draw_panel(axes[1], tables_b, f"Best fit per subject\n({n_subj} subjects)", "B")
    model_handles = [plt.scatter([], [], s=12 * SIZE_PER_PCT, label=f"{n} ({m['desc']})",
                                 **marker_style(n)) for n, m in MODELS.items()]
    size_handles = [plt.scatter([], [], s=pc * SIZE_PER_PCT, facecolors="0.55",
                                edgecolors="white", label=f"{pc}%") for pc in LEGEND_PCTS]
    spacer = Line2D([], [], color="none", label=" ")
    fig.legend(handles=model_handles + [spacer] + size_handles, loc="lower center",
               ncol=6, frameon=False, handletextpad=0.3, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(TITLE, fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95), w_pad=3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{TITLE}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{TITLE}.pdf")
    plt.close(fig)

    counts_csv = RESULTS_DIR / "pq_top_bottom_counts.csv"
    tests_csv = RESULTS_DIR / "pq_top_bottom_tests.csv"
    counts.round(4).to_csv(counts_csv, index=False)
    tests.round(4).to_csv(tests_csv, index=False)

    # ---- terminal summary ----
    print("Item 5: top vs bottom node2vec models - p/q")
    print("  top    = node2vec + MLP on Spearman      bottom = node2vec + LogReg on Kendall\n")
    print(f"Panel A - p/q chosen for classification ({summ_a['Top']['n']} folds each; "
          "described, not tested)")
    for name in MODELS:
        s = summ_a[name]
        print(f"  {name:6s} (1,1) chosen {s['pct_at_1_1']:4.0f}%   {s['n_distinct_pairs']:2d} "
              f"different pairs   typical p {s['geo_mean_p']:.2f} (x/ {s['spread_p_x']:.1f})  "
              f"typical q {s['geo_mean_q']:.2f} (x/ {s['spread_q_x']:.1f})   "
              f"most chosen: {s['most_chosen']}")
    print(f"\nPanel B - best fit per subject ({n_subj} subjects, Spearman vs Kendall, paired; "
          f"{N_PERMUTATIONS} sign flips; Bonferroni cutoff {cutoff:.4f})")
    print(f"  {'test':<28} {'Top':>8} {'Bottom':>8} {'effect size':>28} {'p-value':>9}  significant")
    for r in test_rows[len(MODELS):]:
        eff = f"{r['effect_size']:.2f} ({r['effect_size_type']})"
        print(f"  {r['test']:<28} {r['top_typical']:>8.4f} {r['bottom_typical']:>8.4f} "
              f"{eff:>28} {r['p_value']:>9.4f}  {'YES' if r['significant'] else 'no'}")
    print("  (p and q: geometric means; preferences: mean AUC differences)")

    print("\nSuggested caption:")
    print("  Node2vec walk parameters (p, q) for the best-performing model (filled; MLP on "
          "Spearman networks) and the worst-performing model (hollow; logistic regression on "
          "Kendall networks). (A) The (p, q) selected for classification by nested "
          f"cross-validation in each of {summ_a['Top']['n']} outer folds (4-fold CV repeated "
          "25 times; selection on training subjects only). (B) The (p, q) that best predicted "
          f"each subject's held-out edges (link-prediction AUC; {n_subj} subjects; depends on the "
          "network only, not the classifier). Bubble area is proportional to the share of "
          "folds (A) or subjects (B). Dashed lines mark p = 1 and q = 1 (unbiased walk). The two "
          "models differ in both classifier and correlation measure.")
    print(f"\nSaved ->\n  {(FIG_DIR / TITLE).relative_to(PROJECT_ROOT)}.png / .pdf")
    print(f"  {counts_csv.relative_to(PROJECT_ROOT)}")
    print(f"  {tests_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
