#!/usr/bin/env python3
"""
p/q investigation - Step 5: data-leakage checks + label-permutation test.

Part A - leakage checks (each prints PASS/FAIL):
    1. Fold integrity: in every outer fold of Step 3's CV, train and test subjects are
       disjoint, and within each repeat every subject is tested exactly once.
    2. Selection uses training subjects only: re-run Step 3's (p, q) selection for the
       first outer fold, then again after scrambling the TEST subjects' labels. If the
       held-out labels influenced the choice the result would change; it must not.
       The re-derived choice must also match what Step 3 saved.
    3. Embeddings never see labels: pq_embeddings.embed_subject takes only
       (matrix, p, q), and recomputing one subject's (1, 1) embedding from its
       connectivity matrix reproduces the cached vector.
    4. No duplicate subjects: no two subjects share an identical embedding (which
       would put a test subject's "twin" in the training set).

Part B - permutation test ("could this accuracy happen by luck?"):
    Model: node2vec with p = q = 1 + the same MLP as everywhere else - the setting
    Step 3 showed beats tuning, and so the result that would actually be reported.
    Score: mean accuracy over 4-fold stratified CV repeated N_REPEATS times.
    Null:  the AD/CN labels are randomly shuffled N_PERMUTATIONS times (keeping the
           12/13 split, breaking any real brain-diagnosis link) and the SAME score is
           recomputed each time. This is scikit-learn's permutation_test_score.
    p-value = (1 + #shuffles scoring >= the real labels) / (1 + N_PERMUTATIONS).
    Three tests (one per connectivity method) -> Bonferroni threshold 0.05 / 3.

    The real score here uses N_REPEATS = 5 (not Step 3's 25) so 1,000 shuffles stay
    tractable; the real and shuffled scores must be computed identically, so both use 5.

Caveat for the paper: Spearman is tested partly because it already looked best on
these same subjects, so its p-value is somewhat optimistic (a selection effect).

Outputs:
    outputs/results/pq_permutation_test.csv    real score, null mean/std, p-value per method
    outputs/results/pq_permutation_null.csv    every shuffled score, per method
    outputs/figures/pq_study/pq_permutation_null.png

Usage:
    .venv/bin/python src/pq_permutation_test.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_grid import PQ_GRID, DEEPWALK_BASELINE
from pq_nested_cv import (make_mlp, fmt, nan_subject_ids, select_pq,
                          EMB_DIR, CONN_DIR, N_OUTER, SEED)
from pq_nested_cv import N_REPEATS as STEP3_REPEATS

PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "pq_study"
METHODS = ["pearson", "spearman", "kendall"]

N_REPEATS = 5           # CV repeats inside each (real or shuffled) score
N_PERMUTATIONS = 1000
ALPHA = 0.05

METHOD_COLOR = {"pearson": "#2a78d6", "spearman": "#eb6834", "kendall": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_subjects():
    """Same 25 subjects as Step 3: cached ids/labels minus the NaN-matrix subjects."""
    ids = json.loads((EMB_DIR / "ids.json").read_text())
    labels = np.load(EMB_DIR / "labels.npy")
    excluded = nan_subject_ids()
    keep = np.array([i for i, sid in enumerate(ids) if sid not in excluded])
    return [ids[i] for i in keep], labels[keep], keep


def load_feats(method: str, keep: np.ndarray) -> dict:
    return {(p, q): np.load(EMB_DIR / method / f"p{fmt(p)}_q{fmt(q)}.npy")[keep]
            for p, q in PQ_GRID}


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# --- Part A: leakage checks ---------------------------------------------------------

def check_folds(y: np.ndarray) -> bool:
    outer = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=STEP3_REPEATS, random_state=SEED)
    splits = list(outer.split(np.zeros(len(y)), y))
    disjoint = all(not set(tr) & set(te) for tr, te in splits)
    once = all(sorted(np.concatenate([te for _, te in splits[r * N_OUTER:(r + 1) * N_OUTER]]))
               == list(range(len(y))) for r in range(STEP3_REPEATS))
    return report("fold integrity", disjoint and once,
                  f"{len(splits)} outer folds; train/test disjoint={disjoint}; "
                  f"each subject tested once per repeat={once}")


def check_selection(feats: dict, y: np.ndarray, method: str) -> bool:
    tr, te = next(RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=STEP3_REPEATS,
                                          random_state=SEED).split(np.zeros(len(y)), y))
    pq_real, _ = select_pq(feats, y, tr, inner_seed=SEED)
    y_scrambled = y.copy()
    y_scrambled[te] = 1 - y_scrambled[te]            # flip every held-out label
    pq_scr, _ = select_pq(feats, y_scrambled, tr, inner_seed=SEED)

    saved = pd.read_csv(RESULTS_DIR / "pq_nested_cv.csv")
    row = saved[(saved["method"] == method) & (saved["repeat"] == 0) & (saved["fold"] == 0)].iloc[0]
    matches_step3 = (row["p"], row["q"]) == pq_real
    return report(f"selection ignores test labels ({method})",
                  pq_real == pq_scr and matches_step3,
                  f"chosen (p,q)={pq_real}, with test labels flipped={pq_scr}, "
                  f"Step 3 saved=({fmt(row['p'])}, {fmt(row['q'])})")


def check_embeddings_label_free(ids: list, keep: np.ndarray) -> bool:
    from pq_embeddings import embed_subject           # imports node2vec only when needed
    params = list(inspect.signature(embed_subject).parameters)
    group, stem = ids[0].split("/")
    mat = np.load(CONN_DIR / "pearson" / group / f"{stem}.npy")
    fresh = embed_subject(mat, *DEEPWALK_BASELINE)
    cached = np.load(EMB_DIR / "pearson" / "p1_q1.npy")[keep[0]]
    same = np.allclose(fresh, cached, atol=1e-6)
    return report("embeddings never see labels", params == ["mat", "p", "q"] and same,
                  f"embed_subject{tuple(params)}; recomputed {ids[0]} (pearson, p=q=1) "
                  f"matches cache={same} (max diff {np.abs(fresh - cached).max():.1e})")


def check_duplicates(all_feats: dict) -> bool:
    dupes = [(m, pq) for m, feats in all_feats.items() for pq, X in feats.items()
             if len(np.unique(X.round(10), axis=0)) != len(X)]
    return report("no duplicate subjects", not dupes,
                  f"checked {sum(len(f) for f in all_feats.values())} embedding sets; "
                  f"with duplicate rows: {dupes or 'none'}")


# --- Part B: permutation test --------------------------------------------------------

def plot_null(results: pd.DataFrame, null: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (_, r) in zip(axes, results.iterrows()):
        m = r["method"]
        ax.hist(null[m], bins=bins, color=METHOD_COLOR[m], alpha=0.85,
                edgecolor="white", linewidth=1, zorder=3)
        ax.axvline(r["real_accuracy"], color=INK, linewidth=2, zorder=4)
        ax.text(r["real_accuracy"] + 0.015, ax.get_ylim()[1] * 0.92,
                f"real labels\n{r['real_accuracy']:.3f}\np = {r['p_value']:.3f}",
                fontsize=8.5, color=INK, va="top")
        ax.set_xlim(0, 1)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
        ax.set_title(m.capitalize(), fontsize=11, color=INK, pad=8)
        ax.set_xlabel("mean CV accuracy", fontsize=9, color=INK_2)
    axes[0].set_ylabel("shuffled-label runs", fontsize=9, color=INK_2)
    fig.suptitle(f"Permutation test: node2vec (p = q = 1) + MLP vs {N_PERMUTATIONS} label shuffles",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "Bars = accuracy with shuffled AD/CN labels (pure luck). "
             "Black line = accuracy with the real labels.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    ids, y, keep = load_subjects()
    all_feats = {m: load_feats(m, keep) for m in METHODS}
    print(f"Step 5: leakage checks + permutation test  "
          f"(subjects={len(y)}: AD={int(y.sum())}, CN={int((y == 0).sum())})\n")

    print("Part A - leakage checks")
    ok = [check_folds(y)]
    ok += [check_selection(all_feats[m], y, m) for m in METHODS]
    ok.append(check_embeddings_label_free(ids, keep))
    ok.append(check_duplicates(all_feats))
    print(f"  => {sum(ok)}/{len(ok)} checks passed\n")

    print(f"Part B - permutation test ({N_PERMUTATIONS} shuffles, score = mean accuracy over "
          f"{N_OUTER}-fold x {N_REPEATS} repeats, model = node2vec p=q=1 + MLP)")
    cv = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    rows, null = [], {}
    for m in METHODS:
        X = all_feats[m][DEEPWALK_BASELINE]
        score, perm_scores, pval = permutation_test_score(
            make_mlp(), X, y, cv=cv, n_permutations=N_PERMUTATIONS,
            scoring="accuracy", n_jobs=-1, random_state=SEED)
        null[m] = perm_scores
        rows.append({"method": m, "real_accuracy": score,
                     "null_mean": perm_scores.mean(), "null_std": perm_scores.std(ddof=1),
                     "null_95th_percentile": np.percentile(perm_scores, 95),
                     "n_null_at_or_above_real": int((perm_scores >= score).sum()),
                     "p_value": pval,
                     "significant_bonferroni": pval < ALPHA / len(METHODS)})
        print(f"  {m:8s} real={score:.3f}  null={perm_scores.mean():.3f}+/-{perm_scores.std(ddof=1):.3f}"
              f"  (95th pct {np.percentile(perm_scores, 95):.3f})  p={pval:.4f}"
              f"  {'significant' if pval < ALPHA / len(METHODS) else 'not significant'}"
              f" at {ALPHA}/{len(METHODS)}")

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(RESULTS_DIR / "pq_permutation_test.csv", index=False)
    null_df = pd.DataFrame(null)
    null_df.round(4).to_csv(RESULTS_DIR / "pq_permutation_null.csv", index_label="permutation")
    plot_null(results, null_df, FIG_DIR / "pq_permutation_null.png")

    print(f"\nSaved -> outputs/results/pq_permutation_test.csv, pq_permutation_null.csv, "
          f"outputs/figures/pq_study/pq_permutation_null.png  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
