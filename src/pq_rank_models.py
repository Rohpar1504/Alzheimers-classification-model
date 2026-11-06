#!/usr/bin/env python3
"""
p/q investigation - Item 1: re-rank the 9 node2vec models on the 25 clean subjects.

Scores node2vec (p = q = 1) + {RandomForest, MLP, LogisticReg} on
{Pearson, Spearman, Kendall} and ranks them, to pick the TOP and BOTTOM node2vec
models used in later items (top model AD vs CN; top vs bottom p/q comparison).

Why re-rank: the old ranking in model_comparison.csv used all 28 subjects (including
the 3 NaN subjects) and a single 4-fold split, which is noisy at this sample size.

How:
    - Embeddings: the cached p = q = 1 embeddings from Step 2 (same node2vec settings
      as the main comparison), with the NaN subjects removed. No node2vec is rerun.
    - Classifiers: identical to phase6_model_comparison.CLASSIFIER_CTORS.
    - Evaluation: 4-fold stratified CV repeated 25 times (100 folds) - the same folds
      as Step 3 - reported as mean +/- std. Ranked by mean accuracy, then F1.
    - Also reported: one plain 4-fold run (random_state=0), the way the old table was
      made, to show how much a single split moves the numbers.

Output:
    outputs/results/pq_node2vec_ranking.csv - 9 rows, ranked, top/bottom marked.

Usage:
    .venv/bin/python src/pq_rank_models.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_nested_cv import (make_mlp, metrics, nan_subject_ids, EMB_DIR, RESULTS_DIR,
                          METHODS, N_OUTER, N_REPEATS, SEED, PROJECT_ROOT)

# Same settings as phase6_model_comparison.CLASSIFIER_CTORS.
CLASSIFIERS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=SEED),
    "MLP":          make_mlp,
    "LogisticReg":  lambda: LogisticRegression(max_iter=1000, random_state=SEED),
}
METRICS = ["accuracy", "sensitivity", "specificity", "f1"]


def cv_scores(make_clf, X, y, splits) -> pd.DataFrame:
    """One row of metrics per fold."""
    rows = []
    for tr, te in splits:
        clf = make_clf().fit(X[tr], y[tr])
        rows.append(metrics(y[te], clf.predict(X[te])))
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")      # MLP ConvergenceWarnings on tiny folds
    t0 = time.time()

    ids = json.loads((EMB_DIR / "ids.json").read_text())
    labels = np.load(EMB_DIR / "labels.npy")
    excluded = nan_subject_ids()
    keep = np.array([i for i, sid in enumerate(ids) if sid not in excluded])
    y = labels[keep]
    print(f"Item 1: re-rank node2vec models (p = q = 1)  subjects={len(y)} "
          f"(AD={int(y.sum())}, CN={int((y == 0).sum())})  excluded={sorted(excluded)}")

    dummy = np.zeros(len(y))
    repeated = list(RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS,
                                            random_state=SEED).split(dummy, y))
    single = list(StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                                  random_state=SEED).split(dummy, y))
    print(f"evaluation: {N_OUTER}-fold x {N_REPEATS} repeats = {len(repeated)} folds "
          f"(same as Step 3), plus one plain {N_OUTER}-fold run\n")

    rows = []
    for method in METHODS:
        X = np.load(EMB_DIR / method / "p1_q1.npy")[keep]
        for clf_name, make_clf in CLASSIFIERS.items():
            rep = cv_scores(make_clf, X, y, repeated)
            one = cv_scores(make_clf, X, y, single)
            row = {"model": f"node2vec+{clf_name}", "method": method}
            for m in METRICS:
                row[m] = rep[m].mean()
                row[f"{m}_std"] = rep[m].std(ddof=1)
            row["single_run_accuracy"] = one["accuracy"].mean()
            row["single_run_accuracy_std"] = one["accuracy"].std(ddof=1)
            rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values(["accuracy", "f1"], ascending=False, ignore_index=True))
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["role"] = ""
    df.loc[0, "role"] = "top"
    df.loc[len(df) - 1, "role"] = "bottom"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "pq_node2vec_ranking.csv"
    df.round(4).to_csv(out_csv, index=False)

    print(f"{'rank':>4}  {'model':<22} {'method':<9} {'accuracy':>15} {'F1':>15} "
          f"{'sens':>6} {'spec':>6} {'single 4-fold':>15}")
    for r in df.itertuples():
        print(f"{r.rank:>4}  {r.model:<22} {r.method:<9} "
              f"{r.accuracy:.3f} +/- {r.accuracy_std:.3f} {r.f1:.3f} +/- {r.f1_std:.3f} "
              f"{r.sensitivity:>6.3f} {r.specificity:>6.3f} "
              f"{r.single_run_accuracy:.3f} +/- {r.single_run_accuracy_std:.3f}"
              f"{'   <- ' + r.role.upper() if r.role else ''}")

    top, bottom = df.iloc[0], df.iloc[-1]
    print(f"\nTOP:    {top.model} on {top.method} ({top.accuracy:.3f})")
    print(f"BOTTOM: {bottom.model} on {bottom.method} ({bottom.accuracy:.3f})"
          f"{'  - at or below chance (0.50): no real signal' if bottom.accuracy <= 0.5 else ''}")
    print(f"\nSaved -> {out_csv.relative_to(PROJECT_ROOT)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
