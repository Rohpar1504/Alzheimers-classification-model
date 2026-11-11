#!/usr/bin/env python3
"""
p/q investigation - Step 3: repeated, nested cross-validation over node2vec's (p, q).

Answers the advisor's "print the p (local) and q (global) values across the folds of
the CV": for every outer CV fold we pick the best (p, q) using ONLY that fold's training
subjects, then score it on the held-out subjects, and print/save the choice per fold.

Why NESTED (leakage-free):
    If we picked the (p, q) with the best test accuracy, the test fold would have
    influenced the choice and the reported accuracy would be optimistic. Instead:
        outer loop  - 4-fold stratified CV; the held-out fold is ONLY used for scoring
        inner loop  - 3-fold stratified CV inside the outer training subjects; every
                      (p, q) on the grid is scored here, and the best one is selected
    The chosen (p, q) is then refit on the full outer training set and tested once.
    The embeddings themselves are unsupervised and per-subject (Step 2), so computing
    them before the split leaks no label or cross-subject information.

Why REPEATED:
    One 4-fold run gives only 4 selected (p, q) pairs - too few for the Step 8
    histogram. We repeat the outer 4-fold split N_REPEATS times with different
    shuffles, giving N_REPEATS x 4 selections per connectivity method.

Tie-breaking:
    With ~14 subjects per inner training split, inner accuracies move in coarse steps
    and several (p, q) often tie. Ties are broken toward the DeepWalk baseline (1, 1):
    the setting with the smallest |log2 p| + |log2 q| wins, then grid order. So a
    biased walk is only selected when it actually scores higher than an unbiased one,
    and the histogram does not just reflect the order of the grid.

Classifier:
    The same MLP as the main comparison (phase6_model_comparison.py: one hidden layer
    of 32 units, max_iter=1000, seed 0), so p/q is the only thing that changes.

Subjects:
    By default, the subjects whose connectivity matrices contain NaN (undefined
    correlations) are EXCLUDED, per the advisor decision - they are detected from
    data/connectivity, not hardcoded. Pass --keep-all to use every cached subject.

Also recorded, on the SAME outer folds: the fixed DeepWalk baseline (p = q = 1), so
the tuned result can be compared against not tuning at all.

Output:
    outputs/results/pq_nested_cv.csv - one row per (method, repeat, fold): selected
    p and q, its inner CV accuracy, and test accuracy/sensitivity/specificity/F1 for
    both the selected (p, q) and the (1, 1) baseline. Steps 4 and 8 read this file.

Usage:
    .venv/bin/python src/pq_nested_cv.py              # all 3 methods, NaN subjects excluded
    .venv/bin/python src/pq_nested_cv.py spearman     # one or more methods by name
    .venv/bin/python src/pq_nested_cv.py --keep-all   # include the NaN subjects
    .venv/bin/python src/pq_nested_cv.py kendall --clf logreg
        # Item 5: same nested CV with Logistic Regression (the bottom model's classifier);
        # saved to outputs/results/pq_nested_cv_logreg.csv, the MLP file is not touched
"""

import os

# Single-threaded BLAS (see pq_embeddings.py): thousands of tiny MLP fits gain nothing
# from threading, and it avoids macOS OpenMP conflicts. Must be set before numpy loads.
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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from pq_grid import PQ_GRID, DEEPWALK_BASELINE

PROJECT_ROOT = SRC_DIR.parent
EMB_DIR = PROJECT_ROOT / "data" / "pq_embeddings"
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
METHODS = ["pearson", "spearman", "kendall"]

N_OUTER = 4        # outer folds, same k as the main comparison
N_INNER = 3        # inner folds, used only to select (p, q)
N_REPEATS = 25     # outer CV repeats -> 100 selections per method
SEED = 0


def make_mlp() -> MLPClassifier:
    """Identical to phase6_model_comparison.CLASSIFIER_CTORS['MLP']."""
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=SEED)


def make_logreg() -> LogisticRegression:
    """Identical to phase6_model_comparison.CLASSIFIER_CTORS['LogisticReg']."""
    return LogisticRegression(max_iter=1000, random_state=SEED)


# --clf option: MLP is the default (Step 3); logreg is used for the bottom model (Item 5).
CLASSIFIERS = {"mlp": (make_mlp, "MLP(32)"), "logreg": (make_logreg, "LogisticRegression")}


def fmt(x: float) -> str:
    """Same filename format as pq_embeddings.py: 0.25->'0.25', 1.0->'1'."""
    return f"{x:g}"


def metrics(y_true, y_pred) -> dict:
    """accuracy, sensitivity (AD=1 recall), specificity (CN=0 recall), F1 (for AD)."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1":          f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def nan_subject_ids() -> set:
    """Subjects whose connectivity matrix contains NaN under ANY method."""
    bad = set()
    for method in METHODS:
        for f in (CONN_DIR / method).glob("*/*.npy"):
            if np.isnan(np.load(f)).any():
                bad.add(f"{f.parent.name}/{f.stem}")
    return bad


def baseline_distance(pq) -> float:
    """How far (p, q) is from (1, 1) on the log2 scale - used to break ties."""
    p, q = pq
    return abs(np.log2(p)) + abs(np.log2(q))


def select_pq(feats: dict, y: np.ndarray, train_idx: np.ndarray, inner_seed: int,
              make_clf=make_mlp):
    """
    Inner CV on the outer TRAINING subjects only: score every (p, q) by mean inner
    accuracy and return (best_pq, its_inner_accuracy). The held-out outer fold is
    never touched here.
    """
    y_tr = y[train_idx]
    inner = list(StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=inner_seed)
                 .split(np.zeros(len(y_tr)), y_tr))
    scores = {}
    for pq in PQ_GRID:
        X_tr = feats[pq][train_idx]
        accs = []
        for i_tr, i_va in inner:
            clf = make_clf().fit(X_tr[i_tr], y_tr[i_tr])
            accs.append(accuracy_score(y_tr[i_va], clf.predict(X_tr[i_va])))
        scores[pq] = float(np.mean(accs))

    best_acc = max(scores.values())
    tied = [pq for pq in PQ_GRID if np.isclose(scores[pq], best_acc)]
    best = min(tied, key=baseline_distance)        # min() keeps grid order among equals
    return best, best_acc


def run_method(method: str, keep_idx: np.ndarray, y: np.ndarray, make_clf=make_mlp) -> list:
    """Repeated nested CV for one connectivity method; returns one row per outer fold."""
    feats = {(p, q): np.load(EMB_DIR / method / f"p{fmt(p)}_q{fmt(q)}.npy")[keep_idx]
             for p, q in PQ_GRID}

    outer = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(k, N_OUTER)
        (p, q), inner_acc = select_pq(feats, y, tr, inner_seed=SEED + k, make_clf=make_clf)

        sel = metrics(y[te], make_clf().fit(feats[(p, q)][tr], y[tr]).predict(feats[(p, q)][te]))
        Xb = feats[DEEPWALK_BASELINE]
        base = metrics(y[te], make_clf().fit(Xb[tr], y[tr]).predict(Xb[te]))

        rows.append({"method": method, "repeat": repeat, "fold": fold,
                     "p": p, "q": q, "inner_accuracy": round(inner_acc, 4),
                     **{f"test_{m}": v for m, v in sel.items()},
                     **{f"baseline_{m}": v for m, v in base.items()}})
        print(f"  [{method:8s}] repeat {repeat:>2} fold {fold}  p={fmt(p):>4} q={fmt(q):>4}  "
              f"inner={inner_acc:.3f}  test={sel['accuracy']:.3f}  "
              f"(p=q=1 test={base['accuracy']:.3f})")
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")      # MLP ConvergenceWarnings on tiny folds
    args = sys.argv[1:]
    keep_all = "--keep-all" in args
    methods = [m for m in args if m in METHODS] or list(METHODS)
    clf_key = args[args.index("--clf") + 1] if "--clf" in args else "mlp"
    if clf_key not in CLASSIFIERS:
        sys.exit(f"unknown --clf {clf_key!r}; choose from {list(CLASSIFIERS)}")
    make_clf, clf_label = CLASSIFIERS[clf_key]

    ids = json.loads((EMB_DIR / "ids.json").read_text())
    labels = np.load(EMB_DIR / "labels.npy")
    excluded = set() if keep_all else nan_subject_ids()
    keep_idx = np.array([i for i, sid in enumerate(ids) if sid not in excluded])
    y = labels[keep_idx]

    print("Step 3: repeated nested CV over node2vec (p, q)")
    print(f"subjects={len(y)} (AD={int(y.sum())}, CN={int((y == 0).sum())})  "
          f"excluded={sorted(excluded) or 'none'}")
    print(f"outer={N_OUTER}-fold x {N_REPEATS} repeats, inner={N_INNER}-fold, "
          f"grid={len(PQ_GRID)} (p,q), classifier={clf_label}\n")

    t0 = time.time()
    rows = []
    for method in methods:
        rows += run_method(method, keep_idx, y, make_clf)
        print()
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_all" if keep_all else "") + ("" if clf_key == "mlp" else f"_{clf_key}")
    out_csv = RESULTS_DIR / f"pq_nested_cv{suffix}.csv"
    df.round(4).to_csv(out_csv, index=False)

    print("---- Summary: test accuracy over all outer folds (mean +/- std) ----")
    for method, sub in df.groupby("method", sort=False):
        top = sub.groupby(["p", "q"]).size().sort_values(ascending=False).head(3)
        top_str = ", ".join(f"({fmt(p)},{fmt(q)}) x{n}" for (p, q), n in top.items())
        print(f"  {method:8s}  selected (p,q): {sub['test_accuracy'].mean():.3f} +/- "
              f"{sub['test_accuracy'].std():.3f}   p=q=1: {sub['baseline_accuracy'].mean():.3f} "
              f"+/- {sub['baseline_accuracy'].std():.3f}   most selected: {top_str}")

    print(f"\nSaved -> {out_csv.relative_to(PROJECT_ROOT)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
