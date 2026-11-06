#!/usr/bin/env python3
"""
p/q investigation - Item 2: fit node2vec's (p, q) to EACH subject, by link prediction.

Goal: give every subject their own p and q (so later items can compare AD vs CN).
Classification can't do that - it needs a group of subjects - so instead we ask, for
each subject separately: which walk setting best captures the structure of THIS
person's brain network? Measured by link prediction, as in the node2vec paper
(Grover & Leskovec 2016, Section 4.4).

For each subject, each connectivity measure, and each of 3 random splits:
    1. Build the subject's network exactly as everywhere else (top 20% of |corr|).
    2. Hide 20% of its edges. A maximum spanning forest is never hidden, so every
       region keeps at least one visible edge and is still walked on.
    3. Sample the same number of "fake" pairs: region pairs NOT connected in the
       original network (only among regions that have edges).
    4. For each of the 25 (p, q) pairs, run node2vec on the remaining 80% of edges
       (same settings as every other node2vec run in the project).
    5. Score every hidden edge and fake pair by the cosine similarity of the two
       regions' embeddings, and compute the AUC (chance a hidden edge outscores a
       fake pair; 0.5 = guessing, 1.0 = perfect).
    The 25 pairs share the same hidden/fake sets within a split, so they are compared
    fairly. AUC is averaged over the 3 splits; the subject's (p, q) is the pair with
    the highest mean AUC. Ties go toward (1, 1), as in Step 3.

Labels are never used in the fit - the AD/CN group is only attached to the output
for later plotting. Each subject is fitted on their own network alone.

Outputs (contain ADNI subject IDs -> saved under data/, which git ignores):
    data/pq_subject_fit/subject_pq.csv        one row per subject x measure:
        best p, best q, best AUC, AUC at (1, 1), margin = best - (1, 1)
    data/pq_subject_fit/subject_auc_grid.csv  every subject x measure x (p, q) x split
    data/pq_subject_fit/parts/                per-subject progress files (resumable)

Usage:
    .venv/bin/python src/pq_subject_fit.py              # all 3 measures
    .venv/bin/python src/pq_subject_fit.py spearman     # one or more measures
Interrupted runs pick up where they stopped (finished subjects are skipped).
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
import networkx as nx
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from graph_utils import build_nx_graph
from pq_grid import PQ_GRID, DEEPWALK_BASELINE
from pq_nested_cv import (nan_subject_ids, baseline_distance, fmt, EMB_DIR, CONN_DIR,
                          METHODS, SEED, PROJECT_ROOT)

OUT_DIR = PROJECT_ROOT / "data" / "pq_subject_fit"
PARTS_DIR = OUT_DIR / "parts"
HIDE_FRAC = 0.20   # share of edges hidden as the test set
N_SPLITS = 3       # different random hidden sets per subject


def make_split(G: nx.Graph, rng: np.random.Generator):
    """Return (train_graph, hidden_edges, fake_pairs) for one random split."""
    forest = {tuple(sorted(e)) for e in nx.maximum_spanning_tree(G, weight="weight").edges()}
    edges = [tuple(sorted(e)) for e in G.edges()]
    hideable = [e for e in edges if e not in forest]
    n_hide = int(round(HIDE_FRAC * len(edges)))
    pick = rng.choice(len(hideable), size=n_hide, replace=False)
    hidden = [hideable[i] for i in pick]

    G_train = G.copy()
    G_train.remove_edges_from(hidden)

    active = sorted(n for n in G.nodes() if G.degree(n) > 0)
    fake = set()
    while len(fake) < n_hide:
        i, j = sorted(rng.choice(active, size=2, replace=False).tolist())
        if not G.has_edge(i, j):
            fake.add((i, j))

    # Leakage checks: nothing hidden is visible, fake pairs are truly unconnected.
    assert not any(G_train.has_edge(*e) for e in hidden)
    assert not any(G.has_edge(*e) for e in fake)
    return G_train, hidden, sorted(fake)


def link_auc(emb: np.ndarray, hidden, fake) -> float:
    """AUC of cosine similarity: hidden (real) edges vs fake pairs."""
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.where(norm == 0, 1.0, norm)
    pairs = np.array(hidden + fake)
    scores = np.sum(unit[pairs[:, 0]] * unit[pairs[:, 1]], axis=1)
    truth = np.r_[np.ones(len(hidden)), np.zeros(len(fake))]
    return float(roc_auc_score(truth, scores))


def fit_subject(method: str, subj_idx: int, sid: str) -> Path:
    """All 3 splits x 25 (p, q) for one subject and measure; saves a part file."""
    import warnings
    warnings.filterwarnings("ignore")
    from pq_embeddings import embed_graph_nodes   # loads gensim/node2vec in the worker

    out = PARTS_DIR / f"{method}__{sid.replace('/', '_')}.csv"
    if out.exists():
        return out

    group, subj = sid.split("/")
    G = build_nx_graph(np.load(CONN_DIR / method / group / f"{subj}.npy"))
    rows = []
    for split in range(N_SPLITS):
        rng = np.random.default_rng([SEED, METHODS.index(method), subj_idx, split])
        G_train, hidden, fake = make_split(G, rng)
        for p, q in PQ_GRID:
            emb = embed_graph_nodes(G_train, p, q)
            rows.append({"subject": sid, "group": group, "measure": method,
                         "p": p, "q": q, "split": split + 1,
                         "auc": link_auc(emb, hidden, fake),
                         "n_edges": G.number_of_edges(), "n_hidden": len(hidden)})
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)                            # only complete files count as done
    return out


def summarize(grid: pd.DataFrame) -> pd.DataFrame:
    """Per subject x measure: the winning (p, q), its AUC, AUC at (1, 1), margin."""
    rows = []
    mean_auc = grid.groupby(["measure", "subject", "group", "p", "q"], sort=False)["auc"].mean()
    for (method, sid, group), sub in mean_auc.groupby(level=[0, 1, 2], sort=False):
        scores = {(p, q): v for (_, _, _, p, q), v in sub.items()}
        best_auc = max(scores.values())
        tied = [pq for pq in PQ_GRID if np.isclose(scores[pq], best_auc)]
        p, q = min(tied, key=baseline_distance)
        base = scores[DEEPWALK_BASELINE]
        rows.append({"subject": sid, "group": group, "measure": method,
                     "best_p": p, "best_q": q, "best_auc": best_auc,
                     "auc_at_1_1": base, "margin": best_auc - base})
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    methods = [m for m in sys.argv[1:] if m in METHODS] or list(METHODS)
    ids = json.loads((EMB_DIR / "ids.json").read_text())
    excluded = nan_subject_ids()
    subjects = [(i, sid) for i, sid in enumerate(ids) if sid not in excluded]
    tasks = [(m, i, sid) for m in methods for i, sid in subjects]
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    n_jobs = os.cpu_count() or 1
    todo = [t for t in tasks
            if not (PARTS_DIR / f"{t[0]}__{t[2].replace('/', '_')}.csv").exists()]
    print("Item 2: per-subject (p, q) by link prediction")
    print(f"subjects={len(subjects)}  measures={methods}  splits={N_SPLITS}  "
          f"hide={HIDE_FRAC:.0%}  grid={len(PQ_GRID)} (p,q)")
    print(f"tasks: {len(tasks)} subject x measure ({len(tasks) - len(todo)} already done), "
          f"{len(todo) * N_SPLITS * len(PQ_GRID)} node2vec runs to go, {n_jobs} cores\n")

    t0 = time.time()
    if todo:
        # Time one subject first, then estimate the rest.
        fit_subject(*todo[0])
        dt = time.time() - t0
        rest = todo[1:]
        est = dt * len(rest) / n_jobs
        print(f"first subject took {dt / 60:.1f} min -> estimated {est / 60:.0f} min "
              f"for the remaining {len(rest)} (on {n_jobs} cores)\n", flush=True)
        done = 1
        for _ in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(fit_subject)(*t) for t in rest):
            done += 1
            if done % 5 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)} done  ({el / 60:.1f} min elapsed, "
                      f"~{el / done * (len(todo) - done) / 60:.0f} min left)", flush=True)

    grid = pd.concat([pd.read_csv(PARTS_DIR / f"{m}__{sid.replace('/', '_')}.csv")
                      for m, _, sid in tasks], ignore_index=True)
    best = summarize(grid)
    grid.round(4).to_csv(OUT_DIR / "subject_auc_grid.csv", index=False)
    best.round(4).to_csv(OUT_DIR / "subject_pq.csv", index=False)

    print("\n---- Summary per measure ----")
    for method, sub in best.groupby("measure", sort=False):
        top = sub.groupby(["best_p", "best_q"]).size().sort_values(ascending=False).head(3)
        top_str = ", ".join(f"({fmt(p)},{fmt(q)}) x{n}" for (p, q), n in top.items())
        n11 = int(((sub.best_p == 1) & (sub.best_q == 1)).sum())
        print(f"  {method:8s} picked (1,1): {n11}/{len(sub)}   most common: {top_str}")
        print(f"  {'':8s} best AUC {sub.best_auc.min():.3f}-{sub.best_auc.max():.3f}   "
              f"AUC at (1,1) {sub.auc_at_1_1.min():.3f}-{sub.auc_at_1_1.max():.3f}   "
              f"margin mean {sub.margin.mean():.4f} (max {sub.margin.max():.4f})")
        if sub.auc_at_1_1.min() <= 0.55:
            print(f"  {'':8s} WARNING: some subjects are near chance (AUC <= 0.55) - "
                  f"link prediction may not be working for them")

    split_sd = grid.groupby(["measure", "subject", "p", "q"])["auc"].std().mean()
    print(f"\naverage spread of AUC across the {N_SPLITS} splits: {split_sd:.4f}")
    print(f"Saved -> {(OUT_DIR / 'subject_pq.csv').relative_to(PROJECT_ROOT)}, "
          f"{(OUT_DIR / 'subject_auc_grid.csv').relative_to(PROJECT_ROOT)}  "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
