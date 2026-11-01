#!/usr/bin/env python3
"""
p/q investigation - Step 2: precompute the node2vec embedding cache.

To study p and q we need each subject's node2vec embedding under EVERY (p, q) on the
grid (src/pq_grid.py), for every connectivity method. node2vec is the slow part, and
the later nested/repeated cross-validation (Step 3) reuses each embedding hundreds of
times - so we compute every embedding ONCE here and cache it to disk. Steps 3-4 (the
per-fold p/q selection, the mean/std table, the selection histogram) then just load
these cached features and run a fast classifier on top.

Controlled study:
    Everything is IDENTICAL to the existing node2vec model (phase5b_node2vec_rf.py) -
    same graph construction, same dimensions/walk-length/num-walks/window/seed/worker -
    and ONLY p and q vary. So any downstream difference is attributable to p and q,
    nothing else. The hyperparameters are imported from phase5b so they can never drift
    out of sync.

Classifier-agnostic:
    This step produces embeddings only. The choice of classifier (MLP, matching the
    paper's best node2vec result) happens in Step 3; nothing here depends on it.

Output layout (subject-derived features -> git-ignored, same as data/connectivity):
    data/pq_embeddings/ids.json              ordered list of "GROUP/subject" ids
    data/pq_embeddings/labels.npy            (n_subjects,) int labels, same order
    data/pq_embeddings/<method>/p<p>_q<q>.npy   (n_subjects x DIM) mean-pooled embeddings

Deterministic & resumable:
    Fixed seed; any (method, p, q) whose .npy already exists is skipped, so an
    interrupted run simply continues where it left off.

Usage:
    .venv/bin/python src/pq_embeddings.py            # all 3 methods (default)
    .venv/bin/python src/pq_embeddings.py spearman   # one or more methods by name
"""

import os

# Force single-threaded BLAS BEFORE numpy/gensim load. node2vec's gensim backend can
# intermittently deadlock (process stuck at 0% CPU) under multi-threaded BLAS on macOS.
# We already use workers=1 for reproducibility, so threading buys nothing here. These
# MUST be set before numpy is imported to take effect.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
from node2vec import Node2Vec

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# Reuse the EXACT same settings and graph construction as the deployed node2vec model,
# so this is a controlled study where only p and q change.
from phase5b_node2vec_rf import (
    DIM, WALK_LENGTH, NUM_WALKS, SEED, GROUPS, METHODS, CONN_DIR,
)
from graph_utils import build_nx_graph
from pq_grid import PQ_GRID

PROJECT_ROOT = SRC_DIR.parent
OUT_DIR = PROJECT_ROOT / "data" / "pq_embeddings"


def fmt(x: float) -> str:
    """Compact, filename-safe float: 0.25->'0.25', 1.0->'1', 2.0->'2'."""
    return f"{x:g}"


def embed_subject(mat: np.ndarray, p: float, q: float) -> np.ndarray:
    """
    node2vec on one subject's graph with the given (p, q); return the mean-pooled
    DIM-length feature vector. Identical to phase5b_node2vec_rf.embed_subject except
    that p and q are passed through to the biased walk (phase5b fixes p=q=1).
    """
    G = build_nx_graph(mat)
    n2v = Node2Vec(
        G,
        dimensions=DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=p,
        q=q,
        weight_key="weight",   # use edge weights in the random walks
        workers=1,             # single worker => reproducible with the seed
        seed=SEED,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, seed=SEED)

    emb = np.zeros((G.number_of_nodes(), DIM))
    for node in G.nodes():
        key = str(node)
        if key in model.wv:
            emb[node] = model.wv[key]
    return emb.mean(axis=0)


def ordered_subjects(method: str):
    """Return (ids, labels, paths) in a fixed, reproducible order for one method."""
    ids, labels, paths = [], [], []
    for group, label in GROUPS.items():
        for f in sorted((CONN_DIR / method / group).glob("*.npy")):
            ids.append(f"{group}/{f.stem}")
            labels.append(label)
            paths.append(f)
    return ids, labels, paths


def main() -> None:
    warnings.filterwarnings("ignore")
    methods = [m for m in sys.argv[1:] if m in METHODS] or list(METHODS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Establish subject order ONCE (shared across methods) and persist it.
    ref_ids, ref_labels, _ = ordered_subjects(methods[0])
    np.save(OUT_DIR / "labels.npy", np.array(ref_labels))
    (OUT_DIR / "ids.json").write_text(json.dumps(ref_ids, indent=2))
    n_subj = len(ref_ids)

    total = len(methods) * len(PQ_GRID)
    print(f"Step 2: node2vec embedding cache")
    print(f"methods={methods}  grid={len(PQ_GRID)} (p,q)  subjects={n_subj}")
    print(f"=> {total} (method,p,q) files, {total * n_subj} node2vec runs total")
    print(f"output: {OUT_DIR.relative_to(PROJECT_ROOT)}\n")

    done = 0
    t_start = time.time()
    for method in methods:
        ids, labels, paths = ordered_subjects(method)
        assert ids == ref_ids, f"subject order mismatch for {method}"
        mats = [np.load(pth) for pth in paths]          # load once, reuse per (p,q)
        (OUT_DIR / method).mkdir(parents=True, exist_ok=True)

        for (p, q) in PQ_GRID:
            done += 1
            out = OUT_DIR / method / f"p{fmt(p)}_q{fmt(q)}.npy"
            if out.exists():                            # resumable: skip finished work
                print(f"[{done:>3}/{total}] {method:8s} p={fmt(p):>4} q={fmt(q):>4}  "
                      f"skip (cached)")
                continue
            t0 = time.time()
            X = np.vstack([embed_subject(m, p, q) for m in mats])   # (n_subj, DIM)
            np.save(out, X)
            dt = time.time() - t0
            eta = (time.time() - t_start) / done * (total - done)
            print(f"[{done:>3}/{total}] {method:8s} p={fmt(p):>4} q={fmt(q):>4}  "
                  f"{X.shape}  {dt:5.1f}s  (eta {eta/60:4.1f} min)")

    print(f"\nDone. {total} files in {OUT_DIR.relative_to(PROJECT_ROOT)} "
          f"({(time.time()-t_start)/60:.1f} min total).")


if __name__ == "__main__":
    main()
