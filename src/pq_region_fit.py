#!/usr/bin/env python3
"""
p/q investigation - per-REGION (p, q) for every subject, by link prediction.

Covers the advisor's next step: "two plots, one for p and one for q, with the brain
regions as categories, AD vs CN colour-coded" (plots + tests: pq_region_ad_cn.py).

Item 2 (pq_subject_fit.py) gave each subject ONE (p, q) for the whole brain. Here each
subject gets a (p, q) for EACH of the 116 AAL regions, using the same idea:

For each subject (Spearman network, top 20% of |corr| as everywhere else):
    1. Protect a maximum spanning forest (never hidden, so no region is cut off).
       Shuffle the remaining edges and cut them into N_ROUNDS equal parts, so EVERY
       non-forest edge is hidden exactly once across the rounds (no wasted edges -
       each region needs as many test edges as possible).
    2. In each round, hide one part and run node2vec for each of the 25 (p, q) on the
       rest (same node2vec settings as every other run in the project).
    3. Score each region r separately: its hidden edges (real) vs all regions it is NOT
       connected to in the full network (fake), by cosine similarity of embeddings.
       Region AUC = chance a real hidden edge of r outscores a non-neighbour of r.
       Per-round AUCs are averaged, weighted by r's number of hidden edges that round.
    4. Region's winning (p, q) = highest AUC; ties toward (1, 1), as in Items 2-3.
       Regions with fewer than MIN_HIDDEN hidden edges in total get no winner (NaN).
Labels are never used; the group is only attached for plotting.

Outputs (contain ADNI subject IDs -> data/, git-ignored):
    data/pq_region_fit/region_pq.csv        one row per subject x region:
        best_p, best_q, best_auc, auc_at_1_1, n_hidden (+ region name and area)
    data/pq_region_fit/region_auc_grid.csv  every subject x region x (p, q)
    data/pq_region_fit/parts/               per-subject progress files (resumable)

Usage:
    .venv/bin/python src/pq_region_fit.py              # Spearman (default)
    .venv/bin/python src/pq_region_fit.py pearson      # another measure
Interrupted runs pick up where they stopped.
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

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from graph_utils import build_nx_graph
from pq_grid import PQ_GRID, DEEPWALK_BASELINE
from pq_nested_cv import (nan_subject_ids, baseline_distance, EMB_DIR, CONN_DIR,
                          METHODS, SEED, PROJECT_ROOT)

OUT_DIR = PROJECT_ROOT / "data" / "pq_region_fit"
PARTS_DIR = OUT_DIR / "parts"
LABELS_CSV = PROJECT_ROOT / "data" / "timeseries" / "aal_region_labels.csv"
N_ROUNDS = 5       # every non-forest edge hidden once across the 5 rounds
MIN_HIDDEN = 5     # regions with fewer hidden edges in total get no winner

# Standard AAL area grouping (Tzourio-Mazoyer et al., 2002), cerebellum incl. vermis.
AREAS = ["Frontal", "Temporal", "Parietal", "Occipital", "Limbic", "Insula",
         "Subcortical", "Cerebellum"]


def area_of(name: str) -> str:
    base = name.rsplit("_", 1)[0] if name.endswith(("_L", "_R")) else name
    if base.startswith(("Cerebelum", "Vermis")):
        return "Cerebellum"
    if base in ("Caudate", "Putamen", "Pallidum", "Thalamus"):
        return "Subcortical"
    if base == "Insula":
        return "Insula"
    if base.startswith("Cingulum") or base in ("Hippocampus", "ParaHippocampal", "Amygdala"):
        return "Limbic"
    if base.startswith(("Temporal", "Heschl")):
        return "Temporal"
    if base.startswith(("Occipital", "Calcarine", "Cuneus", "Lingual", "Fusiform")):
        return "Occipital"
    if base in ("Postcentral", "Parietal_Sup", "Parietal_Inf", "SupraMarginal",
                "Angular", "Precuneus"):
        return "Parietal"
    return "Frontal"   # precentral, frontal gyri, SMA, olfactory, rectus, paracentral


def region_table() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV).sort_values("column").reset_index(drop=True)
    assert len(df) == 116, "expected 116 AAL regions"
    df["area"] = df.region_name.map(area_of)
    return df


def make_rounds(G: nx.Graph, rng: np.random.Generator):
    """Split the non-forest edges into N_ROUNDS disjoint hidden sets."""
    forest = {tuple(sorted(e)) for e in nx.maximum_spanning_tree(G, weight="weight").edges()}
    hideable = [tuple(sorted(e)) for e in G.edges() if tuple(sorted(e)) not in forest]
    order = rng.permutation(len(hideable))
    return [[hideable[i] for i in part] for part in np.array_split(order, N_ROUNDS)]


def pair_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg) + 0.5 P(tie), computed over all pairs."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def fit_subject(method: str, subj_idx: int, sid: str) -> Path:
    """All rounds x 25 (p, q) for one subject; per-region AUCs saved to a part file."""
    import warnings
    warnings.filterwarnings("ignore")
    from pq_embeddings import embed_graph_nodes

    out = PARTS_DIR / f"{method}__{sid.replace('/', '_')}.csv"
    if out.exists():
        return out

    group, subj = sid.split("/")
    G = build_nx_graph(np.load(CONN_DIR / method / group / f"{subj}.npy"))
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G, nodelist=range(n), weight=None) > 0
    active = adj.any(axis=1)
    # Fake pairs for region r: active regions not connected to r in the full network.
    non_nbrs = [np.flatnonzero(~adj[r] & active & (np.arange(n) != r)) for r in range(n)]

    rng = np.random.default_rng([SEED, METHODS.index(method), subj_idx, 99])
    rounds = make_rounds(G, rng)

    # sums of (AUC x n_hidden) and n_hidden, per (p, q) and region
    auc_sum = {pq: np.zeros(n) for pq in PQ_GRID}
    n_hid = np.zeros(n)
    for hidden in rounds:
        G_train = G.copy()
        G_train.remove_edges_from(hidden)
        partners = [[] for _ in range(n)]
        for i, j in hidden:
            partners[i].append(j)
            partners[j].append(i)
        counts = np.array([len(x) for x in partners])
        n_hid += counts
        for pq in PQ_GRID:
            emb = embed_graph_nodes(G_train, *pq)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            unit = emb / np.where(norm == 0, 1.0, norm)
            S = unit @ unit.T
            for r in range(n):
                if counts[r] and len(non_nbrs[r]):
                    auc_sum[pq][r] += counts[r] * pair_auc(S[r, partners[r]], S[r, non_nbrs[r]])

    rows = []
    for (p, q), s in auc_sum.items():
        for r in range(n):
            rows.append({"subject": sid, "group": group, "measure": method, "region": r,
                         "p": p, "q": q,
                         "auc": s[r] / n_hid[r] if n_hid[r] else np.nan,
                         "n_hidden": int(n_hid[r])})
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)
    return out


def summarize(grid: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Per subject x region: winning (p, q) (ties toward (1, 1)); NaN if too few edges."""
    rows = []
    for (method, sid, group, r), sub in grid.groupby(["measure", "subject", "group", "region"],
                                                     sort=False):
        n_hidden = int(sub.n_hidden.iloc[0])
        row = {"subject": sid, "group": group, "measure": method, "region": r,
               "region_name": regions.region_name[r], "area": regions.area[r],
               "n_hidden": n_hidden, "best_p": np.nan, "best_q": np.nan,
               "best_auc": np.nan, "auc_at_1_1": np.nan}
        if n_hidden >= MIN_HIDDEN:
            scores = {(p, q): a for p, q, a in sub[["p", "q", "auc"]].itertuples(index=False)}
            best_auc = max(scores.values())
            tied = [pq for pq in PQ_GRID if np.isclose(scores[pq], best_auc)]
            p, q = min(tied, key=baseline_distance)
            row.update({"best_p": p, "best_q": q, "best_auc": best_auc,
                        "auc_at_1_1": scores[DEEPWALK_BASELINE]})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    methods = [m for m in sys.argv[1:] if m in METHODS] or ["spearman"]
    regions = region_table()
    ids = json.loads((EMB_DIR / "ids.json").read_text())
    excluded = nan_subject_ids()
    subjects = [(i, sid) for i, sid in enumerate(ids) if sid not in excluded]
    tasks = [(m, i, sid) for m in methods for i, sid in subjects]
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    part = lambda t: PARTS_DIR / f"{t[0]}__{t[2].replace('/', '_')}.csv"
    todo = [t for t in tasks if not part(t).exists()]
    n_jobs = min(os.cpu_count() or 1, max(len(todo), 1))
    print("Per-region (p, q) by link prediction")
    print(f"subjects={len(subjects)}  measures={methods}  rounds={N_ROUNDS} (every edge hidden "
          f"once)  grid={len(PQ_GRID)} (p,q)  regions={len(regions)}")
    print(f"{len(todo)} of {len(tasks)} subjects to do -> "
          f"{len(todo) * N_ROUNDS * len(PQ_GRID)} node2vec runs on {n_jobs} cores\n", flush=True)

    t0 = time.time()
    done = 0
    for _ in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(fit_subject)(*t) for t in todo):
        done += 1
        print(f"  {done}/{len(todo)} subjects done  ({(time.time() - t0) / 60:.1f} min elapsed)",
              flush=True)

    grid = pd.concat([pd.read_csv(part(t)) for t in tasks], ignore_index=True)
    best = summarize(grid, regions)
    grid.round(4).to_csv(OUT_DIR / "region_auc_grid.csv", index=False)
    best.round(4).to_csv(OUT_DIR / "region_pq.csv", index=False)

    print("\n---- Summary ----")
    for method, sub in best.groupby("measure", sort=False):
        ok = sub.dropna(subset=["best_p"])
        top = ok.groupby(["best_p", "best_q"]).size().sort_values(ascending=False).head(4)
        top_str = ", ".join(f"({p:g},{q:g}) {100 * k / len(ok):.0f}%" for (p, q), k in top.items())
        print(f"  {method}: {len(ok)}/{len(sub)} subject-regions have a winner "
              f"(median {int(sub.n_hidden.median())} hidden edges per region)")
        print(f"  most common winners: {top_str}")
        print(f"  region AUC: median {ok.best_auc.median():.3f}  "
              f"(at (1,1): {ok.auc_at_1_1.median():.3f})")
    print(f"\nSaved -> {(OUT_DIR / 'region_pq.csv').relative_to(PROJECT_ROOT)}, "
          f"{(OUT_DIR / 'region_auc_grid.csv').relative_to(PROJECT_ROOT)}  "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
