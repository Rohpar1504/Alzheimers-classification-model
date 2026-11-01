#!/usr/bin/env python3
"""
p/q investigation - Step 1: node2vec's transition-probability matrix.

This implements, computes, and illustrates the object that p and q actually control:
node2vec's SECOND-ORDER random-walk transition probabilities.

HOW node2vec computes transition probabilities
----------------------------------------------
A node2vec walk is second-order: the next step depends on the current node AND the
previous one. Suppose the walk just moved  t -> v  (previous = t, current = v). For each
neighbour x of v, the UNNORMALISED probability of stepping v -> x is

        pi(v, x) = alpha_pq(t, x) * w(v, x)

where w(v, x) is the edge weight and the bias factor alpha depends on how far x is from
the PREVIOUS node t:

        d_tx = 0  (x == t, i.e. stepping back)        -> alpha = 1/p
        d_tx = 1  (x is also a neighbour of t)         -> alpha = 1
        d_tx = 2  (x is farther; not adjacent to t)    -> alpha = 1/q

Normalising pi over all neighbours x of v gives the transition distribution.

Intuition (ties to the advisor's "p = local, q = global"):
    - small p  -> 1/p large  -> more mass on RETURNING to t         (very local, cautious)
    - large q  -> 1/q small  -> less mass on venturing FARTHER       (local / BFS / motifs)
    - small q  -> 1/q large  -> more mass on venturing FARTHER       (global / DFS / communities)
At p = q = 1 every alpha = 1, so pi(v,x) = w(v,x): the walk is unbiased = DeepWalk.

What this script produces
-------------------------
    1. `transition_distribution(G, t, v, p, q)` - the normalised next-step distribution.
    2. `edge_transition_matrix(G, p, q)` - THE transition-probability matrix of the walk,
       expressed as a first-order Markov chain over DIRECTED EDGES (the standard way to
       write a 2nd-order walk). Rows sum to 1; we verify this.
    3. A figure showing how the probability mass splits between {return, stay-local,
       venture-far} as p and q change, for a representative subject.

Usage:
    .venv/bin/python src/pq_transition.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
import scipy.sparse as sp

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from phase5b_node2vec_rf import build_nx_graph   # same graph node2vec actually runs on
from pq_grid import DEEPWALK_BASELINE

PROJECT_ROOT = SRC_DIR.parent
CONN_DIR = PROJECT_ROOT / "data" / "connectivity"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "pq_study"


def _distance_class(G: nx.Graph, t, x) -> int:
    """How far candidate x is from the previous node t: 0 (==t), 1 (neighbour), else 2."""
    if x == t:
        return 0
    if G.has_edge(t, x):
        return 1
    return 2


def alpha_pq(d_tx: int, p: float, q: float) -> float:
    """node2vec's search-bias factor for a candidate at distance d_tx from previous node."""
    if d_tx == 0:
        return 1.0 / p
    if d_tx == 1:
        return 1.0
    return 1.0 / q


def transition_distribution(G: nx.Graph, t, v, p: float, q: float):
    """
    Normalised next-step probabilities for a walk currently at v having come from t.
    Returns (neighbours, probs) where probs sums to 1 over v's neighbours.
    """
    nbrs = list(G.neighbors(v))
    unnorm = np.array([
        alpha_pq(_distance_class(G, t, x), p, q) * G[v][x]["weight"]
        for x in nbrs
    ])
    Z = unnorm.sum()
    probs = unnorm / Z if Z > 0 else unnorm
    return nbrs, probs


def edge_transition_matrix(G: nx.Graph, p: float, q: float):
    """
    THE node2vec transition-probability matrix, as a first-order Markov chain over the
    graph's DIRECTED edges. For directed edge e=(t,v) and e'=(v,x), entry P[e, e'] is the
    probability of stepping v->x given we arrived at v from t.

    Returns (P, edge_index) where P is a sparse (|E_dir| x |E_dir|) row-stochastic matrix
    and edge_index maps each directed edge (t,v) -> its row/column id.
    """
    directed_edges = [(u, w) for u, w in G.edges()] + [(w, u) for u, w in G.edges()]
    edge_index = {e: i for i, e in enumerate(directed_edges)}
    n = len(directed_edges)
    P = sp.lil_matrix((n, n))
    for (t, v), row in edge_index.items():
        nbrs, probs = transition_distribution(G, t, v, p, q)
        for x, pr in zip(nbrs, probs):
            P[row, edge_index[(v, x)]] = pr
    return P.tocsr(), edge_index


def mass_by_distance_class(G, t, v, p, q):
    """Split the next-step probability mass into {return(d=0), local(d=1), far(d=2)}."""
    nbrs, probs = transition_distribution(G, t, v, p, q)
    mass = {0: 0.0, 1: 0.0, 2: 0.0}
    for x, pr in zip(nbrs, probs):
        mass[_distance_class(G, t, x)] += pr
    return mass


def pick_focal_edge(G):
    """Pick a directed edge (t,v) where all three distance classes exist, for illustration."""
    for t, v in G.edges():
        nbrs = list(G.neighbors(v))
        classes = {_distance_class(G, t, x) for x in nbrs}
        if {0, 1, 2}.issubset(classes):
            return t, v
    # fallback: the edge whose endpoint has the highest degree
    v = max(G.nodes(), key=lambda n: G.degree(n))
    t = next(iter(G.neighbors(v)))
    return t, v


def main():
    warnings.filterwarnings("ignore")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Representative subject: first AD subject, Spearman connectivity (the best method).
    subj = sorted((CONN_DIR / "spearman" / "AD").glob("*.npy"))[0]
    print(f"Representative subject: AD/{subj.stem} (spearman)\n")
    G = build_nx_graph(np.load(subj))
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # (A) Compute THE transition matrix and verify it is row-stochastic.
    print("=== (A) Transition-probability matrix (edge-space Markov chain) ===")
    for (p, q) in [DEEPWALK_BASELINE, (0.25, 1.0), (1.0, 0.25)]:
        P, _ = edge_transition_matrix(G, p, q)
        rowsums = np.asarray(P.sum(axis=1)).ravel()
        active = rowsums[rowsums > 0]          # edges whose current node has neighbours
        print(f"  p={p}, q={q}: matrix {P.shape}, nonzeros={P.nnz}, "
              f"row-sums (active) min={active.min():.6f} max={active.max():.6f}  "
              f"-> {'stochastic OK' if np.allclose(active, 1.0) else 'NOT stochastic'}")

    # Sanity: at p=q=1 the transition must equal edge-weight-normalised (first-order).
    t, v = pick_focal_edge(G)
    nbrs, probs_11 = transition_distribution(G, t, v, 1.0, 1.0)
    w = np.array([G[v][x]["weight"] for x in nbrs]); w = w / w.sum()
    print(f"\n  Sanity at p=q=1: transition == edge-weight-normalised? "
          f"{np.allclose(probs_11, w)}  (confirms p=q=1 is DeepWalk)\n")

    # (B) Illustrate how p and q move the probability mass between regimes.
    print("=== (B) How p/q shift the next-step mass (focal edge "
          f"{t}->{v}) ===")
    settings = [(1.0, 1.0), (0.25, 1.0), (4.0, 1.0), (1.0, 0.25), (1.0, 4.0)]
    labels = ["p=q=1\n(DeepWalk)", "p=0.25\n(return-prone)", "p=4\n(return-averse)",
              "q=0.25\n(global/DFS)", "q=4\n(local/BFS)"]
    returns, locals_, fars = [], [], []
    for (p, q) in settings:
        m = mass_by_distance_class(G, t, v, p, q)
        returns.append(m[0]); locals_.append(m[1]); fars.append(m[2])
        print(f"  p={p}, q={q}: return={m[0]:.3f}  stay-local={m[1]:.3f}  venture-far={m[2]:.3f}")

    # Stacked bar figure.
    x = np.arange(len(settings))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, returns, label="return to previous node (d=0)", color="#d1495b")
    ax.bar(x, locals_, bottom=returns, label="stay local: neighbour of previous (d=1)", color="#edae49")
    ax.bar(x, fars, bottom=np.array(returns) + np.array(locals_),
           label="venture farther (d=2)", color="#30638e")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("share of next-step probability")
    ax.set_title(f"How p and q reshape the node2vec walk (subject AD/{subj.stem}, edge {t}->{v})")
    ax.legend(loc="upper right", fontsize=8)
    out = FIG_DIR / "transition_mass_by_pq.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
