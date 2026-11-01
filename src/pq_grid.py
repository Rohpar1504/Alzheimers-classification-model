#!/usr/bin/env python3
"""
p/q investigation - Step 0: the node2vec search grid (single source of truth).

node2vec's random walk is biased by two parameters:
    p  (return parameter)  - how strongly the walk backtracks to the previous node.
                             p < 1 => return-prone (stays local); p > 1 => return-averse.
    q  (in-out parameter)  - "local vs global" exploration (the advisor's framing).
                             q > 1 => BFS-like / local => structural roles & motifs.
                             q < 1 => DFS-like / global => communities / homophily.

Currently the pipeline fixes p = q = 1, which makes the walk UNBIASED - mathematically
identical to DeepWalk, so node2vec's distinctive biasing is unused. This grid lets us
actually study p and q.

Grid choice: p, q in {0.25, 0.5, 1, 2, 4} -> 25 combinations. This is the standard grid
from Grover & Leskovec (node2vec, KDD 2016) - powers of two centred on 1, symmetric on a
log scale, and it contains (1, 1) so the DeepWalk baseline sits inside the search. Every
later step of the investigation imports these values so the grid is defined in one place.
"""

from itertools import product

# The search values for each parameter (identical set for p and q).
P_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
Q_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]

# All 25 (p, q) combinations to evaluate.
PQ_GRID = [(p, q) for p, q in product(P_VALUES, Q_VALUES)]

# The current (unbiased / DeepWalk-equivalent) setting, kept as an explicit baseline.
DEEPWALK_BASELINE = (1.0, 1.0)


def describe_pq(p: float, q: float) -> str:
    """Human-readable regime label for a (p, q) setting (used in tables/figures)."""
    ret = "return-prone" if p < 1 else "return-averse" if p > 1 else "return-neutral"
    exp = "local/BFS (motifs)" if q > 1 else "global/DFS (communities)" if q < 1 else "neutral"
    if (p, q) == DEEPWALK_BASELINE:
        return "unbiased (DeepWalk)"
    return f"{ret}, {exp}"


if __name__ == "__main__":
    print(f"p/q grid: {len(PQ_GRID)} combinations "
          f"(p in {P_VALUES}, q in {Q_VALUES})\n")
    print(f"{'p':>5} {'q':>5}   regime")
    print("-" * 45)
    for p, q in PQ_GRID:
        print(f"{p:>5} {q:>5}   {describe_pq(p, q)}")
