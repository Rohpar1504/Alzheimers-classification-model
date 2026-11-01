# Does the Correlation Method Matter? Pearson vs. Spearman vs. Kendall Functional Connectivity for Alzheimer's Classification

**Author:** Rohan Pareek — Graduate Researcher, Siebel School of Computing and Data Science, UIUC<br>
**Advisor:** Prof. Pablo D. Robles-Granda

## Research Question
Does the choice of correlation method (Pearson vs. Spearman vs. Kendall) for building
brain functional-connectivity networks from fMRI meaningfully affect downstream
classification of Alzheimer's Disease (AD) vs. Cognitively Normal (CN) subjects?
The comparison is evaluated across several graph-based classification models.

## Dataset
ADNI (Alzheimer's Disease Neuroimaging Initiative) — **28 subjects, balanced 14 AD / 14 CN**,
all **ADNI-2 resting-state fMRI** (3T, TR ≈ 3 s, "Resting State fMRI" protocol).
Groups are **sex-mirrored** (each 8M/6F) and **age-matched** (AD ≈ 78.3, CN ≈ 78.6).
**No MCI or any other condition** — a clean binary split. Data begins as raw DICOM.
Each scan has **140 timepoints** (verified from the NIfTI shapes, 64 × 64 × 48 × 140).

**3 subjects have undefined (NaN) connectivity values** — `AD/013_S_5071`,
`AD/018_S_4733`, `CN/010_S_4442` — caused by brain regions with a flat (constant)
signal (3, 9 and 12 such regions respectively). Per the advisor's decision, these 3
are **excluded from the node2vec p/q study**, leaving **25 subjects (12 AD / 13 CN)**.
The Phase 6 table below still uses all 28 and will be rerun on the 25.

## Pipeline
| Phase | What it does | Script | Status |
|-------|--------------|--------|--------|
| 1 | Setup, data acquisition & organization | — (scaffolding) | ✅ |
| 2 | DICOM → NIfTI conversion (`dcm2niix`) | `src/phase2_dicom_to_nifti.py` | ✅ |
| 3 | AAL atlas parcellation → region time-series (`NiftiLabelsMasker`) | `src/phase3_extract_timeseries.py` | ✅ |
| — | QC: visualize atlas-on-brain alignment | `src/qc_atlas_overlay.py` | ✅ |
| 4 | Build 3 connectivity matrices/subject (Pearson/Spearman/Kendall) | `src/phase4_connectivity.py` | ✅ |
| 5 | Graph construction + **GCN** classifier | `src/phase5_gcn.py` | ✅ |
| 5b | **node2vec** embeddings + Random Forest classifier | `src/phase5b_node2vec_rf.py` | ✅ |
| 5c | **TopER** (topology-inspired graph embedding) + Random Forest classifier | `src/phase5c_toper.py` | ✅ |
| 5d | **GGVec** (nodevectors) embeddings + Random Forest classifier | `src/phase5d_nodevectors.py` | ✅ |
| 5e | **GEE** (Graph Encoder Embedding) + Random Forest classifier | `src/phase5e_gee.py` | ✅ |
| 5f | **Brain Network Transformer** (Transformer + learned clustering) | `src/phase5f_brain_transformer.py` | ✅ |
| 6 | 4-fold CV comparison: models × methods × metrics | `src/phase6_model_comparison.py` | ✅ |
| 7 | Interpret important regions/edges per method | (planned) | ⏭️ |
| 8 | Write-up | `outputs/` | ⏭️ |
| p/q | **node2vec p/q study** — how node2vec's walk parameters behave (see section below) | `src/pq_*.py` | 🔄 Steps 0–5, 8 done |

Each script is **independently runnable** and reads from the previous phase's output folder.
Run scripts with the project venv, e.g. `.venv/bin/python src/phase4_connectivity.py`.

## Models Compared
Six families of graph-based models classify each subject's connectivity network:

- **GCN** (`phase5_gcn.py`) — a Graph Convolutional Network + MLP head that learns
  directly from the graph structure. *Its earlier "collapse" to predicting one class
  was caused by NaN values in 3 subjects, not by the model — see Findings.*
- **node2vec + classifier** (`phase5b`, `phase6`) — node2vec embeds each graph's
  structure (unsupervised), mean-pooled into a per-subject vector, then classified by
  **Random Forest**, **MLP**, or **Logistic Regression** (a sigmoid shallow classifier).
  Settings: 64 dimensions, walk length 20, 200 walks per node, window 10, and
  **p = q = 1** (an unbiased walk, equivalent to DeepWalk). This is the only model here
  that uses random walks, so it is the subject of the p/q study below.
- **TopER + classifier** (`phase5c`, `phase6`) — TopER (Topological Evolution Rate;
  Tola et al., NeurIPS 2025) is a graph-*level* topological embedding: for each
  filtration function (`deg_cen` degree centrality, `fricci` Forman-Ricci curvature —
  the latter uses edge weight = `|correlation|`) it fits a line across thresholds and
  returns just `[intercept, slope]` per sub/super-level, giving an 8-dimensional,
  low-dimensional, interpretable feature vector per subject — no pooling needed, unlike
  node2vec. Classified the same way: **Random Forest**, **MLP**, or **Logistic
  Regression**.
- **GGVec + classifier** (`phase5d`, `phase6` via `ggvec_bridge.py`) — GGVec, from the
  `nodevectors` package (Ranger), embeds each node by directly minimizing a loss over
  edge weights via alternating relaxation/contraction passes — no random walks,
  closer in spirit to GloVe than to node2vec's skip-gram approach. Mean-pooled into a
  per-subject vector like node2vec, classified by **Random Forest**, **MLP**, or
  **Logistic Regression**. **Runs in a separate virtual environment**
  (`~/.venvs/nodevectors` — see Setup) because its `csrgraph` dependency hard-pins
  `networkx==2.5.1`, incompatible with node2vec's and TopER's `networkx>=3`
  requirement — `phase6` bridges it in via a subprocess call to `ggvec_bridge.py`
  rather than importing it directly, so it still appears in the unified comparison
  table. `phase5d_nodevectors.py` also runs it standalone (leave-one-out CV) if you
  just want GGVec's numbers on their own.
- **GEE + classifier** (`phase5e`, `phase6`) — Graph Encoder Embedding (Shen, Wang &
  Priebe, IEEE TPAMI 2023; actively extended in 2024-2025 follow-up papers) is a
  fast, deterministic, linear embedding - a mechanism different from every other
  model here (not walk-based, not filtration-based, not a neural net). Unlike the
  other unsupervised embeddings, GEE requires a per-node CLASS LABEL as input: we
  use each AAL region's **hemisphere** (Left/Right/Midline), derived directly and
  unambiguously from the region name suffix in `aal_region_labels.csv` - a real,
  fixed anatomical property already in the data (54 L / 54 R / 8 Vermis-midline),
  not invented or derived from the correlation matrices. Each node's embedding
  dimension *k* = its total edge weight to every node in hemisphere *k*, giving a
  3-dimensional, directly interpretable feature per region ("how connected is this
  region to left/right/midline structures") - mean-pooled into a per-subject vector
  and classified the same way as the others.
- **Brain Network Transformer (BNT)** (`phase5f`, `phase6`) — Kan, Dai, Cui, Zhang,
  Guo & Yang, NeurIPS 2022. A Transformer applied directly to the connectivity
  matrix: each region's own row (its full "connection profile") is its feature
  vector; self-attention learns pairwise region relationships; a learned
  "Orthonormal Clustering Readout" (soft clustering via the DEC algorithm) pools
  regions into functional-module-like clusters before an MLP head classifies AD vs
  CN. Like the GCN, this is **supervised, end-to-end** (not an unsupervised
  embedding + shallow classifier), trained fresh within each CV fold. Uses the
  unmodified official code, cloned read-only into
  `reference/BrainNetworkTransformer/` (same convention as `reference/SGCN/`).
  **Note:** since the whole point of "connection profile as node feature" is the
  *complete* profile, BNT uses the RAW connectivity matrix, not the top-20%
  thresholded graph the other 5 models share - an inherent property of its
  full-attention design, not a shortcut (documented in `phase5f`'s docstring).
  **PyTorch compatibility:** their 2022 code predates a PyTorch 2.0 API change
  (`TransformerEncoderLayer` now passes an `is_causal` argument their code doesn't
  expect) - no installable PyTorch version avoids this. Per the advisor's explicit
  decision, a minimal compatibility shim is applied at runtime (their file on disk
  is untouched; only the call signature is widened to discard the new argument -
  see `phase5f_brain_transformer.py`'s docstring for the full accounting).

All models see identical graphs (top-20% strongest edges by `|correlation|`) so any
performance difference reflects the model or the connectivity method, not preprocessing.

## Current Status & Findings (indicative, n = 28)
Phase 6 evaluates every model × method — **14 models × 3 methods = 42 rows** (GCN,
node2vec, TopER, GGVec, GEE, and now the Brain Network Transformer) — with
**4-fold stratified cross-validation**, reporting accuracy, sensitivity,
specificity, and F1, each as **mean +/- std dev across the 4 folds**
(`outputs/results/model_comparison.csv`).

- **Best result overall is node2vec + MLP on Spearman** — accuracy **0.857 +/- 0.117**,
  F1 **0.839 +/- 0.141**. This is the most credible finding: it has held up (and
  slightly improved from the original 0.821) across every rerun, including after
  adding TopER, fixing a graph-construction bug (below), and adding GGVec, GEE, and
  the Brain Network Transformer.
- **The GCN and Brain Network Transformer "collapse" is a NaN artifact, not a model
  result.** Both report accuracy **0.500 +/- 0.082** and sensitivity **0.000** (every
  subject predicted CN) in `model_comparison.csv`. The cause: the 3 NaN subjects (see
  Dataset). `phase5_gcn.build_graph` uses `np.quantile` and keeps the raw NaN values,
  so the NaN spreads into the model's weights in every fold and it can only output
  class 0. **Confirmed for the GCN:** with NaN set to 0 it no longer collapses, though it
  is still weak (accuracy 0.250 / 0.357 / 0.357 on Pearson / Spearman / Kendall). **BNT
  is very likely the same** (it reads the raw matrix, NaN included) but has not been
  rerun yet. The earlier conclusion that "supervised deep nets fail at this sample size"
  is therefore withdrawn until both are rerun on the 25 clean subjects.
- **TopER's best results tie at accuracy 0.500** (TopER + MLP on Spearman and on
  Kendall; TopER + LogisticReg on Pearson) — meaningfully weaker than node2vec here,
  and fully reproducible (TopER has no randomness). TopER does **not** reproduce
  node2vec's Spearman signal, reinforcing that the "Spearman helps" effect is tied
  to specific representations, not a method-independent property of the data.
- **GEE's best result is GEE + RandomForest on Kendall** — accuracy **0.536 +/- 0.137**,
  F1 **0.371 +/- 0.307**. Also fully reproducible (GEE has no randomness). Its
  3-dimensional hemisphere-based embedding is the lowest-dimensional and most
  directly interpretable of any model here, but also among the weakest signals
  overall - a reasonable outcome given hemisphere alone is a coarse structural
  summary compared to the other methods' richer representations.
- **GGVec's best result under 4-fold CV is GGVec + LogisticReg on Pearson** —
  accuracy **0.429 +/- 0.000** this run. **GGVec's own class has no fixed random
  seed** (confirmed from its source - `nodevectors.GGVec` takes no
  `random_state`/`seed` parameter), so its embeddings - and every downstream number
  - genuinely change between runs on identical code and data: this same "best GGVec
  result" has read 0.571-0.79 in earlier reruns of this exact pipeline. Treat any
  single GGVec number as illustrative only; a multi-run average would be the honest
  way to report it, and is worth doing before this goes in any write-up.
- **Data-quality bug found and fixed (2026):** `graph_utils.build_nx_graph` (the graph
  construction shared by node2vec/TopER/GGVec/GEE) computed its edge threshold with
  `np.quantile`, which silently returns `NaN` when its input contains `NaN` values.
  Three subjects (`CN/010_S_4442`, `AD/013_S_5071`, `AD/018_S_4733`) have `NaN` entries
  in their connectivity matrices themselves (undefined correlation for some region
  pair, likely a near-constant region time-series from Phase 3) across **all three**
  correlation methods — every edge then silently failed the (NaN) threshold check,
  producing a completely edgeless graph for those subjects. node2vec and TopER
  degraded silently (this is what caused TopER's earlier-reported NaN-fit issue);
  GGVec's dependency crashed outright on the empty edge list, which is how this was
  actually caught. Fixed with `np.nanquantile` + explicitly skipping NaN-valued pairs
  (see `graph_utils.py`); all `phase6` numbers in this README reflect the fix. This fix
  did **not** reach the GCN (`phase5_gcn.py` builds its own graph) or BNT (raw matrix),
  which is what caused their collapse above. The NaN values come from regions with a
  flat signal; the advisor decided to exclude these 3 subjects (25 remain).

> ⚠️ **Caveat:** at 28 subjects these numbers remain **indicative, not conclusive**
> (small folds, single random seed, multiple combinations searched). Expanding the cohort
> from 16 → 28, and separately fixing the graph-construction bug above, both acted as
> robustness checks and showed most single-run "signal" was noise — node2vec + MLP +
> Spearman is the one result that has persisted through every change. A multi-seed
> robustness check and further data collection are planned.

**To do for Phase 6:** rerun on the 25 clean subjects (needs `aal_region_labels.csv`
restored, plus `omegaconf` and `reference/BrainNetworkTransformer/` for BNT); add a
fail-loud NaN check in Phase 4; average GGVec over several runs.

## node2vec p/q Study (in progress)
node2vec has two walk parameters. **p** controls stepping back: below 1 the walk often
returns to the node it just left; above 1 it avoids returning. **q** controls how far it
goes: below 1 the walk heads outward across the network (global, DFS-like); above 1 it
stays near where it started (local, BFS-like). **p = q = 1 is neutral** (DeepWalk). This
study asks how p and q behave on brain networks and whether tuning them helps classify
AD vs CN. All steps use the **25 clean subjects** and node2vec + MLP.

| Step | What it does | Script | Status |
|------|--------------|--------|--------|
| 0 | Define the grid: p, q ∈ {0.25, 0.5, 1, 2, 4} → 25 pairs (Grover & Leskovec 2016) | `src/pq_grid.py` | ✅ |
| 1 | Compute node2vec's second-order transition probabilities | `src/pq_transition.py` | ✅ (figure not yet saved) |
| 2 | Cache embeddings for every subject × 25 pairs × 3 measures (75 sets) | `src/pq_embeddings.py` | ✅ |
| 3 | Nested CV: choose p/q per fold using training subjects only, then test | `src/pq_nested_cv.py` | ✅ |
| 4 | Mean ± std table of chosen p and q (arithmetic and geometric means) | `src/pq_selection_summary.py` | ✅ |
| 5 | Data-leakage checks + permutation test | `src/pq_permutation_test.py` | ✅ |
| 8 | Histogram and scatter plot of chosen p/q | `src/pq_selection_summary.py` | ✅ |
| 6, 7 | AD vs CN p/q; link q to motifs; transition matrix on the Spearman network | (planned) | ⏭️ |

**Method:** 4-fold stratified CV repeated 25 times (100 outer folds per measure). In
each fold, p/q is chosen by an inner 3-fold CV on the training subjects only, then
tested once on the held-out subjects. Ties are broken toward (1, 1). The p = q = 1
baseline is scored on the same folds.

**Results:**

| Measure | Tuned p/q accuracy | p = q = 1 accuracy | Most chosen pair | Permutation p-value |
|---|---|---|---|---|
| Pearson | 0.525 ± 0.195 | 0.616 ± 0.186 | (1, 1) × 17 | 0.122 |
| **Spearman** | 0.575 ± 0.210 | **0.779 ± 0.196** | (1, 1) × 31 | **0.001** ✔ |
| Kendall | 0.432 ± 0.186 | 0.472 ± 0.167 | (0.25, 2) × 18 | 0.571 |

- **Tuning p and q does not help.** It overfits with ~19 training subjects per fold, and
  the default p = q = 1 is more accurate for every measure.
- **The chosen values are unstable.** They spread across the grid, and their geometric
  means stay near 1 (0.6–1.3) with a wide spread (about ×/÷ 2.5). Spearman is the most
  consistent, picking (1, 1) in 31 of 100 folds.
- **Only Spearman beats chance.** With p = q = 1, its accuracy (0.838 over 5 CV repeats)
  beat all 1,000 label shuffles (p = 0.001, significant at the Bonferroni level
  0.05 / 3). Shuffled-label accuracy centres at 0.50 for all three measures.
- **No data leakage.** All 6 checks pass: folds are disjoint; the chosen p/q does not
  change when test labels are flipped; embeddings never see labels; no duplicate rows.
- **Caveat:** Spearman already looked best on these same subjects, so its p-value is
  somewhat optimistic. Confirming it on new subjects is the strongest next check.

**Run it** (Step 2 must run first; Steps 3–5 take about 30, 1 and 15 minutes):
```bash
.venv/bin/python src/pq_embeddings.py          # Step 2: cache embeddings
.venv/bin/python src/pq_nested_cv.py           # Step 3: all 3 measures (a single measure overwrites the CSV)
.venv/bin/python src/pq_selection_summary.py   # Steps 4 and 8: table + figures
.venv/bin/python src/pq_permutation_test.py    # Step 5: leakage checks + permutation test
.venv/bin/python src/pq_transition.py          # Step 1: transition-probability figure
```
Outputs: `outputs/results/pq_*.csv` and `outputs/figures/pq_study/`.

**Next (agreed with the advisor):** plot p/q for AD vs CN subjects (histogram and
scatter); compare p/q between the top and bottom node2vec models; histogram of chosen
p/q before vs after label randomization; and build the transition-probability matrix
from p, q and the Spearman network.

## Folder Structure
```
.
├── data/                              # ALL subject data is git-ignored (ADNI DUA)
│   ├── raw/{AD,CN}/                    # raw DICOM series, sorted by diagnosis
│   ├── nifti/{AD,CN}/                  # converted .nii.gz + .json sidecars (Phase 2)
│   ├── timeseries/                     # (time_points x 116 regions) matrix per subject (Phase 3)
│   │   ├── {AD,CN}/                    #   one .npy per subject
│   │   └── aal_region_labels.csv       #   116 AAL region names (column order)
│   ├── connectivity/{pearson,spearman,kendall}/{AD,CN}/   # (116 x 116) matrices (Phase 4)
│   ├── pq_embeddings/{pearson,spearman,kendall}/  # cached p/q embeddings (git-ignored; Step 2)
│   ├── metadata/                       # ADNI per-scan XML metadata
│   └── subjects.csv                    # labels + demographics (MMSE/CDR/APOE/scanner)
├── src/                               # one script per phase (+ QC)
│   ├── graph_utils.py                  # shared graph construction (build_nx_graph)
│   ├── ggvec_bridge.py                 # subprocess bridge: nodevectors venv -> phase6
│   ├── phase5e_gee.py                  # GEE model (hemisphere labels) - Phase 5e
│   ├── phase5f_brain_transformer.py    # Brain Network Transformer - Phase 5f
│   └── pq_*.py                         # node2vec p/q study (Steps 0-5, 8)
├── outputs/
│   ├── figures/
│   │   ├── atlas_qc/                   # atlas-on-brain overlays (git-ignored: brain images)
│   │   ├── connectivity/               # per-subject connectivity heatmaps
│   │   └── pq_study/                   # p/q histogram, scatter, permutation-test figures
│   ├── models/                         # trained GCN (.pt) & RF (.joblib) — git-ignored
│   └── results/
│       ├── model_comparison.csv        # Phase 6 metrics table
│       └── pq_*.csv                    # p/q study results
├── reference/SGCN/                    # cloned reference repo — READ ONLY, git-ignored
├── reference/BrainNetworkTransformer/ # cloned reference repo — READ ONLY, git-ignored
├── requirements.txt
├── requirements-nodevectors.txt        # separate env for Phase 5d — see Setup
├── .gitignore
└── README.md
```

## Setup (macOS, Apple Silicon)
```bash
# 1. System tool for DICOM -> NIfTI conversion (Phase 2)
brew install dcm2niix

# 2. Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Python dependencies (nilearn, torch, torch-geometric, node2vec, toper, scikit-learn, ...)
pip install -r requirements.txt

# 4. Clone SGCN as a READ-ONLY architecture reference (not copied wholesale)
git clone https://github.com/Houliang-Zhou/SGCN.git reference/SGCN

# 5. Clone the Brain Network Transformer's official code (used UNMODIFIED - Phase 5f)
git clone https://github.com/Wayfear/BrainNetworkTransformer.git reference/BrainNetworkTransformer
```

**Note on Phase 5f (Brain Network Transformer):** their 2022 code predates a
PyTorch 2.0 API change (`TransformerEncoderLayer.forward()` now passes an
`is_causal` argument their `InterpretableTransformerEncoder._sa_block()` override
doesn't accept). No installable PyTorch version on any Python we could test avoids
this - checked PyPI (Python 3.11 and 3.12) and PyTorch's own wheel index, all only
serve torch>=2.0.0. Per the advisor's explicit decision (asked directly, given the
"use their code unmodified" instruction), `phase5f_brain_transformer.py` applies a
minimal runtime compatibility shim: their cloned file is never edited - at import
time, their original `_sa_block` function object is captured and wrapped in a new
outer function that accepts and discards the extra `is_causal` argument, then calls
their original, unchanged function body. See the script's docstring for the full
reasoning.

**Note on Phase 6:** loading `torch` (GCN), `gensim`/node2vec, and `networkit`
(TopER's `GraphRicciCurvature` dependency) in the same process can segfault on macOS
due to conflicting bundled OpenMP runtimes. If `phase6_model_comparison.py` crashes
with no traceback, run it with:
```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python src/phase6_model_comparison.py
```

**Note on Phase 5d (GGVec/nodevectors):** `nodevectors`' `csrgraph` dependency calls
`nx.adj_matrix()`, removed in networkx 3.x, so it hard-pins `networkx==2.5.1` -
incompatible with node2vec's and TopER's `networkx>=3` requirement (that old networkx
in turn needs `numpy<2.0`, which needs `scipy<1.14`). This genuinely cannot share an
environment with the rest of the project, so it needs its own venv - **required for
`phase6_model_comparison.py` too**, which shells out to it via `src/ggvec_bridge.py`:
```bash
python3 -m venv ~/.venvs/nodevectors
~/.venvs/nodevectors/bin/pip install -r requirements-nodevectors.txt

# Standalone GGVec-only results (leave-one-out CV):
~/.venvs/nodevectors/bin/python src/phase5d_nodevectors.py
```

## Attribution
GCN architecture and graph construction are **inspired by** the SGCN repository
(https://github.com/Houliang-Zhou/SGCN) and the IEEE paper *"Multi-Modal Diagnosis of
Alzheimer's Disease Using Interpretable Graph Convolutional Networks"*
(https://ieeexplore.ieee.org/document/10606492). node2vec and the p/q grid follow Grover
& Leskovec, *"node2vec: Scalable Feature Learning for Networks,"* KDD 2016
(https://arxiv.org/abs/1607.00653). TopER embeddings use the unmodified `toper` pip package
(https://github.com/AstritTola/TopER; Tola, Taiwo, Akcora & Coskunuzer, *"TopER:
Topological Embeddings in Graph Representation Learning,"* NeurIPS 2025). GGVec
embeddings use the unmodified `nodevectors` pip package
(https://github.com/VHRanger/nodevectors, Ranger). GEE embeddings use the
unmodified `gee` package (https://github.com/cshen6/GraphEmd; Shen, C., Wang, Q.,
Priebe, C.E., *"One-hot graph encoder embedding,"* IEEE TPAMI 45(6), 2023). The
Brain Network Transformer (Phase 5f) uses the unmodified official implementation
(https://github.com/Wayfear/BrainNetworkTransformer; Kan, X., Dai, W., Cui, H.,
Zhang, Z., Guo, Y., Yang, C., *"BrainNetworkTransformer,"* NeurIPS 2022), with only
a runtime PyTorch-version compatibility shim (see Setup) - not their algorithm.
All pipeline code is written from scratch; inspiration is credited inline.

## Data Ethics
ADNI data is used under its Data Use Agreement. **No subject data — raw scans, NIfTI,
time-series, connectivity matrices, metadata, `subjects.csv`, or brain-image figures — is
committed to version control** (see `.gitignore`). Only code, abstract result tables
and aggregate result figures (e.g. the p/q study plots) are tracked.
