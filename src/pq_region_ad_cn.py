#!/usr/bin/env python3
"""
p/q investigation - p and q per brain region, AD vs CN, with significance.

Covers the advisor's next step:
    "Create two plots: one for p and one for q; for each, consider the regions as
     categories. Partition each category between AD and CN subjects by colour coding
     the points."  +  "Give the significance levels in the table for p and q."

Reads each subject's per-region winning (p, q) from pq_region_fit.py (Spearman networks,
the top model's input) and makes:
    Main figures (one for p, one for q) - categories = 8 brain areas (AAL grouping).
        Each dot = one subject's typical p (or q) in that area: geometric mean of the
        winning values over the area's regions. AD filled orange / CN hollow blue,
        side by side; black bar = group geometric mean; * = significant after FDR.
    Supplementary figures - categories = all 116 regions. Each subject's winning value
        is discrete (0.25 ... 4), so per region the AD and CN subjects are shown as
        bubbles (area = number of subjects) on two rows, AD above CN.
    Table - for every area and every region, for p and for q: AD and CN geometric means,
        AD/CN ratio, permutation p-value (10,000 label shuffles, two-sided, difference
        in mean log2 value) and the FDR-corrected p-value (Benjamini-Hochberg),
        corrected separately within the area tests (16) and the region tests (232).
        Group-level only, no subject IDs.

Usage (needs pq_region_fit.py first):
    .venv/bin/python src/pq_region_ad_cn.py
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
from pq_ad_cn_scatter import GROUPS, TICK_LABELS, marker_style, FIG_DIR, RESULTS_DIR, PROJECT_ROOT
from pq_grid import P_VALUES
from pq_region_fit import OUT_DIR, AREAS

MEASURE = "spearman"
N_PERMUTATIONS = 10_000
ALPHA = 0.05
SEED = 0
PARAMS = [("p", "best_p"), ("q", "best_q")]
LOG_TICKS = np.log2(P_VALUES)


def main_fig_name(param: str) -> str:
    return f"{param} values per brain region - AD vs CN"


# ---- statistics ---------------------------------------------------------------------
def perm_pvalue(values: np.ndarray, is_ad: np.ndarray, rng) -> tuple[float, float]:
    """AD - CN difference in means and its two-sided permutation p-value (vectorised)."""
    real = values[is_ad].mean() - values[~is_ad].mean()
    perms = rng.permuted(np.tile(is_ad, (N_PERMUTATIONS, 1)), axis=1)
    n_ad = is_ad.sum()
    ad_mean = (perms * values).sum(axis=1) / n_ad
    cn_mean = (~perms * values).sum(axis=1) / (len(values) - n_ad)
    null = ad_mean - cn_mean
    return real, (1 + np.sum(np.abs(null) >= np.abs(real) - 1e-12)) / (1 + N_PERMUTATIONS)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def test_family(wide: dict, level: str, groups: pd.Series, rng) -> pd.DataFrame:
    """wide[param] = (subjects x categories) table of log2 values. One test per cell."""
    rows = []
    for param, table in wide.items():
        for cat in table.columns:
            col = table[cat].dropna()
            is_ad = (groups.loc[col.index] == "AD").to_numpy()
            v = col.to_numpy()
            row = {"level": level, "category": cat, "parameter": param,
                   "n_AD": int(is_ad.sum()), "n_CN": int((~is_ad).sum())}
            if is_ad.sum() >= 2 and (~is_ad).sum() >= 2:
                real, pval = perm_pvalue(v, is_ad, rng)
                row.update({"AD_geo_mean": 2 ** v[is_ad].mean(),
                            "CN_geo_mean": 2 ** v[~is_ad].mean(),
                            "AD_CN_ratio": 2 ** real, "p_value": pval})
            rows.append(row)
    df = pd.DataFrame(rows)
    ok = df.p_value.notna()
    df.loc[ok, "p_FDR"] = fdr_bh(df.loc[ok, "p_value"].to_numpy())
    df["significant_FDR"] = df.p_FDR < ALPHA
    return df


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---- figures ------------------------------------------------------------------------
def style_axis(ax):
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    ax.tick_params(length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main_figure(param: str, table: pd.DataFrame, groups: pd.Series, tests: pd.DataFrame,
                counts: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.09, 3.0))
    jit = np.random.default_rng(SEED)
    for x, area in enumerate(AREAS):
        for group, g in GROUPS.items():
            v = table.loc[groups == group, area].dropna().to_numpy()
            xs = x + g["dx"] * 1.3 + jit.uniform(-0.08, 0.08, len(v))
            ax.scatter(xs, v, s=16, zorder=3, **marker_style(group))
            ax.plot([x + g["dx"] * 1.3 - 0.13, x + g["dx"] * 1.3 + 0.13], [v.mean()] * 2,
                    color="0.15", lw=1.2, zorder=4)
        r = tests[(tests.category == area) & (tests.parameter == param)].iloc[0]
        if r.significant_FDR:
            ax.text(x, LOG_TICKS[-1] + 0.25, stars(r.p_FDR), ha="center", fontsize=10)
    ax.axhline(0, color="0.6", lw=0.7, ls="--", zorder=0)
    ax.set_xticks(range(len(AREAS)), [f"{a}\n({counts[a]})" for a in AREAS])
    ax.set_yticks(LOG_TICKS, TICK_LABELS)
    ax.set_ylim(LOG_TICKS[0] - 0.3, LOG_TICKS[-1] + 0.6)
    ax.set_xlim(-0.6, len(AREAS) - 0.4)
    ax.set_ylabel(f"{param} (area mean)")
    style_axis(ax)
    n = groups.value_counts()
    handles = [plt.scatter([], [], s=16, label=f"{g} (n = {n.get(g, 0)})", **marker_style(g))
               for g in GROUPS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False)
    fig.suptitle(main_fig_name(param), fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def supp_figure(param: str, col: str, best: pd.DataFrame, regions: pd.DataFrame,
                tests: pd.DataFrame) -> plt.Figure:
    """All 116 regions in two columns, grouped by area (a bold header row per area);
    AD and CN bubble rows per region."""
    regions = (regions.assign(order=regions.area.map(AREAS.index))
               .sort_values(["order", "column"]).reset_index(drop=True))
    items = []                                   # ("header", area) or ("region", row)
    for area in AREAS:
        items.append(("header", area))
        items += [("region", r) for r in regions[regions.area == area].itertuples()]
    half = (len(items) + 1) // 2
    while items[half][0] == "region" and items[half - 1][0] == "header":
        half += 1                                # never leave a header alone at the bottom
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 9.8))
    for ax, part in zip(axes, (items[:half], items[half:])):
        labels = []
        for row_i, (kind, r) in enumerate(part):
            y0 = -row_i
            if kind == "header":
                labels.append(r)
                if row_i:
                    ax.axhline(y0 + 0.5, color="0.75", lw=0.6)
                continue
            sub = best[best.region == r.column]
            for group, dy in (("AD", 0.17), ("CN", -0.17)):
                vals = sub.loc[sub.group == group, col].dropna()
                if len(vals):
                    c = vals.value_counts()
                    ax.scatter(np.log2(c.index.to_numpy()), np.full(len(c), y0 + dy),
                               s=c.to_numpy() * 7, zorder=3, **marker_style(group))
            t = tests[(tests.category == r.region_name) & (tests.parameter == param)]
            mark = stars(t.p_FDR.iloc[0]) if len(t) and t.significant_FDR.iloc[0] else ""
            labels.append(f"{r.region_name} {mark}".strip())
        ax.set_yticks(-np.arange(len(part)), labels, fontsize=5.5)
        for tick, (kind, _) in zip(ax.get_yticklabels(), part):
            if kind == "header":
                tick.set_fontweight("bold")
                tick.set_fontsize(7)
        ax.set_ylim(-len(part) + 0.4, 0.6)
        ax.set_xticks(LOG_TICKS, TICK_LABELS)
        ax.set_xlim(LOG_TICKS[0] - 0.5, LOG_TICKS[-1] + 0.5)
        ax.axvline(0, color="0.6", lw=0.7, ls="--", zorder=0)
        ax.set_xlabel(param)
        ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
        ax.tick_params(length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.scatter([], [], s=3 * 7, label=f"{g} (bubble area = subjects)",
                           **marker_style(g)) for g in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{main_fig_name(param)} (all 116 regions)", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985), w_pad=2)
    return fig


def main() -> None:
    best = pd.read_csv(OUT_DIR / "region_pq.csv")
    best = best[best.measure == MEASURE]
    regions = (best[["region", "region_name", "area"]].drop_duplicates()
               .sort_values("region").rename(columns={"region": "column"}).reset_index(drop=True))
    groups = best.drop_duplicates("subject").set_index("subject").group.sort_index()
    rng = np.random.default_rng(SEED)

    # subjects x categories tables of log2 winning values
    area_wide, region_wide = {}, {}
    for param, col in PARAMS:
        lw = best.assign(v=np.log2(best[col]))
        area_wide[param] = lw.pivot_table(index="subject", columns="area", values="v",
                                          aggfunc="mean")[AREAS].loc[groups.index]
        region_wide[param] = lw.pivot_table(index="subject", columns="region_name",
                                            values="v", aggfunc="mean").loc[groups.index]
        region_wide[param] = region_wide[param].reindex(columns=regions.region_name)

    area_tests = test_family(area_wide, "area", groups, rng)
    region_tests = test_family(region_wide, "region", groups, rng)
    region_tests.insert(2, "area", region_tests.category.map(
        dict(zip(regions.region_name, regions.area))))
    tests = pd.concat([area_tests, region_tests], ignore_index=True)
    counts = regions.area.value_counts().to_dict()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for param, col in PARAMS:
        for fig, name in ((main_figure(param, area_wide[param], groups, area_tests, counts),
                           main_fig_name(param)),
                          (supp_figure(param, col, best, regions, region_tests),
                           f"{main_fig_name(param)} (all 116 regions)")):
            fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
            fig.savefig(FIG_DIR / f"{name}.pdf")
            plt.close(fig)
            saved.append(name)

    out_csv = RESULTS_DIR / "pq_region_ad_cn_tests.csv"
    tests.round(4).to_csv(out_csv, index=False)

    # ---- terminal summary ----
    n = groups.value_counts()
    print(f"p and q per brain region - AD vs CN ({MEASURE.title()} networks, "
          f"AD={n.get('AD', 0)}, CN={n.get('CN', 0)})")
    print(f"{N_PERMUTATIONS} label shuffles per test; FDR (Benjamini-Hochberg) within "
          f"areas ({len(area_tests)} tests) and within regions ({len(region_tests)} tests)\n")
    print(f"  {'area':<12} {'param':>5} {'AD':>6} {'CN':>6} {'AD/CN':>6} {'p-value':>8} "
          f"{'p (FDR)':>8}  significant")
    for r in area_tests.itertuples():
        print(f"  {r.category:<12} {r.parameter:>5} {r.AD_geo_mean:>6.2f} {r.CN_geo_mean:>6.2f} "
              f"{r.AD_CN_ratio:>6.2f} {r.p_value:>8.4f} {r.p_FDR:>8.4f}  "
              f"{'YES' if r.significant_FDR else 'no'}")
    print("\n  regions, smallest p-values:")
    for r in region_tests.sort_values("p_value").head(8).itertuples():
        print(f"  {r.category:<20} {r.parameter:>2}  AD {r.AD_geo_mean:.2f}  CN {r.CN_geo_mean:.2f}"
              f"  p={r.p_value:.4f}  p(FDR)={r.p_FDR:.3f}  {'YES' if r.significant_FDR else 'no'}")
    print(f"  regions significant after FDR: {int(region_tests.significant_FDR.sum())} of "
          f"{int(region_tests.p_value.notna().sum())}")

    print("\nSuggested caption (main figures):")
    print("  Node2vec walk parameter p (or q) per brain area for subjects with AD (filled) and "
          "without AD (hollow), Spearman networks. Each subject's winning p (q) was found per "
          "AAL region by link prediction (held-out edges of that region; diagnosis not used), "
          "then averaged (geometric mean) over the regions of each area (number of regions in "
          "brackets). Black bars: group geometric means. Asterisks: AD vs CN difference "
          "significant after false-discovery-rate correction (two-sided permutation test, "
          f"{N_PERMUTATIONS:,} label shuffles).")
    print("\nSaved ->")
    for name in saved:
        print(f"  {(FIG_DIR / name).relative_to(PROJECT_ROOT)}.png / .pdf")
    print(f"  {out_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
