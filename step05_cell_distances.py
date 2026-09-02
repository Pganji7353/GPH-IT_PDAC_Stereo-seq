#!/usr/bin/env python3
"""
Compute nearest-neighbour spatial distances from mPDAC and ePDAC to all
other significantly enriched cell types within each ecotype, for each
treatment (Sham, IT, GPH, GPH+IT).

Significance: one-sided Fisher exact p < 0.05, pct > 0.01%, n >= 1.
Distances are in µm (Stereo-seq unit × 0.5).

Reads:
    downstream_analysis/processed_data/merged_annotated.h5ad
    downstream_analysis/processed_data/unified_ecotype_assignments.csv

Outputs:
    downstream_analysis/figures/cell_distances/
        ecotype_distances_<treatment>.csv      (per treatment)
        ecotype_distances_all_treatments.csv   (combined)
        distances_mPDAC_all_treatments.png/pdf  (mPDAC cross-treatment comparison)
        distances_ePDAC_all_treatments.png/pdf  (ePDAC cross-treatment comparison)
        superplot_pdac_distances.png/pdf        (summary super plot)

Usage:
    python step05_cell_distances.py --input_dir .
"""

import argparse
import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact

# ── Publication-quality settings ─────────────────────────────────────────────
import matplotlib.font_manager as _fm
_pref = next((f for f in ['Arial', 'Liberation Sans', 'FreeSans', 'DejaVu Sans']
              if f in {x.name for x in _fm.fontManager.ttflist}), 'sans-serif')
plt.rcParams.update({
    'font.family': _pref, 'font.size': 10,
    'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'legend.title_fontsize': 10,
    'axes.linewidth': 2.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150,
})

TREATMENT_ORDER  = ['Sham', 'IT', 'GPH', 'GPH+IT']
SOURCE_CELLTYPES = ['mPDAC', 'ePDAC']
COORD_TO_UM      = 0.5    # 1 Stereo-seq unit = 0.5 µm
MIN_PCT          = 0.01   # minimum % in ecotype for significance
MIN_COUNT        = 1      # minimum cell count in ecotype × cell_type
MIN_TGT_COUNT    = 10     # minimum target cells in ecotype for a distance row


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sort_ecotypes(lst):
    def key(s):
        m = re.search(r'\d+', s) if isinstance(s, str) else None
        return int(m.group()) if m else 999
    return sorted(lst, key=key)


def nearest_neighbour_distances(src_coords, tgt_coords):
    """Return NN distance from each src point to its closest tgt point."""
    if len(tgt_coords) == 0:
        return np.full(len(src_coords), np.nan)
    tree = cKDTree(tgt_coords.astype(float))
    d, _ = tree.query(src_coords.astype(float), k=1)
    return d


def significant_celltypes(ct_arr, eco_arr, ecotype):
    """
    One-sided Fisher exact (greater) p < 0.05, pct > MIN_PCT, count >= MIN_COUNT.
    Background = all cells in the treatment.
    Returns dict {cell_type: p_value} for all significant cell types.
    """
    in_eco = eco_arr == ecotype
    n_eco  = int(in_eco.sum())
    if n_eco == 0:
        return {}

    sig = {}
    for ct in sorted(set(ct_arr) - {'Unassigned', 'nan', ''}):
        in_ct = ct_arr == ct
        a = int(( in_eco &  in_ct).sum())
        b = int(( in_eco & ~in_ct).sum())
        c = int((~in_eco &  in_ct).sum())
        d = int((~in_eco & ~in_ct).sum())
        if a < MIN_COUNT or (a / n_eco * 100.0) <= MIN_PCT:
            continue
        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        if p < 0.05:
            sig[ct] = round(float(p), 6)
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# Distance computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_distances_for_treatment(adata, treatment):
    """Compute mPDAC/ePDAC → target distances for all ecotypes in one treatment."""
    mask_t   = adata.obs['treatment'].astype(str) == treatment
    adata_t  = adata[mask_t]
    if adata_t.n_obs == 0:
        return pd.DataFrame()

    ct_t  = adata_t.obs['cell_type_auto'].astype(str).values
    eco_t = adata_t.obs['ecotype'].astype(str).values

    ecotypes = sort_ecotypes([e for e in np.unique(eco_t) if e not in ('Unassigned', 'nan', '')])
    rows = []

    for ecotype in ecotypes:
        sig      = significant_celltypes(ct_t, eco_t, ecotype)
        pdac_sig = [s for s in SOURCE_CELLTYPES if s in sig]
        if not pdac_sig:
            continue

        # Spatial distances measured within the ecotype only
        mask_eco = (adata.obs['treatment'].astype(str) == treatment) & \
                   (adata.obs['ecotype'].astype(str) == ecotype)
        sub = adata[mask_eco]
        if sub.n_obs == 0:
            continue

        coords = sub.obsm['spatial'][:, :2].astype(float) * COORD_TO_UM
        ct_sub = sub.obs['cell_type_auto'].astype(str).values

        for src_ct in pdac_sig:
            src_mask = ct_sub == src_ct
            n_src    = int(src_mask.sum())
            if n_src == 0:
                continue
            src_coords = coords[src_mask]

            # Only TME targets: exclude the PDAC source types themselves
            tme_sig = [t for t in sig if t not in SOURCE_CELLTYPES]

            for tgt_ct in tme_sig:
                tgt_mask = ct_sub == tgt_ct
                n_tgt    = int(tgt_mask.sum())

                # Skip if target has too few cells in this ecotype for reliable distance
                if n_tgt < MIN_TGT_COUNT:
                    continue

                d     = nearest_neighbour_distances(src_coords, coords[tgt_mask])
                valid = d[~np.isnan(d)]
                rows.append(dict(
                    treatment      = treatment,
                    ecotype        = ecotype,
                    source         = src_ct,
                    source_pvalue  = sig[src_ct],
                    target         = tgt_ct,
                    target_pvalue  = sig[tgt_ct],
                    mean_dist      = round(float(np.mean(valid)),   4) if len(valid) else np.nan,
                    median_dist    = round(float(np.median(valid)), 4) if len(valid) else np.nan,
                    min_dist       = round(float(np.min(valid)),    4) if len(valid) else np.nan,
                    max_dist       = round(float(np.max(valid)),    4) if len(valid) else np.nan,
                    std_dist       = round(float(np.std(valid)),    4) if len(valid) else np.nan,
                    n_source       = n_src,
                    n_target       = n_tgt,
                ))

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Combined all-treatment comparison plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_treatments_combined(df_all, out_dir):
    """
    Creates one figure per source cell type (mPDAC / ePDAC).
    Each figure: 1 row × 4 columns (treatments).
    Each panel: Y = target cell type, X = median distance (µm).
    Dots = individual ecotypes (colored), grey line = mean across ecotypes.
    Dot size ∝ n_target.  Y-axis labels on left column only.
    All panels share the same X scale for direct cross-treatment comparison.
    Saved as distances_mPDAC_all_treatments.png/pdf and
              distances_ePDAC_all_treatments.png/pdf
    """
    df = df_all[df_all['median_dist'].notna()].copy()
    sources = [s for s in SOURCE_CELLTYPES if s in df['source'].values]
    treats  = [t for t in TREATMENT_ORDER   if t in df['treatment'].values]
    if not sources or not treats:
        return

    all_targets = sorted(df['target'].unique())
    n_tgt   = len(all_targets)
    tgt_idx = {t: i for i, t in enumerate(all_targets)}
    xmax    = df['median_dist'].quantile(0.97) * 1.1

    ecotypes = sort_ecotypes(df['ecotype'].unique().tolist())
    eco_cmap = matplotlib.colormaps.get_cmap('tab10').resampled(max(len(ecotypes), 10))
    eco_pal  = {e: eco_cmap(i) for i, e in enumerate(ecotypes)}
    trt_colors = {'Sham': '#4393c3', 'IT': '#74c476', 'GPH': '#fd8d3c', 'GPH+IT': '#d6604d'}

    n_cols = len(treats)

    # Two output configs: publication (2×2") and PPT (14×6")
    CONFIGS = [
        dict(
            fig_w=2.0, fig_h=2.0,
            stem_lw=1.2, dot_min=5, dot_max=22, dot_scale=0.15,
            spine_lw=0.4, grid_lw=0.4, axvline_lw=0.4, tick_len=2,
            title_fs=5, ytick_fs=4, ylabel_fs=4.5, xtick_fs=3.5,
            legend_fs=3.5, legend_title_fs=4, supxlabel_fs=4, suptitle_fs=5,
            title_pad=2, title_bbox_pad=0.2,
            left=0.28, bottom=0.30, wspace=0.08,
            dpi_png=300, suffix='_small',
        ),
        dict(
            fig_w=14.0, fig_h=6.5,
            stem_lw=3.5, dot_min=60, dot_max=300, dot_scale=1.5,
            spine_lw=1.0, grid_lw=0.5, axvline_lw=0.8, tick_len=4,
            title_fs=28, ytick_fs=23, ylabel_fs=26, xtick_fs=22,
            legend_fs=23, legend_title_fs=24, supxlabel_fs=26, suptitle_fs=28,
            # The treatment name is drawn in a rounded box; at pad=8 the box's
            # lower edge dropped into the axes and sat on the first row (qCAF).
            # Pad has to clear the bbox padding as well as the text itself.
            title_pad=30, title_bbox_pad=0.5,
            left=0.20, bottom=0.32, wspace=0.38,
            dpi_png=300, suffix='',
        ),
    ]

    for src in sources:
        for cfg in CONFIGS:
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(cfg['fig_w'], cfg['fig_h']),
                sharey=False, sharex=False,
            )
            axes = [axes] if n_cols == 1 else list(axes)

            for c, trt in enumerate(treats):
                ax  = axes[c]
                sub = df[(df['source'] == src) & (df['treatment'] == trt)]

                trt_xmax = max(sub['median_dist'].quantile(0.97) * 1.25, 20) if not sub.empty else 50

                for tgt in all_targets:
                    yi  = tgt_idx[tgt]
                    row = sub[sub['target'] == tgt]
                    if row.empty:
                        ax.axhline(yi, color='#f5f5f5', linewidth=cfg['grid_lw'],
                                   linestyle='--', zorder=0)
                        continue
                    mean_val = row['median_dist'].mean()
                    ax.plot([0, mean_val], [yi, yi],
                            color='#555555', linewidth=cfg['stem_lw'],
                            zorder=1, solid_capstyle='round')
                    for _, r2 in row.iterrows():
                        size = max(cfg['dot_min'], min(cfg['dot_max'],
                                                        r2['n_target'] * cfg['dot_scale']))
                        ax.scatter(r2['median_dist'], yi,
                                   color=eco_pal.get(r2['ecotype'], 'gray'),
                                   s=size, alpha=0.9, zorder=3,
                                   edgecolors='white', linewidth=cfg['spine_lw'] * 1.5)

                for yi in range(n_tgt):
                    ax.axhline(yi, color='#f0f0f0', linewidth=cfg['grid_lw'], zorder=0)

                ax.set_xlim(0, trt_xmax)
                ax.set_ylim(-0.7, n_tgt - 0.3)
                ax.axvline(0, color='#555555', linewidth=cfg['axvline_lw'])
                sns.despine(ax=ax)
                for spine in ax.spines.values():
                    spine.set_linewidth(cfg['spine_lw'])
                ax.tick_params(axis='both', width=cfg['spine_lw'],
                               length=cfg['tick_len'])

                ax.set_title(trt, fontsize=cfg['title_fs'],
                             color='white', pad=cfg['title_pad'],
                             bbox=dict(facecolor=trt_colors.get(trt, '#888888'),
                                       edgecolor='none',
                                       boxstyle=f"round,pad={cfg['title_bbox_pad']}"))

                ax.set_yticks(range(n_tgt))
                if c == 0:
                    ax.set_yticklabels(all_targets, fontsize=cfg['ytick_fs'])
                else:
                    ax.set_yticklabels([])
                _bold_lw = cfg['spine_lw'] * 2.5
                ax.spines['left'].set_visible(True)
                ax.spines['left'].set_linewidth(_bold_lw)
                ax.spines['bottom'].set_linewidth(_bold_lw)
                ax.tick_params(axis='y', pad=max(cfg['spine_lw'] * 4, 3))
                ax.set_ylabel('')
                ax.tick_params(axis='y', length=0, pad=max(cfg['spine_lw'] * 4, 3))

                ax.set_xlabel('')
                ax.tick_params(axis='x', labelsize=cfg['xtick_fs'], pad=2)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))

            # The mPDAC and ePDAC figures are stacked into a single composite
            # panel and carry the SAME ecotype palette, so only the lower one
            # draws the legend, two identical legends in one panel is pure
            # wasted area. `sources` order decides which is lower.
            is_last_src = (src == sources[-1])
            if is_last_src:
                handles = [mpatches.Patch(color=eco_pal[e], label=e)
                           for e in ecotypes]
                # No legend title: an "Ecotype" title here rises into the
                # "Median Distance (µm)" supxlabel just above it. The CC*
                # swatch labels are self-describing, so the title only costs a
                # collision.
                fig.legend(handles=handles,
                           loc='lower center', bbox_to_anchor=(0.5, 0.0),
                           ncol=min(len(ecotypes), 6), frameon=False,
                           fontsize=cfg['legend_fs'],
                           handlelength=0.8, handleheight=0.6,
                           columnspacing=0.5)

            # cfg['bottom'] reserves room for the legend strip; without a
            # legend most of that reservation is just white space, so pull it
            # in -- but not so far that the axis label meets the tick numbers.
            _bottom = cfg['bottom'] if is_last_src else cfg['bottom'] * 0.72
            # y=0.17 is tuned against bottom=0.32 (the legend case). On the
            # legend-less figure the margin shrinks, so the label has to come
            # down with it or it lands on top of the tick numbers.
            _supx_y = 0.17 if is_last_src else _bottom * 0.38
            fig.supxlabel('Median Distance (µm)', fontsize=cfg['supxlabel_fs'],
                          fontweight='normal', y=_supx_y)
            # suptitle defaults to va='top', so y is the TOP of the text and it
            # grows downward. At y=1.01 with axes top=0.88 the title ran down
            # into the coloured treatment badges (which sit just above the axes
            # and are ~title_fs tall). Drop the axes to make room for the badges
            # and lift the title so the two never meet.
            fig.suptitle(f'Spatial Distances from {src} to TME Cell Types',
                         fontsize=cfg['suptitle_fs'], y=1.04)
            fig.subplots_adjust(top=0.82, bottom=_bottom,
                                left=cfg['left'], right=0.99,
                                wspace=cfg['wspace'])

            src_tag = src.replace('+', '_plus_')
            suffix  = cfg['suffix']
            # The main (no-suffix) PNG is the curated source for fig7_b (via
            # stitch_vertical); the small (2x2") suffixed variant is not.
            if not suffix:
                plt.savefig(os.path.join(out_dir,
                            f'distances_{src_tag}_all_treatments{suffix}.png'),
                            dpi=cfg['dpi_png'], bbox_inches='tight', facecolor='white')
                plt.savefig(os.path.join(out_dir,
                            f'distances_{src_tag}_all_treatments.pdf'),
                            dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            label = 'small (2×2")' if suffix else 'large (PPT)'
            print(f"    Saved {label}: distances_{src_tag}_all_treatments{suffix}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Super plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_superplot(df_all, out_dir):
    """
    3-panel publication figure:
      A: Heatmap: mPDAC median distance (µm) per cell type × treatment
      B: Heatmap: ePDAC median distance (µm) per cell type × treatment
      C: Proximity bias bar: mPDAC_dist − ePDAC_dist averaged across treatments
          Red = cell type is closer to ePDAC; Blue = closer to mPDAC
    Each heatmap cell = weighted median (weight = n_target) across ecotypes.
    """
    df = df_all[df_all['median_dist'].notna()].copy()

    def weighted_median_dist(sub):
        if sub.empty:
            return np.nan
        w = sub['n_target'].values.astype(float)
        return float(np.average(sub['median_dist'].values, weights=w)) \
               if w.sum() > 0 else sub['median_dist'].mean()

    # Aggregate: source × treatment × target → single distance value
    records = []
    for src in SOURCE_CELLTYPES:
        for trt in TREATMENT_ORDER:
            sub_st = df[(df['source'] == src) & (df['treatment'] == trt)]
            for tgt in sub_st['target'].unique():
                records.append({
                    'source': src, 'treatment': trt, 'target': tgt,
                    'dist': weighted_median_dist(sub_st[sub_st['target'] == tgt]),
                })
    agg_df = pd.DataFrame(records)

    def make_pivot(src):
        sub = agg_df[agg_df['source'] == src]
        if sub.empty:
            return pd.DataFrame()
        return sub.pivot_table(index='target', columns='treatment',
                               values='dist', aggfunc='mean') \
                  .reindex(columns=TREATMENT_ORDER)

    piv_m = make_pivot('mPDAC')
    piv_e = make_pivot('ePDAC')

    all_targets = sorted(set(
        (piv_m.index.tolist() if not piv_m.empty else []) +
        (piv_e.index.tolist() if not piv_e.empty else [])
    ))
    if not all_targets:
        print("  WARNING: No aggregated data for super plot, skipping.")
        return

    piv_m = piv_m.reindex(all_targets) if not piv_m.empty else \
            pd.DataFrame(np.nan, index=all_targets, columns=TREATMENT_ORDER)
    piv_e = piv_e.reindex(all_targets) if not piv_e.empty else \
            pd.DataFrame(np.nan, index=all_targets, columns=TREATMENT_ORDER)

    # Panel C: mean delta across treatments (positive = farther from mPDAC, i.e. closer to ePDAC)
    # Aggregate each pivot independently first so cell types with non-overlapping
    # treatment coverage (mPDAC in Sham only, ePDAC in IT only, etc.) still produce a delta.
    m_mean = piv_m.mean(axis=1, skipna=True)
    e_mean = piv_e.mean(axis=1, skipna=True)
    delta = (m_mean - e_mean).dropna().sort_values()

    all_vals = np.concatenate([
        piv_m.values[~np.isnan(piv_m.values)],
        piv_e.values[~np.isnan(piv_e.values)],
    ])
    vmax = float(np.percentile(all_vals, 95)) if len(all_vals) else 1000.0

    n_tgt = len(all_targets)
    fig   = plt.figure(figsize=(20, max(8, n_tgt * 0.6 + 4)))
    gs    = fig.add_gridspec(1, 3, width_ratios=[3, 3, 2.5])
    ax_m, ax_e, ax_d = (fig.add_subplot(gs[i]) for i in range(3))

    hm_kw = dict(cmap='RdYlBu_r', vmin=0, vmax=vmax,
                 linewidths=0.5, linecolor='white',
                 cbar_kws={'shrink': 0.65, 'label': 'Median dist (µm)', 'pad': 0.02})

    # ── Panel A, mPDAC ───────────────────────────────────────────────────────
    if piv_m.notna().any().any():
        sns.heatmap(piv_m, ax=ax_m, **hm_kw)
        ax_m.figure.axes[-1].tick_params(labelsize=11)
        ax_m.figure.axes[-1].yaxis.label.set_size(11)
        ax_m.set_title('A.  mPDAC → Cell Type\nMedian Distance (µm)',
                       fontsize=20, pad=10)
        ax_m.set_xlabel('Treatment', fontsize=19)
        ax_m.set_ylabel('Target Cell Type', fontsize=19)
        plt.setp(ax_m.get_xticklabels(), rotation=30, ha='right', fontsize=16)
        plt.setp(ax_m.get_yticklabels(), rotation=0, fontsize=16)
    else:
        ax_m.text(0.5, 0.5, 'No mPDAC data', ha='center', va='center',
                  transform=ax_m.transAxes, fontsize=19)
        ax_m.set_title('A.  mPDAC → Cell Type Distance', fontsize=20)

    # ── Panel B, ePDAC ───────────────────────────────────────────────────────
    if piv_e.notna().any().any():
        sns.heatmap(piv_e, ax=ax_e, **hm_kw)
        ax_e.figure.axes[-1].tick_params(labelsize=11)
        ax_e.figure.axes[-1].yaxis.label.set_size(11)
        ax_e.set_title('B.  ePDAC → Cell Type\nMedian Distance (µm)',
                       fontsize=20, pad=10)
        ax_e.set_xlabel('Treatment', fontsize=19)
        ax_e.set_ylabel('')
        plt.setp(ax_e.get_xticklabels(), rotation=30, ha='right', fontsize=16)
        # Hide Panel B y-tick labels, same cell types as Panel A, avoids overlap
        ax_e.set_yticklabels([])
        ax_e.tick_params(axis='y', length=0)
    else:
        ax_e.text(0.5, 0.5, 'No ePDAC data', ha='center', va='center',
                  transform=ax_e.transAxes, fontsize=19)
        ax_e.set_title('B.  ePDAC → Cell Type Distance', fontsize=20)

    # ── Panel C, Proximity bias ──────────────────────────────────────────────
    if len(delta) > 0:
        colors = ['#d73027' if v > 0 else '#4575b4' for v in delta.values]
        ax_d.barh(range(len(delta)), delta.values,
                  color=colors, edgecolor='white', linewidth=0.5, height=0.7)
        ax_d.set_yticks(range(len(delta)))
        ax_d.set_yticklabels(delta.index, fontsize=16)
        ax_d.axvline(0, color='black', linewidth=1.5)
        ax_d.set_xlabel('mPDAC dist − ePDAC dist (µm)', fontsize=17)
        ax_d.set_title('C.  Proximity Bias\n● Red = closer to ePDAC\n● Blue = closer to mPDAC',
                       fontsize=19, pad=8)
        ax_d.plot([], [], 's', color='#d73027', markersize=10, label='Closer to ePDAC')
        ax_d.plot([], [], 's', color='#4575b4', markersize=10, label='Closer to mPDAC')
        ax_d.legend(loc='lower right', fontsize=16, frameon=False)
        # Force symmetric x-axis so absence of blue is visually unambiguous.
        xmax_abs = float(np.nanmax(np.abs(delta.values))) if len(delta) else 1.0
        ax_d.set_xlim(-xmax_abs * 1.1, xmax_abs * 1.1)
        sns.despine(ax=ax_d)
    else:
        ax_d.text(0.5, 0.5, 'Insufficient overlap\nbetween mPDAC and ePDAC\nfor comparison',
                  ha='center', va='center', transform=ax_d.transAxes, fontsize=19)
        ax_d.set_title('C.  Proximity Bias', fontsize=19)

    fig.suptitle('Spatial Proximity of mPDAC and ePDAC to TME Cell Types\n'
                 '(weighted median nearest-neighbour distance across ecotypes)',
                 fontsize=23)
    # subplots_adjust: tight_layout breaks with seaborn heatmap colorbars
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.12, right=0.97, wspace=0.75)

    # PNG at 150 dpi (viewable); PDF at 300 dpi (print)
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(out_dir, 'superplot_pdac_distances.png'),
    # dpi=150, bbox_inches='tight', facecolor='white')
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(out_dir, 'superplot_pdac_distances.pdf'),
    # dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("    Saved: superplot_pdac_distances.png/pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point (called standalone or from step2)
# ══════════════════════════════════════════════════════════════════════════════

def run_cell_distance_analysis(adata, out_dir):
    """
    Main analysis entry point. Can be called from step2 or as a standalone script.
    Requires adata.obs['ecotype'] to be populated before calling.
    """
    os.makedirs(out_dir, exist_ok=True)

    if 'ecotype' not in adata.obs.columns:
        print("  WARNING: 'ecotype' column not found. "
              "Run generate_ecotype_panels.py first.")
        return None

    n_assigned = (adata.obs['ecotype'] != 'Unassigned').sum()
    print(f"  Cells with ecotype: {n_assigned:,} / {adata.n_obs:,}")

    # Verify required cell type labels exist
    ct_present = set(adata.obs['cell_type_auto'].unique())
    found = [s for s in SOURCE_CELLTYPES if s in ct_present]
    if not found:
        print(f"  WARNING: Neither mPDAC nor ePDAC found in cell_type_auto. "
              f"Present types: {sorted(ct_present)}")
        return None
    print(f"  Source cell types found: {found}")

    all_dfs = []
    for treatment in TREATMENT_ORDER:
        print(f"  [{treatment}] computing distances...")
        df_t = compute_distances_for_treatment(adata, treatment)
        if df_t.empty:
            print(f"    No significant mPDAC/ePDAC ecotypes in {treatment}")
            continue

        tname    = treatment.replace('+', '_plus_')
        csv_path = os.path.join(out_dir, f'ecotype_distances_{tname}.csv')
        df_t.to_csv(csv_path, index=False)
        print(f"    {len(df_t)} rows  |  ecotypes: "
              f"{sort_ecotypes(df_t['ecotype'].unique().tolist())}")
        all_dfs.append(df_t)

    if not all_dfs:
        print("  No distance data produced.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, 'ecotype_distances_all_treatments.csv'), index=False)
    print(f"\n  Combined CSV: {len(df_all)} rows  → ecotype_distances_all_treatments.csv")

    print("  Generating combined treatment comparison plot...")
    plot_all_treatments_combined(df_all, out_dir)

    print("  Generating super plot...")
    plot_superplot(df_all, out_dir)

    return df_all


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Compute spatial distances from mPDAC/ePDAC to other '
                    'cell types within each ecotype.')
    parser.add_argument('--input_dir', default=os.path.expanduser('~/stereo-seq/stereoseq-analysis'),
                        help='WORK_DIR (same as used in SLURM scripts)')
    parser.add_argument('--ecotype_csv', default=None,
                        help='Path to unified_ecotype_assignments.csv '
                             '(auto-detected if not provided)')
    args = parser.parse_args()

    h5ad_path = os.path.join(args.input_dir, 'downstream_analysis',
                             'processed_data', 'merged_annotated.h5ad')
    eco_csv   = args.ecotype_csv or os.path.join(
        args.input_dir, 'downstream_analysis',
        'processed_data', 'unified_ecotype_assignments.csv')
    out_dir   = os.path.join(args.input_dir, 'downstream_analysis',
                             'figures', 'cell_distances')

    if not os.path.exists(h5ad_path):
        sys.exit(f"ERROR: {h5ad_path} not found. Run Step 2 first.")

    print("=" * 60)
    print("CELL DISTANCE ANALYSIS, mPDAC / ePDAC")
    print("=" * 60)
    print(f"Input:      {h5ad_path}")
    print(f"Ecotype CSV: {eco_csv}")
    print(f"Output:     {out_dir}\n")

    print("Loading merged_annotated.h5ad...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    # Load ecotype assignments
    if 'ecotype' not in adata.obs.columns:
        if not os.path.exists(eco_csv):
            sys.exit(
                f"ERROR: No 'ecotype' in h5ad and no ecotype CSV at:\n  {eco_csv}\n"
                "Run generate_ecotype_panels.py first."
            )
        print(f"Loading ecotype assignments from CSV...")
        eco_df = pd.read_csv(eco_csv, index_col=0)
        if 'ecotype' not in eco_df.columns:
            sys.exit("ERROR: 'ecotype' column not found in ecotype CSV.")

        # Diagnostics: print sample indices to detect mismatch
        print(f"  CSV rows: {len(eco_df):,}  |  h5ad cells: {adata.n_obs:,}")
        print(f"  CSV index sample:  {eco_df.index[:3].tolist()}")
        print(f"  h5ad obs_names sample: {adata.obs_names[:3].tolist()}")
        overlap = len(set(eco_df.index.astype(str)) & set(adata.obs_names.astype(str)))
        print(f"  Index overlap: {overlap:,} cells")

        if overlap > 0:
            # Cast index to str to match obs_names (CSV may read integers)
            eco_df.index = eco_df.index.astype(str)
            eco_map = eco_df['ecotype'].to_dict()
            adata.obs['ecotype'] = adata.obs_names.map(eco_map).fillna('Unassigned')
        elif len(eco_df) == adata.n_obs:
            # Fallback: same number of rows → assign positionally
            print("  WARNING: No index overlap, using positional assignment.")
            adata.obs['ecotype'] = eco_df['ecotype'].values
        else:
            sys.exit(
                f"ERROR: CSV index does not match h5ad obs_names and row counts differ "
                f"({len(eco_df):,} vs {adata.n_obs:,}).\n"
                f"Re-run step04_ecotype_panels.py to regenerate the CSV."
            )
        counts = adata.obs['ecotype'].value_counts().to_dict()
        print(f"  Ecotype distribution: {counts}")
    else:
        print(f"  Using ecotype from h5ad: "
              f"{adata.obs['ecotype'].value_counts().to_dict()}")

    run_cell_distance_analysis(adata, out_dir)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import stitch_vertical
    rel = os.path.join('downstream_analysis', 'figures', 'cell_distances')
    stitch_vertical(args.input_dir, [
        os.path.join(rel, 'distances_mPDAC_all_treatments'),
        os.path.join(rel, 'distances_ePDAC_all_treatments'),
    ], 'fig7_b')

    print("\n" + "=" * 60)
    print("CELL DISTANCE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:<55} {kb:>7.1f} KB")


if __name__ == '__main__':
    main()
