#!/usr/bin/env python3
"""
Annotation-level summary figures from the prior-adjusted merged_annotated.h5ad
(after cell-type priors have been applied).

Figures produced:
  fig_A_umap_all_celltypes.png/pdf, UMAP colored by cell type
  fig_B_umap_per_treatment.png/pdf, 4-panel UMAP per treatment
  fig_C_proportion_stacked_bar.png/pdf, Stacked bar: proportions per treatment
  fig_D_proportion_heatmap.png/pdf, Heatmap of proportions
  fig_E_proportion_grouped_bar.png/pdf, Grouped bar: top cell types
  fig_F_absolute_counts_bar.png/pdf, Absolute cell counts per treatment
  fig_G_marker_dotplot.png/pdf, Marker genes per cell type (dotplot)
  fig_H_marker_heatmap.png/pdf, Marker genes heatmap
  fig_I_spatial_celltype_maps.png/pdf, Spatial scatter per treatment
  (PAGA trajectory plots live in step07_cell_subtype_plots.py)

Usage:
    python step08_annotation_summary_plots.py --h5ad_path merged_annotated.h5ad --out_dir figures/annotation_summary
"""

import argparse
import os
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
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.sparse as sp

import matplotlib.font_manager as _fm

# ── Publication-quality settings ─────────────────────────────────────────────
_pref = next(
    (f for f in ['Arial', 'Liberation Sans', 'FreeSans', 'DejaVu Sans']
     if f in {x.name for x in _fm.fontManager.ttflist}),
    'sans-serif'
)
plt.rcParams.update({
    'font.family':           _pref,
    'font.size':             10,
    'axes.titlesize':        10,
    'axes.labelsize':        10,
    'xtick.labelsize':       10,
    'ytick.labelsize':       10,
    'legend.fontsize':       10,
    'legend.title_fontsize': 10,
    'pdf.fonttype':          42,
    'ps.fonttype':           42,
    'figure.dpi':            150,
})

# ── Constants ─────────────────────────────────────────────────────────────────
TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
TREAT_COLORS = {
    'Sham':   '#9E9E9E',
    'IT':     '#4E79A7',
    'GPH':    '#59A14F',
    'GPH+IT': '#E15759',
}
MAX_CELLS_UMAP   = 5_000   # per cell type for balanced subsample
MAX_SPATIAL_CELLS = 30_000  # per treatment for spatial plots

# Canonical marker genes per cell type (mouse)
MARKER_GENES = {
    'mPDAC':               ['Krt18', 'Krt19', 'Epcam', 'Kras', 'Cdh1', 'Muc1'],
    'ePDAC':               ['Vim', 'Fn1', 'Zeb1', 'Zeb2', 'Cdh2', 'Snai1'],
    'myCAFs':              ['Acta2', 'Tagln', 'Myl9', 'Postn', 'Hhip'],
    'iCAF':                ['Il6', 'Cxcl1', 'Pdgfra', 'Cxcl12', 'Has1'],
    'apCAFs':              ['H2-Ab1', 'Cd74', 'H2-Aa', 'Slpi'],
    'qCAF':                ['Col1a1', 'Col1a2', 'Col3a1', 'Dcn', 'Mfap5'],
    'M1 Macrophage':       ['Il1b', 'Nos2', 'Tnf', 'Cxcl10', 'Cd80'],
    'M2 Macrophage':       ['Mrc1', 'Cd163', 'Arg1', 'Il10', 'Tgfb1'],
    'CD4 T cells':         ['Cd4', 'Il7r', 'Ccr7', 'Foxp3'],
    'CD8 T cells':         ['Cd8a', 'Gzmb', 'Ifng', 'Prf1'],
    'Effector CD4+ T cells': ['Cd4', 'Ifng', 'Tnf', 'Il2'],
    'Cytotoxic T cells':   ['Gzmb', 'Gzma', 'Prf1', 'Fasl'],
    'NK cells':            ['Nkg7', 'Ncr1', 'Klrb1c', 'Klrd1'],
    'Tregs':               ['Foxp3', 'Il2ra', 'Ctla4', 'Ikzf2'],
    'B cells':             ['Cd79a', 'Ms4a1', 'Cd79b', 'Ighm'],
    'Endothelial cells':   ['Pecam1', 'Cdh5', 'Kdr', 'Vegfr2', 'Tek'],
}

# Flat deduplicated gene list preserving order
ALL_MARKER_GENES = list(dict.fromkeys(
    g for genes in MARKER_GENES.values() for g in genes
))


# ── Helpers ───────────────────────────────────────────────────────────────────

# The only figure this script builds that ends up in the paper (via
# collect_results.py); every other one is still computed but not saved.
_CURATED_STEMS = {'fig_D_proportion_heatmap'}


def save_fig(path_no_ext, dpi=300):
    if os.path.basename(path_no_ext) not in _CURATED_STEMS:
        return
    for ext in ('png', 'pdf'):
        # pad_inches: bbox_inches='tight' alone was cropping rotated axis/
        # colorbar labels a hair too close at large font sizes (lost the
        # final letter of "Squares" on the elbow plot at 32pt) -- a small
        # margin avoids the same failure here now that the colorbar label
        # has grown to 32pt too.
        plt.savefig(f"{path_no_ext}.{ext}", dpi=dpi,
                    bbox_inches='tight', pad_inches=0.15, facecolor='white')


def get_expr(matrix):
    if sp.issparse(matrix):
        return np.asarray(matrix.todense()).flatten()
    return np.asarray(matrix).flatten()


def filter_genes(genes, adata, min_expr_fraction=0.005):
    """Return genes present in dataset AND expressed in >= min_expr_fraction of cells."""
    if adata.raw is not None:
        var_names = adata.raw.var_names
        get_col   = lambda g: adata.raw[:, g].X
    else:
        var_names = adata.var_names
        get_col   = lambda g: adata[:, g].X
    avail = set(var_names)
    kept  = []
    for g in dict.fromkeys(genes):
        if g not in avail:
            continue
        x = get_col(g)
        n_expr = int((x > 0).nnz) if sp.issparse(x) else int((x > 0).sum())
        if n_expr / max(x.shape[0], 1) >= min_expr_fraction:
            kept.append(g)
    return kept


def build_ct_palette(cell_types):
    """
    Maximally distinct publication-quality color palette.
    Known cell types get fixed, biologically-grouped colors that vary across
    both hue and lightness so all 16 are clearly separable on a white background.
    Unknown types fall back to a supplemental palette.
    """
    FIXED_COLORS = {
        # Tumor cells: red / orange
        'mPDAC':                  '#E53935',  # vivid red
        'ePDAC':                  '#F57C00',  # vivid orange
        # CAF subtypes: purple / amber / hot-pink / warm-brown
        'myCAFs':                 '#7B1FA2',  # deep purple
        'iCAF':                   '#F9A825',  # amber / gold
        'apCAFs':                 '#E91E63',  # hot pink
        'qCAF':                   '#795548',  # warm medium brown
        # Macrophages: lime-green (M1 inflammatory) / espresso-brown (M2 suppressive)
        'M1 Macrophage':          '#7CB342',  # lime green  (distinct from all greens below)
        'M2 Macrophage':          '#4E342E',  # dark espresso brown
        # T cells: 5 clearly different colors spanning blue→teal→green→orchid
        'CD4 T cells':            '#1565C0',  # royal blue
        'CD8 T cells':            '#0097A7',  # dark cyan  (distinct shade from royal blue)
        'Effector CD4+ T cells':  '#2E7D32',  # dark forest green
        'Cytotoxic T cells':      '#00BFA5',  # medium teal-green
        'Tregs':                  '#E040FB',  # vivid orchid/magenta (unique, nothing else pink-purple)
        # Other immune
        'NK cells':               '#FF6F00',  # amber-orange (warm; stands apart from cool T-cell blues)
        'B cells':                '#5C6BC0',  # indigo periwinkle (lighter & more violet than CD4 royal blue)
        # Structural: steel blue-grey
        'Endothelial cells':      '#546E7A',  # steel blue-grey
    }

    # Supplemental fallback colors for any cell type not in FIXED_COLORS
    FALLBACK = [
        '#D84315', '#00838F', '#AD1457', '#6A1B9A', '#827717',
        '#4A148C', '#880E4F', '#BF360C', '#006064', '#1B5E20',
    ]
    palette = {}
    fallback_idx = 0
    for ct in cell_types:
        if ct in FIXED_COLORS:
            palette[ct] = FIXED_COLORS[ct]
        else:
            palette[ct] = FALLBACK[fallback_idx % len(FALLBACK)]
            fallback_idx += 1
    return palette


def compute_fresh_umap_from_counts(adata, max_per_ct=MAX_CELLS_UMAP, random_state=42,
                                    n_top_genes=3000, n_pcs=40,
                                    n_neighbors=15, min_dist=0.3, spread=1.0):
    """
    Compute UMAP from scratch on a balanced subsample using raw counts.
    Fresh local PCA (not the global Harmony embedding) gives better cluster
    separation because Harmony PCA coords from 1.2M cells lose structure
    when subsampled.

    Pipeline: balanced subsample → normalize → log1p → HVG → scale → PCA → neighbors → UMAP
    """
    ct_col = 'cell_type_auto'
    rng = np.random.RandomState(random_state)
    idxs = []
    for ct in adata.obs[ct_col].unique():
        ct_idx = np.where(adata.obs[ct_col] == ct)[0]
        n = min(len(ct_idx), max_per_ct)
        idxs.extend(rng.choice(ct_idx, n, replace=False).tolist())
    idxs = np.array(sorted(set(idxs)))
    print(f"  Balanced subsample: {len(idxs):,} cells "
          f"({adata.obs[ct_col].nunique()} cell types, max {max_per_ct}/type)")

    # Start from raw counts if available
    if adata.raw is not None:
        a = adata.raw.to_adata()[idxs].copy()
        a.obs = adata.obs.iloc[idxs].copy()
        print(f"  Using raw counts: {a.n_vars:,} genes")
    else:
        a = adata[idxs].copy()
        print(f"  Using adata.X: {a.n_vars:,} genes")

    # Standard scanpy preprocessing on the subsample
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    n_hvg = min(n_top_genes, a.n_vars - 1)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor='seurat',
                                 min_mean=0.0125, min_disp=0.25, subset=True)
    print(f"  HVGs selected: {a.n_vars:,}")
    sc.pp.scale(a, max_value=10)
    n_comps = min(n_pcs, a.n_obs - 1, a.n_vars - 1)
    sc.tl.pca(a, n_comps=n_comps, svd_solver='arpack')
    sc.pp.neighbors(a, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(a, min_dist=min_dist, spread=spread, random_state=random_state)
    x_range = a.obsm['X_umap'][:, 0]
    y_range = a.obsm['X_umap'][:, 1]
    print(f"  UMAP done. x: {x_range.min():.2f}–{x_range.max():.2f}, "
          f"y: {y_range.min():.2f}–{y_range.max():.2f}")
    return a


# ── Figure A: UMAP all cell types ─────────────────────────────────────────────

def _add_centroid_labels(ax, coords, labels, ct_palette, fontsize=9, min_cells=10):
    """
    Draw one text label per cell type at the median UMAP coordinate of that cluster.
    Uses a white outline (path_effects) so labels are readable over any color.
    Only labels clusters with >= min_cells points to avoid tiny/scattered clusters.
    """
    import matplotlib.patheffects as pe
    for ct in sorted(set(labels)):
        mask = labels == ct
        if mask.sum() < min_cells:
            continue
        cx = float(np.median(coords[mask, 0]))
        cy = float(np.median(coords[mask, 1]))
        ax.text(cx, cy, ct, fontsize=fontsize, ha='center', va='center',
                fontweight='bold', color='white',
                path_effects=[
                    pe.withStroke(linewidth=2.5,
                                  foreground=ct_palette.get(ct, '#333333'))
                ])


def fig_umap_all(adata_sub, ct_palette, out_dir):
    print("  Fig A: UMAP, all cell types...")
    ct_col = 'cell_type_auto'
    adata_sub.obs[ct_col] = adata_sub.obs[ct_col].astype('category')
    cats   = list(adata_sub.obs[ct_col].cat.categories)
    coords = adata_sub.obsm['X_umap']
    labels = adata_sub.obs[ct_col].astype(str).values

    adata_sub.uns[f'{ct_col}_colors'] = [
        matplotlib.colors.to_hex(ct_palette.get(c, '#cccccc')) for c in cats
    ]

    fig, ax = plt.subplots(figsize=(14, 11))
    # Draw dots: NO on-data legend (avoids label pile-up in dense center)
    sc.pl.umap(adata_sub, color=ct_col, ax=ax, show=False,
               size=45, alpha=0.85, frameon=False,
               legend_loc='none',
               title='')
    # In-plot centroid labels removed, numbered side legend already handles
    # cross-referencing and the labels were piling up illegibly in dense regions.

    ax.set_title('Cell Type Annotation', fontsize=20, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP 1', fontsize=15, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=15, fontweight='bold')

    # Side legend: numbered for easy cross-referencing
    handles = [mpatches.Patch(color=ct_palette.get(c, '#cccccc'), label=f'{i+1}. {c}')
               for i, c in enumerate(cats)]
    ax.legend(handles=handles, title='Cell Type',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=9, title_fontsize=11)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_A_umap_all_celltypes'))
    plt.close()
    print("    Saved: fig_A_umap_all_celltypes")


# ── Figure B: UMAP per treatment ──────────────────────────────────────────────

def fig_umap_per_treatment(adata_sub, ct_palette, out_dir):
    print("  Fig B: UMAP, per treatment...")
    ct_col  = 'cell_type_auto'
    coords  = adata_sub.obsm['X_umap']
    labels  = adata_sub.obs[ct_col].astype(str).values
    treats  = [t for t in TREATMENT_ORDER if t in adata_sub.obs['treatment'].unique()]
    unique  = sorted(set(labels))
    unique_by_size = sorted(unique, key=lambda ct: (labels == ct).sum(), reverse=True)

    # Assign palette into adata for scanpy
    adata_sub.obs[ct_col] = adata_sub.obs[ct_col].astype('category')
    cats = list(adata_sub.obs[ct_col].cat.categories)
    adata_sub.uns[f'{ct_col}_colors'] = [
        matplotlib.colors.to_hex(ct_palette.get(c, '#cccccc')) for c in cats
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (ax, treat) in enumerate(zip(axes, treats)):
        t_mask = adata_sub.obs['treatment'].values == treat
        a_t    = adata_sub[t_mask]

        # Grey background: all cells
        ax.scatter(coords[:, 0], coords[:, 1], s=2, color='#E8E8E8',
                   alpha=0.4, linewidths=0, rasterized=True)

        # Foreground: treatment cells via scanpy, no on-data labels
        sc.pl.umap(a_t, color=ct_col, ax=ax, show=False,
                   size=28, alpha=0.85, frameon=False,
                   legend_loc='none', title='')

        # Centroid labels removed for clarity

        n = int(t_mask.sum())
        ax.set_title(f'{treat}  (n={n:,})', fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP 1' if i >= 2 else '', fontsize=12)
        ax.set_ylabel('UMAP 2' if i % 2 == 0 else '', fontsize=12)

    # Hide unused panels
    for j in range(len(treats), len(axes)):
        axes[j].set_visible(False)

    handles = [mpatches.Patch(color=ct_palette.get(c, '#cccccc'), label=c) for c in cats]
    fig.legend(handles=handles, title='Cell Type',
               bbox_to_anchor=(1.01, 0.98), loc='upper left',
               frameon=False, fontsize=9, title_fontsize=11)
    fig.suptitle('Cell Type Distribution per Treatment', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_fig(os.path.join(out_dir, 'fig_B_umap_per_treatment'))
    plt.close()
    print("    Saved: fig_B_umap_per_treatment")


# ── Figure C: Stacked proportion bar ──────────────────────────────────────────

def fig_proportion_stacked(prop_df, ct_palette, out_dir):
    print("  Fig C: Stacked proportion bar...")
    treats = [t for t in TREATMENT_ORDER if t in prop_df.index]
    cts    = list(prop_df.columns)

    fig, ax = plt.subplots(figsize=(8, 7))
    bottom = np.zeros(len(treats))
    for ct in cts:
        vals = prop_df.reindex(treats)[ct].values.astype(float)
        ax.bar(treats, vals, bottom=bottom, color=ct_palette[ct],
               label=ct, edgecolor='none', width=0.65)
        bottom += vals

    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Treatment', fontsize=15, fontweight='bold')
    ax.set_ylabel('Proportion', fontsize=15, fontweight='bold')
    ax.set_title('Cell Type Composition per Treatment', fontsize=18, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    sns.despine(ax=ax)

    handles = [mpatches.Patch(color=ct_palette[ct], label=ct) for ct in cts]
    ax.legend(handles=handles, title='Cell Type',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=10, title_fontsize=11)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_C_proportion_stacked_bar'))
    plt.close()
    print("    Saved: fig_C_proportion_stacked_bar")


# ── Figure D: Proportion heatmap ──────────────────────────────────────────────

def fig_proportion_heatmap(prop_df, out_dir):
    print("  Fig D: Proportion heatmap...")
    treats = [t for t in TREATMENT_ORDER if t in prop_df.index]
    data   = prop_df.reindex(treats).T  # cell types × treatments

    # Row pitch kept modest: this panel is the tallest in the Fig. 6 montage and
    # an over-tall canvas here forces the whole composite off 16:9, which costs
    # every panel rendered font size on the slide.
    fig, ax = plt.subplots(figsize=(11, max(9, len(data) * 0.68 + 2)))
    sns.heatmap(
        data, ax=ax,
        cmap='YlOrRd', annot=True, fmt='.2f',
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Proportion', 'shrink': 0.6},
        annot_kws={'size': 26},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label('Proportion', fontsize=32)
    # Font sizes match panel H (the volcano): labels 32pt, ticks 28pt. No
    # title, this panel is identified by its composite letter alone.
    ax.set_xlabel('Treatment', fontsize=32)
    ax.set_ylabel('Cell Type', fontsize=32)
    plt.setp(ax.get_xticklabels(), fontsize=28, rotation=0)
    plt.setp(ax.get_yticklabels(), fontsize=28, rotation=0)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_D_proportion_heatmap'))
    plt.close()
    print("    Saved: fig_D_proportion_heatmap")


# ── Figure E: Grouped proportion bar (top cell types) ─────────────────────────

def fig_proportion_grouped(prop_df, ct_palette, out_dir, top_n=10):
    print("  Fig E: Grouped proportion bar...")
    treats = [t for t in TREATMENT_ORDER if t in prop_df.index]
    # Top N by mean proportion
    top_cts = prop_df.mean(axis=0).nlargest(top_n).index.tolist()
    data    = prop_df.reindex(treats)[top_cts]

    x     = np.arange(len(treats))
    width = 0.8 / len(top_cts)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ct in enumerate(top_cts):
        offset = (i - len(top_cts) / 2 + 0.5) * width
        ax.bar(x + offset, data[ct].values, width=width * 0.9,
               color=ct_palette[ct], label=ct, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(treats, fontsize=13, fontweight='bold')
    ax.set_xlabel('Treatment', fontsize=15, fontweight='bold')
    ax.set_ylabel('Proportion', fontsize=15, fontweight='bold')
    ax.set_title(f'Top {top_n} Cell Types per Treatment', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    sns.despine(ax=ax)
    ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=10, title_fontsize=11)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_E_proportion_grouped_bar'))
    plt.close()
    print("    Saved: fig_E_proportion_grouped_bar")


# ── Figure F: Absolute counts bar ─────────────────────────────────────────────

def fig_absolute_counts(count_df, ct_palette, out_dir):
    print("  Fig F: Absolute counts bar...")
    treats = [t for t in TREATMENT_ORDER if t in count_df.index]
    cts    = list(count_df.columns)

    fig, ax = plt.subplots(figsize=(8, 7))
    bottom = np.zeros(len(treats))
    for ct in cts:
        vals = count_df.reindex(treats)[ct].values.astype(float)
        ax.bar(treats, vals, bottom=bottom, color=ct_palette[ct],
               label=ct, edgecolor='none', width=0.65)
        bottom += vals

    ax.set_xlabel('Treatment', fontsize=15, fontweight='bold')
    ax.set_ylabel('Cell Count', fontsize=15, fontweight='bold')
    ax.set_title('Cell Count per Treatment', fontsize=18, fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    sns.despine(ax=ax)

    handles = [mpatches.Patch(color=ct_palette[ct], label=ct) for ct in cts]
    ax.legend(handles=handles, title='Cell Type',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=10, title_fontsize=11)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_F_absolute_counts_bar'))
    plt.close()
    print("    Saved: fig_F_absolute_counts_bar")


# ── Figure G: Marker dotplot ───────────────────────────────────────────────────

def fig_marker_dotplot(adata, out_dir):
    print("  Fig G: Marker dotplot...")
    cell_types  = sorted(ct for ct in adata.obs['cell_type_auto'].unique() if ct != 'Unassigned')
    genes       = filter_genes(ALL_MARKER_GENES, adata)

    # Build groupby with ordered categories
    adata.obs['cell_type_auto'] = pd.Categorical(
        adata.obs['cell_type_auto'], categories=cell_types, ordered=True)

    if not genes:
        print("    SKIP: no marker genes found")
        return

    use_raw = adata.raw is not None
    n_ct    = len(cell_types)
    n_genes = len(genes)

    try:
        fig_w = max(16, n_ct * 1.8 + 4)
        fig_h = max(12, n_genes * 0.62 + 4)
        dp = sc.pl.dotplot(
            adata, genes, groupby='cell_type_auto',
            use_raw=use_raw,
            color_map='Reds',
            swap_axes=True,           # genes on y, cell types on x
            title='Marker Genes per Cell Type',
            figsize=(fig_w, fig_h),
            show=False,
            return_fig=True,
            colorbar_title='Mean\nExpression',
            size_title='Fraction of\nCells (%)',
        )
        dp.style(
            cmap='Reds',
            dot_edge_color='black',
            dot_edge_lw=0.4,
            grid=True,
        )
        dp.make_figure()
        _dp_path = os.path.join(out_dir, 'fig_G_marker_dotplot')
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(f'{_dp_path}.png', dpi=200, bbox_inches='tight', facecolor='white')
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(f'{_dp_path}.pdf', bbox_inches='tight', facecolor='white')
        plt.close('all')
        print("    Saved: fig_G_marker_dotplot")
    except Exception as e:
        print(f"    WARNING: marker dotplot failed ({e}), using fallback")
        plt.close('all')
        try:
            # Manual matplotlib dotplot fallback
            mean_expr = pd.DataFrame(index=genes, columns=cell_types, dtype=float).fillna(0)
            frac_expr = pd.DataFrame(index=genes, columns=cell_types, dtype=float).fillna(0)
            for ct in cell_types:
                mask = adata.obs['cell_type_auto'] == ct
                for g in genes:
                    try:
                        if use_raw:
                            x = adata.raw[:, g].X[mask.values]
                        else:
                            x = adata[mask, g].X
                        arr = x.toarray().flatten() if sp.issparse(x) else np.asarray(x).flatten()
                        mean_expr.loc[g, ct] = float(arr.mean())
                        frac_expr.loc[g, ct] = float((arr > 0).mean() * 100)
                    except Exception:
                        pass
            fig_w2 = max(16, n_ct * 1.8 + 4)
            fig_h2 = max(12, n_genes * 0.62 + 4)
            fig, ax = plt.subplots(figsize=(fig_w2, fig_h2))
            max_me = mean_expr.values.max() if mean_expr.values.max() > 0 else 1.0
            for gi, g in enumerate(genes):
                for ci, ct in enumerate(cell_types):
                    me = float(mean_expr.loc[g, ct])
                    fe = float(frac_expr.loc[g, ct])
                    ax.scatter(ci, gi, s=max(4, fe * 2.5),
                               c=[[me / max_me, 0.1, 0.1]], alpha=0.85,
                               edgecolors='black', linewidths=0.3)
            ax.set_xticks(range(n_ct))
            ax.set_xticklabels(cell_types, rotation=45, ha='right', fontsize=11)
            ax.set_yticks(range(n_genes))
            ax.set_yticklabels(genes, fontsize=10)
            ax.set_xlim(-0.6, n_ct - 0.4)
            ax.set_ylim(-0.6, n_genes - 0.4)
            ax.set_title('Marker Genes per Cell Type', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.25, linewidth=0.5)
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, max_me))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Mean Expression', shrink=0.5, pad=0.02)
            sns.despine(ax=ax, left=True, bottom=True)
            plt.tight_layout()
            _dp_path = os.path.join(out_dir, 'fig_G_marker_dotplot')
            # Not a curated paper figure -- skip saving (computation above still used).
            # plt.savefig(f'{_dp_path}.png', dpi=200, bbox_inches='tight', facecolor='white')
            # Not a curated paper figure -- skip saving (computation above still used).
            # plt.savefig(f'{_dp_path}.pdf', bbox_inches='tight', facecolor='white')
            plt.close('all')
            print("    Saved: fig_G_marker_dotplot (fallback)")
        except Exception as e2:
            print(f"    ERROR: fallback also failed: {e2}")
            plt.close('all')


# ── Figure H: Marker heatmap ──────────────────────────────────────────────────

def fig_marker_heatmap(adata, out_dir):
    print("  Fig H: Marker heatmap...")
    cell_types = sorted(ct for ct in adata.obs['cell_type_auto'].unique() if ct != 'Unassigned')
    genes      = filter_genes(ALL_MARKER_GENES, adata)

    if not genes:
        print("    SKIP: no marker genes found")
        return

    use_raw = adata.raw is not None
    n_ct    = len(cell_types)
    n_genes = len(genes)

    try:
        mp = sc.pl.matrixplot(
            adata, genes, groupby='cell_type_auto',
            use_raw=use_raw,
            cmap='RdBu_r',
            swap_axes=True,
            title='Marker Genes per Cell Type (Mean Expression)',
            figsize=(max(10, n_ct * 1.1 + 2), max(6, n_genes * 0.45 + 2)),
            show=False,
            return_fig=True,
            colorbar_title='Mean\nExpression',
        )
        mp.style(cmap='RdBu_r', edge_lw=0.3)
        # Not a curated paper figure -- skip saving (computation above still used).
        # mp.savefig(os.path.join(out_dir, 'fig_H_marker_heatmap.png'),
        # dpi=300, bbox_inches='tight', facecolor='white')
        # Not a curated paper figure -- skip saving (computation above still used).
        # mp.savefig(os.path.join(out_dir, 'fig_H_marker_heatmap.pdf'),
        # dpi=300, bbox_inches='tight', facecolor='white')
        plt.close('all')
        print("    Saved: fig_H_marker_heatmap")
    except Exception as e:
        print(f"    WARNING: marker heatmap failed: {e}")
        plt.close('all')


# ── Figure I: Spatial cell type maps ──────────────────────────────────────────

def fig_spatial_maps(adata, ct_palette, out_dir):
    print("  Fig I: Spatial cell type maps...")
    if 'spatial' not in adata.obsm:
        print("    SKIP: no spatial coordinates")
        return

    treats  = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    ct_col  = 'cell_type_auto'
    labels  = adata.obs[ct_col].astype(str).values
    unique  = sorted(set(labels))
    coords  = adata.obsm['spatial']

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for ax, treat in zip(axes, treats):
        t_mask = adata.obs['treatment'].values == treat
        # Subsample for speed
        t_idx  = np.where(t_mask)[0]
        if len(t_idx) > MAX_SPATIAL_CELLS:
            rng   = np.random.RandomState(42)
            t_idx = rng.choice(t_idx, MAX_SPATIAL_CELLS, replace=False)

        # Cap each cell type at max_per_type so rare populations are visible
        max_per_type = 800
        rng = np.random.default_rng(42)
        keep_idx = []
        for ct in unique:
            ct_positions = np.where(labels[t_idx] == ct)[0]
            if len(ct_positions) > max_per_type:
                ct_positions = rng.choice(ct_positions, max_per_type, replace=False)
            keep_idx.append(ct_positions)
        keep_idx = np.concatenate(keep_idx)
        shuf = rng.permutation(len(keep_idx))
        plot_idx = keep_idx[shuf]

        cell_colors = np.array(
            [matplotlib.colors.to_rgb(ct_palette.get(labels[t_idx[i]], '#cccccc'))
             for i in plot_idx]
        )
        ax.scatter(
            coords[t_idx[plot_idx], 0],
            coords[t_idx[plot_idx], 1],
            c=cell_colors, s=12.0, alpha=0.9,
            linewidths=0, rasterized=True,
        )

        n = int(t_mask.sum())
        ax.set_title(f'{treat}  (n={n:,})', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel('X coordinate', fontsize=11)
        ax.set_ylabel('Y coordinate', fontsize=11)
        sns.despine(ax=ax)

    for j in range(len(treats), len(axes)):
        axes[j].set_visible(False)

    handles = [mpatches.Patch(color=ct_palette[ct], label=ct) for ct in unique]
    fig.legend(handles=handles, title='Cell Type',
               bbox_to_anchor=(1.01, 0.98), loc='upper left',
               frameon=False, fontsize=9, title_fontsize=10)
    fig.suptitle('Spatial Distribution of Cell Types', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    save_fig(os.path.join(out_dir, 'fig_I_spatial_celltype_maps'))
    plt.close()
    print("    Saved: fig_I_spatial_celltype_maps")


# ── Figure J: Per-treatment proportion summary table ──────────────────────────

def fig_proportion_table(prop_df, count_df, out_dir):
    print("  Fig J: Proportion summary table...")
    treats = [t for t in TREATMENT_ORDER if t in prop_df.index]
    cts    = list(prop_df.columns)
    n_cols = len(cts)
    n_rows = len(treats)

    # Wide figure: 1.8 in per column + 1.5 for row labels; tall enough for header + rows
    fig_w = max(20, n_cols * 1.8 + 1.5)
    fig_h = max(5, n_rows * 1.4 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    pct_data = (prop_df.reindex(treats) * 100).round(1)
    cnt_data = count_df.reindex(treats)

    cell_text = []
    for treat in treats:
        row = []
        for ct in cts:
            p = pct_data.loc[treat, ct] if ct in pct_data.columns else 0.0
            n = int(cnt_data.loc[treat, ct]) if ct in cnt_data.columns else 0
            row.append(f'{p:.1f}%\n({n:,})')
        cell_text.append(row)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=treats,
        colLabels=cts,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 2.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=7,
                                rotation=40, ha='right', va='bottom')
            cell.set_height(0.22)
        elif c == -1:
            cell.set_facecolor('#ECF0F1')
            cell.set_text_props(fontweight='bold', fontsize=9)
            cell.set_width(0.12)
        else:
            cell.set_facecolor('#FDFEFE' if r % 2 == 0 else '#EBF5FB')

    ax.set_title('Cell Type Proportions & Counts per Treatment',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_J_proportion_table'))
    plt.close()
    print("    Saved: fig_J_proportion_table")


def _run_paga(a, ct_col, n_neighbors=15, n_pcs=30):  # kept for potential internal use
    """Run neighbors → PAGA on AnnData `a`. Returns the object (modified in-place)."""
    n_comps = min(n_pcs, a.n_obs - 1, a.n_vars - 1)
    n_nb    = min(n_neighbors, a.n_obs - 1)
    sc.pp.neighbors(a, n_neighbors=n_nb, n_pcs=n_comps, use_rep='X_pca')
    sc.tl.paga(a, groups=ct_col)
    return a


def fig_paga_trajectory(adata_sub, ct_palette, out_dir):
    """PAGA combined + per-treatment trajectory plots."""
    print("  Fig K: PAGA trajectory plots...")

    ct_col = 'cell_type_auto'
    cats   = sorted(adata_sub.obs[ct_col].astype(str).unique())

    # Assign consistent colors into .uns so sc.pl.paga picks them up
    adata_sub.obs[ct_col] = pd.Categorical(adata_sub.obs[ct_col], categories=cats)
    adata_sub.uns[f'{ct_col}_colors'] = [
        matplotlib.colors.to_hex(ct_palette.get(c, '#cccccc')) for c in cats
    ]

    # ── Combined PAGA (all treatments) ────────────────────────────────────────
    print("    Running PAGA (combined)...")
    _run_paga(adata_sub, ct_col)

    # Dynamic threshold: show only top-20 strongest connections
    import scipy.sparse as sp
    _conn = adata_sub.uns['paga']['connectivities']
    _vals = _conn.data if sp.issparse(_conn) else _conn[_conn > 0].flatten()
    if len(_vals) > 0:
        _sorted = np.sort(_vals)[::-1]
        _n_show = min(20, len(_sorted))
        _paga_thresh = float(_sorted[_n_show - 1]) * 0.999  # just below the 20th value
    else:
        _paga_thresh = 0.4
    print(f"    PAGA combined threshold (top-20 edges): {_paga_thresh:.4f}  (range {_vals.min():.4f}–{_vals.max():.4f})")

    fig, ax = plt.subplots(figsize=(12, 10))
    sc.pl.paga(
        adata_sub, color=ct_col,
        ax=ax, show=False,
        threshold=_paga_thresh,
        node_size_scale=2.5, edge_width_scale=1.5,
        fontsize=9, fontoutline=2,
        title='PAGA — All Treatments Combined',
    )
    ax.set_title('PAGA — All Treatments Combined', fontsize=18, fontweight='bold', pad=10)
    handles = [mpatches.Patch(color=ct_palette.get(c, '#cccccc'), label=c) for c in cats]
    ax.legend(handles=handles, title='Cell Type',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=9, title_fontsize=11)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_K_paga_combined'))
    plt.close()
    print("    Saved: fig_K_paga_combined")

    # ── Per-treatment PAGA, 2×2 grid ─────────────────────────────────────────
    treats = [t for t in TREATMENT_ORDER if t in adata_sub.obs['treatment'].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()

    for i, (ax, treat) in enumerate(zip(axes, treats)):
        t_mask = adata_sub.obs['treatment'].values == treat
        n_cells = int(t_mask.sum())
        if n_cells < 20:
            ax.text(0.5, 0.5, f'{treat}\nInsufficient cells',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(treat, fontsize=14, fontweight='bold')
            continue

        a_t = adata_sub[t_mask].copy()
        # Assign colors into subset AnnData
        a_t.obs[ct_col] = pd.Categorical(a_t.obs[ct_col].astype(str), categories=cats)
        a_t.uns[f'{ct_col}_colors'] = [
            matplotlib.colors.to_hex(ct_palette.get(c, '#cccccc')) for c in cats
        ]

        print(f"    Running PAGA ({treat}, {n_cells:,} cells)...")
        try:
            _run_paga(a_t, ct_col, n_neighbors=min(10, n_cells - 1))
            _conn_t = a_t.uns['paga']['connectivities']
            _vals_t = _conn_t.data if sp.issparse(_conn_t) else _conn_t[_conn_t > 0].flatten()
            if len(_vals_t) > 0:
                _sorted_t = np.sort(_vals_t)[::-1]
                _n_show_t = min(15, len(_sorted_t))
                _thresh_t = float(_sorted_t[_n_show_t - 1]) * 0.999
            else:
                _thresh_t = 0.4
            sc.pl.paga(
                a_t, color=ct_col,
                ax=ax, show=False,
                threshold=_thresh_t,
                node_size_scale=2.0, edge_width_scale=1.2,
                fontsize=8, fontoutline=2,
                title='',
            )
        except Exception as e:
            ax.text(0.5, 0.5, f'{treat}\nPAGA failed:\n{e}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9,
                    wrap=True)

        ax.set_title(f'{treat}  (n={n_cells:,})', fontsize=15, fontweight='bold')

    for j in range(len(treats), len(axes)):
        axes[j].set_visible(False)

    handles = [mpatches.Patch(color=ct_palette.get(c, '#cccccc'), label=c) for c in cats]
    fig.legend(handles=handles, title='Cell Type',
               bbox_to_anchor=(1.01, 0.98), loc='upper left',
               frameon=False, fontsize=9, title_fontsize=11)
    fig.suptitle('PAGA Trajectory — Per Treatment', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    save_fig(os.path.join(out_dir, 'fig_K_paga_per_treatment'))
    plt.close()
    print("    Saved: fig_K_paga_per_treatment")


# ── Figure L: PAGA trajectory, CAF subtypes per treatment ────────────────────

CAF_TYPES = ['myCAFs', 'iCAF', 'apCAFs', 'qCAF']


def fig_paga_cafs(adata_sub, ct_palette, out_dir):
    """PAGA trajectory plots focused on CAF subtypes."""
    """
    Generate PAGA trajectory plots focused on CAF subtypes (myCAFs / iCAF / apCAFs / qCAF):
      fig_L_paga_cafs_combined.png, all treatments combined, CAF subset only
      fig_L_paga_cafs_per_treatment.png, 2×2 grid, CAF subset per treatment
    Context cells (non-CAF) are included in the subsample so PAGA can show
    CAF→non-CAF connectivity, but only CAF nodes are labelled.
    """
    print("  Fig L: PAGA, CAF subtypes...")

    ct_col   = 'cell_type_auto'
    cats     = sorted(adata_sub.obs[ct_col].astype(str).unique())
    caf_present = [c for c in CAF_TYPES if c in cats]
    if not caf_present:
        print("    SKIP: no CAF cell types found in subsample")
        return

    # Subset to CAF cells + their immediate neighbours in expression space
    # (include all cells for connectivity, then highlight CAFs)
    caf_mask = adata_sub.obs[ct_col].isin(caf_present).values
    n_caf    = int(caf_mask.sum())
    print(f"    CAF cells in subsample: {n_caf:,}  ({', '.join(caf_present)})")

    if n_caf < 10:
        print("    SKIP: too few CAF cells for PAGA")
        return

    def _paga_caf_subset(a, label, ax, n_nb=10):
        """Run PAGA on full `a`, draw with CAFs highlighted."""
        a.obs[ct_col] = pd.Categorical(a.obs[ct_col].astype(str), categories=cats)
        a.uns[f'{ct_col}_colors'] = [
            matplotlib.colors.to_hex(ct_palette.get(c, '#cccccc')) for c in cats
        ]
        _run_paga(a, ct_col, n_neighbors=min(n_nb, a.n_obs - 1))
        sc.pl.paga(
            a, color=ct_col,
            ax=ax, show=False,
            node_size_scale=2.5, edge_width_scale=1.5,
            fontsize=9, fontoutline=2,
            title='',
        )

    # ── Combined ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    a_caf = adata_sub.copy()
    print("    Running CAF PAGA (combined)...")
    try:
        _paga_caf_subset(a_caf, 'combined', ax)
    except Exception as e:
        ax.text(0.5, 0.5, f'PAGA failed:\n{e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)

    ax.set_title('PAGA — CAF Subtypes (All Treatments)', fontsize=18, fontweight='bold', pad=10)

    # Build legend: CAFs bold, others greyed
    handles = []
    for c in cats:
        color  = ct_palette.get(c, '#cccccc')
        weight = 'bold' if c in caf_present else 'normal'
        alpha  = 1.0    if c in caf_present else 0.45
        h = mpatches.Patch(facecolor=color, label=c, alpha=alpha)
        handles.append(h)
    ax.legend(handles=handles, title='Cell Type (CAFs bold)',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=9, title_fontsize=11)
    plt.tight_layout()
    save_fig(os.path.join(out_dir, 'fig_L_paga_cafs_combined'))
    plt.close()
    print("    Saved: fig_L_paga_cafs_combined")

    # ── Per-treatment 2×2 grid ────────────────────────────────────────────────
    treats = [t for t in TREATMENT_ORDER if t in adata_sub.obs['treatment'].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()

    for i, (ax, treat) in enumerate(zip(axes, treats)):
        t_mask  = adata_sub.obs['treatment'].values == treat
        n_cells = int(t_mask.sum())
        n_caf_t = int((t_mask & caf_mask).sum())

        if n_caf_t < 5 or n_cells < 20:
            ax.text(0.5, 0.5, f'{treat}\nInsufficient CAF cells\n(n={n_caf_t})',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{treat}  (CAFs: {n_caf_t})', fontsize=14, fontweight='bold')
            continue

        a_t = adata_sub[t_mask].copy()
        print(f"    Running CAF PAGA ({treat}, {n_caf_t} CAF / {n_cells:,} total cells)...")
        try:
            _paga_caf_subset(a_t, treat, ax, n_nb=min(10, n_cells - 1))
        except Exception as e:
            ax.text(0.5, 0.5, f'{treat}\nPAGA failed:\n{e}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)

        ax.set_title(f'{treat}  (CAFs: {n_caf_t:,} / {n_cells:,} cells)',
                     fontsize=15, fontweight='bold')

    for j in range(len(treats), len(axes)):
        axes[j].set_visible(False)

    handles = []
    for c in cats:
        color = ct_palette.get(c, '#cccccc')
        alpha = 1.0 if c in caf_present else 0.45
        handles.append(mpatches.Patch(facecolor=color, label=c, alpha=alpha))
    fig.legend(handles=handles, title='Cell Type (CAFs bold)',
               bbox_to_anchor=(1.01, 0.98), loc='upper left',
               frameon=False, fontsize=9, title_fontsize=11)
    fig.suptitle('PAGA Trajectory — CAF Subtypes per Treatment',
                 fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    save_fig(os.path.join(out_dir, 'fig_L_paga_cafs_per_treatment'))
    plt.close()
    print("    Saved: fig_L_paga_cafs_per_treatment")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Regenerate annotation figures from the prior-adjusted h5ad')
    parser.add_argument(
        '--h5ad_path',
        default=os.path.join(os.path.expanduser('~'), 'stereo-seq', 'stereoseq-analysis',
                             'downstream_analysis', 'processed_data', 'merged_annotated.h5ad'))
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.expanduser('~'), 'stereo-seq', 'stereoseq-analysis',
                             'downstream_analysis', 'figures', '07_annotation_post_prior'))
    parser.add_argument(
        '--skip_umap', action='store_true',
        help='Skip UMAP recomputation (use if already done)')
    args = parser.parse_args()

    if not os.path.exists(args.h5ad_path):
        print(f"ERROR: h5ad not found: {args.h5ad_path}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("STEP 2 POST-7: ANNOTATION PLOTS (POST-PRIOR)")
    print("=" * 65)
    print(f"h5ad:    {args.h5ad_path}")
    print(f"out_dir: {args.out_dir}")

    print("\nLoading h5ad...")
    adata = sc.read_h5ad(args.h5ad_path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    print(f"  Cell types: {sorted(adata.obs['cell_type_auto'].unique())}")
    print(f"  Treatments: {sorted(adata.obs['treatment'].unique())}")
    print(f"  raw available: {adata.raw is not None}")

    cell_types = sorted(ct for ct in adata.obs['cell_type_auto'].unique() if ct != 'Unassigned')
    ct_palette = build_ct_palette(cell_types)

    # ── Compute proportions & counts from prior-adjusted h5ad ────────────────
    print("\nComputing proportions from prior-adjusted data...")
    counts_raw = (adata.obs
                  .groupby(['treatment', 'cell_type_auto'])
                  .size()
                  .unstack(fill_value=0))
    treats_present = [t for t in TREATMENT_ORDER if t in counts_raw.index]
    counts_df = counts_raw.reindex(index=treats_present, columns=cell_types, fill_value=0)
    prop_df   = counts_df.div(counts_df.sum(axis=1), axis=0)

    print("  Proportions (%):")
    print((prop_df * 100).round(1).to_string())

    # ── Recompute UMAP from raw counts (fresh local PCA, balanced subsample) ──
    adata_sub = None
    if not args.skip_umap:
        print("\nRecomputing UMAP from raw counts (fresh PCA on balanced subsample)...")
        try:
            adata_sub = compute_fresh_umap_from_counts(adata)
            umap_df = pd.DataFrame(
                adata_sub.obsm['X_umap'],
                columns=['UMAP1', 'UMAP2'],
                index=adata_sub.obs_names,
            )
            umap_df['cell_type'] = adata_sub.obs['cell_type_auto'].values
            umap_df['treatment'] = adata_sub.obs['treatment'].values
            umap_df.to_csv(os.path.join(args.out_dir, 'umap_coordinates.csv'))
            print(f"  UMAP ready: {adata_sub.n_obs:,} cells")
        except Exception as e:
            print(f"  WARNING: UMAP computation failed: {e}")
            import traceback; traceback.print_exc()
            adata_sub = None
    else:
        print("\nSkipping UMAP (--skip_umap)")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")

    if adata_sub is not None:
        fig_umap_all(adata_sub, ct_palette, args.out_dir)
        fig_umap_per_treatment(adata_sub, ct_palette, args.out_dir)

    fig_proportion_stacked(prop_df, ct_palette, args.out_dir)
    fig_proportion_heatmap(prop_df, args.out_dir)
    fig_proportion_grouped(prop_df, ct_palette, args.out_dir)
    fig_absolute_counts(counts_df, ct_palette, args.out_dir)
    fig_marker_dotplot(adata, args.out_dir)
    fig_marker_heatmap(adata, args.out_dir)
    fig_spatial_maps(adata, ct_palette, args.out_dir)
    fig_proportion_table(prop_df, counts_df, args.out_dir)

    if adata_sub is not None:
        fig_paga_trajectory(adata_sub, ct_palette, args.out_dir)
        fig_paga_cafs(adata_sub, ct_palette, args.out_dir)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    base_dir = args.out_dir.split('downstream_analysis')[0] or '.'
    collect(base_dir, {
        os.path.join('downstream_analysis', 'figures', 'annotation_summary', 'fig_D_proportion_heatmap'): 'fig6_b',
    })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ALL FIGURES COMPLETE")
    print("=" * 65)
    pngs = [f for f in os.listdir(args.out_dir) if f.endswith('.png')]
    print(f"  {len(pngs)} PNG files in: {args.out_dir}")
    for f in sorted(pngs):
        size = os.path.getsize(os.path.join(args.out_dir, f)) / 1024
        print(f"    {f}  ({size:.0f} KB)")


if __name__ == '__main__':
    main()
