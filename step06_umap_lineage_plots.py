#!/usr/bin/env python3
"""
Publication-quality UMAP plots, style matching Figure 2A of the paper.

Computed from a fresh PCA on the balanced subsample's normalized .X,
bypassing the Harmony-corrected embedding: Harmony removes batch (treatment)
differences, which also flattens cell-type separation in the UMAP, so a
plain PCA is used here to keep cell types visually distinct.

5 figures produced:
  umap_all_combined.png/pdf, all cells, colored by cell type
  umap_Sham.png/pdf, Sham highlighted on grey
  umap_IT.png/pdf, IT highlighted
  umap_GPH.png/pdf, GPH highlighted
  umap_GPH_plus_IT.png/pdf, GPH+IT highlighted

Usage:
    python step06_umap_lineage_plots.py --input_dir .
    python step06_umap_lineage_plots.py --input_dir . --force_recompute
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import trustworthiness

import matplotlib.font_manager as _fm
_pref = next((f for f in ['Arial', 'Liberation Sans', 'FreeSans', 'DejaVu Sans']
              if f in {x.name for x in _fm.fontManager.ttflist}), 'sans-serif')
plt.rcParams.update({
    'font.family': _pref, 'font.size': 10,
    'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 10,  'ytick.labelsize': 10,
    'legend.fontsize': 10, 'legend.title_fontsize': 10,
    'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
DEFAULT_MARKERS = [
    'PTPRC', 'CD3D', 'MS4A1', 'NKG7', 'LYZ',
    'COL1A1', 'PECAM1', 'EPCAM', 'KRT19',
]

CELLTYPE_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
    '#8da0cb', '#e78ac3', '#a6d854', '#e5c494', '#ffd92f',
    '#b3b3b3',
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _palette(cell_types):
    return {ct: CELLTYPE_PALETTE[i % len(CELLTYPE_PALETTE)]
            for i, ct in enumerate(sorted(cell_types))}


def _umap_xy(adata):
    coords = adata.obsm['X_umap']
    valid  = ~np.isnan(coords[:, 0])
    return coords[:, 0], coords[:, 1], valid


# The 4 lineage panels (immune/stromal/epithelial_tumor/other) are the only
# figures this script builds that end up in the paper (suppl8_*); every other
# UMAP view is still computed but not saved.
_CURATED_STEMS = {
    'umap_lineage_immune', 'umap_lineage_stromal',
    'umap_lineage_epithelial_tumor', 'umap_lineage_other',
}


def _save(fig, path_no_ext):
    if os.path.basename(path_no_ext) not in _CURATED_STEMS:
        plt.close(fig)
        return
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_no_ext}.{ext}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _bottom_legend(fig, ax, pal):
    n    = len(pal)
    ncol = max(3, -(-n // 3))
    handles = [mpatches.Patch(color=pal[ct], label=ct)
               for ct in sorted(pal)]
    ax.legend(handles=handles,
              loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=ncol, frameon=False,
              fontsize=8.5, handlelength=1.0, handleheight=0.85,
              columnspacing=0.8, handletextpad=0.4)


def _draw(ax, x, y, valid, fore_mask, pal, ct_order, obs_ct,
          xlim, ylim, title, grey_back=True):
    if grey_back and fore_mask is not None:
        back = valid & ~fore_mask
        ax.scatter(x[back], y[back], c='#d0d0d0', s=1.7, alpha=0.16,
                   rasterized=True, linewidths=0, zorder=1)

    pmask = valid if fore_mask is None else (valid & fore_mask)
    for ct in ct_order[::-1]:          # rare types drawn last → on top
        m = pmask & (obs_ct == ct).values
        if m.sum() == 0:
            continue
        ax.scatter(x[m], y[m], c=pal[ct], s=2.4, alpha=0.88,
                   rasterized=True, linewidths=0, zorder=2)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    # Sized for a panel that lands at roughly a quarter page in the composite:
    # at the old 8-13pt the labels were illegible once scaled down. Not bold --
    # figure content stays regular weight, only the panel letters are bold.
    ax.set_title(title, fontsize=30, pad=10)
    ax.set_xlabel('UMAP_1', fontsize=26)
    ax.set_ylabel('UMAP_2', fontsize=26)
    ax.tick_params(axis='both', labelsize=22, length=5, width=1.0)
    for sp_ in ax.spines.values():
        sp_.set_linewidth(0.6)


def _categorical_palette(values):
    cats = sorted(pd.Series(values).dropna().astype(str).unique())
    cmap = plt.get_cmap('tab20', max(3, len(cats)))
    return {c: mcolors.to_hex(cmap(i)) for i, c in enumerate(cats)}


def _infer_lineage(cell_type):
    ct = str(cell_type).lower()
    immune_kw = ['t cell', 'b cell', 'nk', 'macrophage', 'monocyte',
                 'dendritic', 'neutrophil', 'mast']
    stromal_kw = ['caf', 'fibro', 'endothelial', 'pericyte', 'smooth muscle']
    epithelial_kw = ['pdac', 'epithelial', 'tumor', 'malignant', 'ductal', 'acinar']
    if any(k in ct for k in immune_kw):
        return 'Immune'
    if any(k in ct for k in stromal_kw):
        return 'Stromal'
    if any(k in ct for k in epithelial_kw):
        return 'Epithelial/Tumor'
    return 'Other'


def _resolve_markers(adata, markers):
    if adata.raw is not None:
        names = pd.Index(adata.raw.var_names)
    else:
        names = pd.Index(adata.var_names)
    low_to_name = {str(g).lower(): str(g) for g in names}
    resolved = {}
    missing = []
    for g in markers:
        hit = low_to_name.get(str(g).lower())
        if hit is None:
            missing.append(g)
        else:
            resolved[g] = hit
    return resolved, missing


def _gene_vector(adata, gene_name):
    src = adata.raw if adata.raw is not None else adata
    vec = src[:, gene_name].X
    if sp.issparse(vec):
        vec = vec.toarray()
    vec = np.asarray(vec).reshape(-1).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Plot functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_combined(adata, x, y, valid, xlim, ylim, pal, ct_order, out_dir):
    print("  umap_all_combined ...")
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.22)
    _draw(ax, x, y, valid, None, pal, ct_order,
          adata.obs['cell_type_auto'], xlim, ylim,
          'Annotated Cell Types', grey_back=False)
    ax.text(0.02, 0.02, f"n = {int(valid.sum()):,}",
            transform=ax.transAxes, fontsize=8, color='#555', va='bottom')
    _bottom_legend(fig, ax, pal)
    _save(fig, os.path.join(out_dir, 'umap_all_combined'))
    print("    → umap_all_combined.png/pdf")


def plot_treatment(adata, x, y, valid, xlim, ylim, pal, ct_order,
                   treatment, out_dir):
    safe = treatment.replace('+', '_plus_')
    print(f"  umap_{safe} ...")
    tmask = (adata.obs['treatment'] == treatment).values
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.22)
    _draw(ax, x, y, valid, tmask, pal, ct_order,
          adata.obs['cell_type_auto'], xlim, ylim,
          treatment, grey_back=True)
    n = int((valid & tmask).sum())
    ax.text(0.02, 0.02, f"n = {n:,}", transform=ax.transAxes,
            fontsize=8, color='#555', va='bottom')
    _bottom_legend(fig, ax, pal)
    _save(fig, os.path.join(out_dir, f'umap_{safe}'))
    print(f"    → umap_{safe}.png/pdf")


def plot_unsupervised_clusters(adata, x, y, valid, xlim, ylim, out_dir):
    if 'leiden_umap' not in adata.obs:
        print("  unsupervised UMAP skipped (missing leiden_umap).")
        return
    lab = adata.obs['leiden_umap'].astype(str)
    has = valid & (~pd.isna(adata.obs['leiden_umap']).values)
    if int(has.sum()) < 100:
        print("  unsupervised UMAP skipped (insufficient clustered cells).")
        return

    pal = _categorical_palette(lab[has])
    order = pd.Series(lab[has]).value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.20)
    _draw(ax, x, y, valid, None, pal, order, lab, xlim, ylim,
          'Unsupervised Clusters (Leiden)', grey_back=False)
    ax.text(0.02, 0.02, f"clusters = {len(order)}",
            transform=ax.transAxes, fontsize=8, color='#555', va='bottom')

    ncol = max(4, -(-len(order) // 4))
    handles = [mpatches.Patch(color=pal[c], label=c) for c in order]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=ncol, frameon=False, fontsize=8,
              handlelength=1.0, handleheight=0.8, columnspacing=0.8)
    _save(fig, os.path.join(out_dir, 'umap_unsupervised_leiden'))
    print("    → umap_unsupervised_leiden.png/pdf")


def plot_lineage_panels(adata, x, y, valid, xlim, ylim, pal, ct_order, out_dir):
    if 'cell_lineage' not in adata.obs:
        print("  lineage panel skipped (missing cell_lineage).")
        return
    lineages = [l for l in ['Immune', 'Stromal', 'Epithelial/Tumor', 'Other']
                if l in set(adata.obs['cell_lineage'])]
    if len(lineages) == 0:
        print("  lineage panel skipped (no lineage categories).")
        return

    n = len(lineages)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, lin in enumerate(lineages):
        ax = axes[i]
        lmask = (adata.obs['cell_lineage'] == lin).values
        _draw(ax, x, y, valid, lmask, pal, ct_order, adata.obs['cell_type_auto'],
              xlim, ylim, lin, grey_back=True)
        ax.text(0.02, 0.02, f"n = {int((valid & lmask).sum()):,}",
                transform=ax.transAxes, fontsize=22, color='#555', va='bottom')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Deeper bottom band: the shared legend is now set at panel-label size
    # rather than 8pt, so it needs real room instead of a thin strip.
    #
    # hspace also has to clear a full stack of chrome between the rows -- the
    # upper row's tick labels, its "UMAP_1" xlabel, and then the lower row's
    # title. At 0.28 the xlabel was landing on top of the row beneath it.
    fig.subplots_adjust(bottom=0.20, hspace=0.52, wspace=0.18)
    handles = [mpatches.Patch(color=pal[ct], label=ct) for ct in sorted(pal)]
    fig.legend(handles=handles, loc='lower center',
               ncol=max(4, -(-len(handles) // 3)), frameon=False,
               fontsize=24, handlelength=1.4, handleheight=1.0,
               columnspacing=1.4, handletextpad=0.6)
    _save(fig, os.path.join(out_dir, 'umap_lineage_panels'))
    print("    → umap_lineage_panels.png/pdf")

    # Individual per-lineage files (self-contained, own legend) for use as
    # standalone supplementary sub-panels rather than only the combined grid.
    handles = [mpatches.Patch(color=pal[ct], label=ct) for ct in sorted(pal)]
    for lin in lineages:
        fig_i, ax_i = plt.subplots(figsize=(8, 7))
        lmask = (adata.obs['cell_lineage'] == lin).values
        _draw(ax_i, x, y, valid, lmask, pal, ct_order, adata.obs['cell_type_auto'],
              xlim, ylim, lin, grey_back=True)
        ax_i.text(0.02, 0.02, f"n = {int((valid & lmask).sum()):,}",
                  transform=ax_i.transAxes, fontsize=22, color='#555', va='bottom')
        fig_i.subplots_adjust(bottom=0.32)
        fig_i.legend(handles=handles, loc='lower center',
                    ncol=max(4, -(-len(handles) // 3)), frameon=False,
                    fontsize=14, handlelength=1.4, handleheight=1.0,
                    columnspacing=1.4, handletextpad=0.6)
        slug = lin.lower().replace('/', '_').replace(' ', '_')
        _save(fig_i, os.path.join(out_dir, f'umap_lineage_{slug}'))
        print(f"    → umap_lineage_{slug}.png/pdf")


def plot_marker_overlays(adata, x, y, valid, xlim, ylim, out_dir, markers=None):
    markers = DEFAULT_MARKERS if markers is None else markers
    resolved, missing = _resolve_markers(adata, markers)
    if len(missing) > 0:
        print(f"  Marker genes not found: {missing}")
    if len(resolved) == 0:
        print("  marker overlays skipped (no markers found).")
        return

    n = len(resolved)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows))
    axes = np.array(axes).reshape(-1)

    xv = x[valid]
    yv = y[valid]
    for i, (label, gene) in enumerate(resolved.items()):
        ax = axes[i]
        expr = _gene_vector(adata, gene)[valid]
        hi = float(np.percentile(expr, 99.5)) if expr.size > 0 else 1.0
        hi = max(hi, 1e-6)
        expr_clip = np.clip(expr, 0, hi)
        # Grey background for all points, then overlay expression intensity
        ax.scatter(xv, yv, c='#e0e0e0', s=1.2, alpha=0.30,
                   linewidths=0, rasterized=True, zorder=1)
        sca = ax.scatter(xv, yv, c=expr_clip, cmap='viridis',
                         s=2.2, alpha=0.90, linewidths=0,
                         rasterized=True, zorder=2)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{label} ({gene})', fontsize=12)
        ax.set_xlabel('UMAP_1', fontsize=10)
        ax.set_ylabel('UMAP_2', fontsize=10)
        ax.tick_params(axis='both', labelsize=8, length=3, width=0.6)
        for sp_ in ax.spines.values():
            sp_.set_linewidth(0.6)
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Expr', fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(hspace=0.22, wspace=0.22)
    _save(fig, os.path.join(out_dir, 'umap_marker_overlays'))
    print("    → umap_marker_overlays.png/pdf")


def write_umap_metrics(adata, out_dir):
    if 'X_umap' not in adata.obsm:
        return
    emb = adata.obsm['X_umap']
    valid = ~np.isnan(emb[:, 0])
    if int(valid.sum()) < 500:
        return

    e = emb[valid]
    ct = adata.obs['cell_type_auto'].astype(str).values[valid]
    ln = adata.obs['cell_lineage'].astype(str).values[valid] if 'cell_lineage' in adata.obs else None
    ld = adata.obs['leiden_umap'].astype(str).values[valid] if 'leiden_umap' in adata.obs else None

    summary = {}

    def _safe_silhouette(labels):
        vals, cnt = np.unique(labels, return_counts=True)
        keep = np.isin(labels, vals[cnt >= 20])
        labels2 = labels[keep]
        emb2 = e[keep]
        if len(np.unique(labels2)) < 3 or emb2.shape[0] < 500:
            return np.nan
        return float(silhouette_score(emb2, labels2))

    summary['n_cells_valid_umap'] = int(valid.sum())
    summary['n_cell_types'] = int(pd.Series(ct).nunique())
    summary['silhouette_cell_type'] = _safe_silhouette(ct)
    try:
        summary['calinski_harabasz_cell_type'] = float(calinski_harabasz_score(e, ct))
        summary['davies_bouldin_cell_type'] = float(davies_bouldin_score(e, ct))
    except Exception:
        summary['calinski_harabasz_cell_type'] = np.nan
        summary['davies_bouldin_cell_type'] = np.nan

    if ln is not None:
        summary['n_lineages'] = int(pd.Series(ln).nunique())
        summary['silhouette_lineage'] = _safe_silhouette(ln)

    if ld is not None:
        # Cluster purity: dominant cell type fraction in each Leiden cluster
        rows = []
        for cl, dcl in pd.DataFrame({'cluster': ld, 'cell_type': ct}).groupby('cluster'):
            n = len(dcl)
            top_frac = dcl['cell_type'].value_counts(normalize=True).iloc[0]
            top_label = dcl['cell_type'].value_counts().index[0]
            rows.append({'cluster': cl, 'n_cells': n,
                         'top_cell_type': top_label, 'top_fraction': float(top_frac)})
        purity_df = pd.DataFrame(rows).sort_values('n_cells', ascending=False)
        if not purity_df.empty:
            summary['leiden_clusters'] = int(purity_df.shape[0])
            summary['mean_cluster_purity_unweighted'] = float(purity_df['top_fraction'].mean())
            summary['mean_cluster_purity_weighted'] = float(
                np.average(purity_df['top_fraction'], weights=purity_df['n_cells'])
            )
            purity_df.to_csv(os.path.join(out_dir, 'umap_leiden_cluster_purity.csv'), index=False)

    # Include best layout metrics if present
    for k, v in adata.uns.get('umap_best_params', {}).items():
        summary[f'best_param_{k}'] = v
    for k, v in adata.uns.get('umap_best_metrics', {}).items():
        summary[f'best_metric_{k}'] = v

    lines = ["UMAP quantitative quality metrics", "-" * 36]
    for k in sorted(summary):
        lines.append(f"{k}: {summary[k]}")
    out_txt = os.path.join(out_dir, 'umap_quantitative_metrics.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"    → {os.path.basename(out_txt)}")

# ══════════════════════════════════════════════════════════════════════════════
# Balanced subsample
# ══════════════════════════════════════════════════════════════════════════════

def _subsample(adata, max_per_ct, rng):
    keep = []
    for ct in adata.obs['cell_type_auto'].unique():
        idx = np.where((adata.obs['cell_type_auto'] == ct).values)[0]
        n   = min(len(idx), max_per_ct)
        keep.extend(rng.choice(idx, size=n, replace=False).tolist())
    keep = sorted(keep)
    print(f"  Balanced subsample: {len(keep):,} cells "
          f"({adata.obs['cell_type_auto'].nunique()} types, ≤{max_per_ct:,} each)")
    return adata[keep].copy(), np.array(keep)


# ══════════════════════════════════════════════════════════════════════════════
# Fresh PCA: bypasses Harmony to recover cell-type separation
# ══════════════════════════════════════════════════════════════════════════════

def _select_discriminative_genes(X, labels, top_per_ct=25):
    """
    Balanced per-cell-type marker gene selection.

    For EACH cell type, find the top `top_per_ct` genes most specifically
    upregulated in that type vs all others (mean difference score).
    Combine all selected genes → ensures every cell type is represented equally
    in PCA, preventing dominant types (iCAF, mPDAC) from collapsing the space.

    Without balance: F-statistic selects iCAF-dominated genes (F=5245),
    PC1 = iCAF vs everything, rest collapses into a blob.
    With balance: each of 16 cell types contributes ~25 marker genes → PCA
    spans 16-dimensional cell type space → separated clusters in UMAP.

    X          : (n_cells, n_genes) dense float array
    labels     : (n_cells,) cell type strings
    top_per_ct : top N marker genes per cell type
    Returns    : boolean gene mask
    """
    unique_cts = np.unique(labels)
    selected   = set()

    for ct in unique_cts:
        mask_ct    = labels == ct
        mask_other = ~mask_ct
        if mask_ct.sum() < 5 or mask_other.sum() < 5:
            continue
        # mean expression in this type vs all others
        mean_ct    = X[mask_ct].mean(axis=0)
        mean_other = X[mask_other].mean(axis=0)
        score      = mean_ct - mean_other          # higher = more specific to ct
        top_idx    = np.argsort(score)[::-1][:top_per_ct]
        selected.update(top_idx.tolist())

    gene_mask = np.zeros(X.shape[1], dtype=bool)
    gene_mask[list(selected)] = True
    print(f"  Selected {gene_mask.sum()} marker genes "
          f"(≤{top_per_ct} per cell type × {len(unique_cts)} types)")
    return gene_mask


def _fresh_pca(sub):
    """
    Compute fresh PCA for the subsample: two paths:

    PATH A (preferred): sub.raw exists → raw counts available
      normalize_total → log1p → highly_variable_genes (4000, seurat_v3)
      → scale → PCA 50 PCs
      This is the gold-standard pipeline used in Nature/Cell spatial papers.

    PATH B (fallback): no raw counts → .X is already scaled
      Select top 25 marker genes per cell type (balanced, per-type)
      → PCA on those discriminative genes
      Prevents dominant types from collapsing the embedding.
    """
    labels = sub.obs['cell_type_auto'].values

    # ── PATH A: start from raw counts ────────────────────────────────────────
    if sub.raw is not None:
        print("  PATH A: raw counts found, running full pipeline ...")
        work = sub.raw.to_adata()
        work.obs = sub.obs.copy()

        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)

        # HVGs: seurat_v3 flavor, 4000 genes
        try:
            sc.pp.highly_variable_genes(work, n_top_genes=4000,
                                        flavor='seurat_v3', span=1.0)
        except Exception:
            sc.pp.highly_variable_genes(work, n_top_genes=4000)
        n_hvg = work.var['highly_variable'].sum()
        print(f"  HVGs selected: {n_hvg}")
        work = work[:, work.var['highly_variable']]

        sc.pp.scale(work, max_value=10)
        if sp.issparse(work.X):
            work.X = work.X.toarray()
        work.X = np.nan_to_num(work.X, nan=0.0, posinf=10.0, neginf=-10.0)

        n_pcs = min(50, work.n_vars - 1, work.n_obs - 1)
        sc.pp.pca(work, n_comps=n_pcs, svd_solver='arpack')
        print(f"  PCA done: {n_pcs} PCs from {work.n_vars} HVGs")

        sub.obsm['X_pca_fresh'] = work.obsm['X_pca']
        return sub, n_pcs

    # ── PATH B: .X is already scaled, use balanced marker gene selection ─────
    print("  PATH B: no raw counts, using balanced marker gene selection ...")
    X = sub.X.toarray() if sp.issparse(sub.X) else np.array(sub.X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"  .X: shape={X.shape}, min={X.min():.2f}, max={X.max():.2f}")

    gene_mask = _select_discriminative_genes(X, labels, top_per_ct=25)
    X_sub = X[:, gene_mask]

    n_pcs = min(50, X_sub.shape[1] - 1, X_sub.shape[0] - 1)
    print(f"  PCA on {gene_mask.sum()} marker genes → {n_pcs} PCs ...")

    from sklearn.decomposition import PCA as skPCA
    pca_model = skPCA(n_components=n_pcs, random_state=42)
    X_pca = pca_model.fit_transform(X_sub).astype(np.float32)
    var_exp = pca_model.explained_variance_ratio_[:5]
    print(f"  Variance explained PC1-5: {[f'{v:.3f}' for v in var_exp]}")

    sub.obsm['X_pca_fresh'] = X_pca
    return sub, n_pcs


# ══════════════════════════════════════════════════════════════════════════════
# UMAP computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_umap(adata, max_per_ct=5000, seed=42):
    """
    Compute publication-quality UMAP:
      1. Balanced subsample (≤max_per_ct per cell type)
      2. Fresh PCA from .X  (not Harmony)
      3. Neighbours: n_neighbors=15, metric='cosine'
      4. UMAP: min_dist=0.05, spread=1.2
    """
    rng = np.random.default_rng(seed)
    sub, idx = _subsample(adata, max_per_ct, rng)

    sub, _ = _fresh_pca(sub)

    print("  Auto-tuning UMAP parameters for best separation ...")
    sub, best_cfg, best_metrics = _compute_best_umap(sub, seed=seed)

    print("  Leiden clustering on best-neighbor graph ...")
    sc.pp.neighbors(
        sub,
        use_rep='X_pca_fresh',
        n_pcs=min(40, sub.obsm['X_pca_fresh'].shape[1]),
        n_neighbors=best_cfg['n_neighbors'],
        metric=best_cfg['metric'],
        random_state=seed
    )
    try:
        sc.tl.leiden(sub, resolution=1.0, key_added='leiden_umap', random_state=seed)
    except Exception:
        print("  Leiden unavailable; using Louvain fallback.")
        sc.tl.louvain(sub, resolution=1.0, key_added='leiden_umap', random_state=seed)

    # Write back; non-subsampled cells get NaN
    coords = np.full((adata.n_obs, 2), np.nan)
    coords[idx] = sub.obsm['X_umap']
    adata.obsm['X_umap'] = coords

    subsampled = np.zeros(adata.n_obs, dtype=bool)
    subsampled[idx] = True
    adata.obs['umap_subsampled'] = subsampled

    leiden_all = np.array([np.nan] * adata.n_obs, dtype=object)
    leiden_all[idx] = sub.obs['leiden_umap'].astype(str).values
    adata.obs['leiden_umap'] = pd.Categorical(leiden_all)

    adata.uns['umap_best_params'] = {
        'n_neighbors': int(best_cfg['n_neighbors']),
        'metric': str(best_cfg['metric']),
        'min_dist': float(best_cfg['min_dist']),
        'spread': float(best_cfg['spread']),
    }
    adata.uns['umap_best_metrics'] = {k: float(v) for k, v in best_metrics.items()}
    print(f"  UMAP stored for {len(idx):,} cells.")
    return adata


def _is_blob(adata):
    """Return (is_blob, fraction_in_centre) for the stored X_umap."""
    if 'X_umap' not in adata.obsm:
        return True, 1.0
    xv = adata.obsm['X_umap'][:, 0]
    yv = adata.obsm['X_umap'][:, 1]
    xv = xv[~np.isnan(xv)];  yv = yv[~np.isnan(yv)]
    if len(xv) < 100:
        return True, 1.0
    xr = xv.max() - xv.min();  yr = yv.max() - yv.min()
    frac = ((np.abs(xv - xv.mean()) < xr * 0.40) &
            (np.abs(yv - yv.mean()) < yr * 0.40)).mean()
    return float(frac) > 0.70, float(frac)


def _layout_quality_score(emb, labels, pca_repr, seed=42):
    """
    Score a UMAP embedding for biological interpretability and geometry.
    Higher is better.
    """
    valid = ~np.isnan(emb[:, 0])
    embv = emb[valid]
    labv = labels[valid]
    pcav = pca_repr[valid]

    if embv.shape[0] < 500:
        return -1.0, {}

    # Evaluate on a capped subset for speed/stability
    rng = np.random.default_rng(seed)
    n_eval = min(8000, embv.shape[0])
    pick = rng.choice(embv.shape[0], size=n_eval, replace=False)
    e = embv[pick]
    l = labv[pick]
    p = pcav[pick]

    unique, counts = np.unique(l, return_counts=True)
    valid_types = unique[counts >= 20]
    keep = np.isin(l, valid_types)
    e2 = e[keep]
    l2 = l[keep]
    p2 = p[keep]

    if len(np.unique(l2)) < 3 or e2.shape[0] < 500:
        return -1.0, {}

    sil = float(silhouette_score(e2, l2, metric='euclidean'))
    trust = float(trustworthiness(p2, e2, n_neighbors=15))

    xv = e2[:, 0]
    yv = e2[:, 1]
    xr = xv.max() - xv.min()
    yr = yv.max() - yv.min()
    frac_center = ((np.abs(xv - xv.mean()) < xr * 0.40) &
                   (np.abs(yv - yv.mean()) < yr * 0.40)).mean()
    anti_blob = 1.0 - float(frac_center)

    score = 0.60 * sil + 0.30 * trust + 0.10 * anti_blob
    return score, {
        'silhouette': sil,
        'trustworthiness': trust,
        'anti_blob': anti_blob,
        'frac_center': float(frac_center),
    }


def _compute_best_umap(sub, seed=42):
    """
    Try several neighborhood/UMAP settings and keep the best-scoring layout.
    """
    labels = sub.obs['cell_type_auto'].values
    pca = sub.obsm['X_pca_fresh']
    trials = [
        {'n_neighbors': 15, 'metric': 'cosine', 'min_dist': 0.08, 'spread': 1.1},
        {'n_neighbors': 20, 'metric': 'cosine', 'min_dist': 0.20, 'spread': 1.0},
        {'n_neighbors': 30, 'metric': 'cosine', 'min_dist': 0.25, 'spread': 1.2},
        {'n_neighbors': 40, 'metric': 'cosine', 'min_dist': 0.35, 'spread': 1.3},
    ]

    best_score = -1e9
    best_emb = None
    best_cfg = None
    best_metrics = {}

    for i, cfg in enumerate(trials, start=1):
        trial = sub.copy()
        print(f"  UMAP tuning {i}/{len(trials)}: "
              f"n_neighbors={cfg['n_neighbors']}, min_dist={cfg['min_dist']}, "
              f"spread={cfg['spread']}")
        sc.pp.neighbors(
            trial,
            use_rep='X_pca_fresh',
            n_pcs=min(40, trial.obsm['X_pca_fresh'].shape[1]),
            n_neighbors=cfg['n_neighbors'],
            metric=cfg['metric'],
            random_state=seed
        )
        sc.tl.umap(
            trial,
            min_dist=cfg['min_dist'],
            spread=cfg['spread'],
            n_components=2,
            maxiter=500,
            random_state=seed
        )
        score, metrics = _layout_quality_score(
            trial.obsm['X_umap'], labels, pca, seed=seed
        )
        print("    score={:.4f} (sil={:.3f}, trust={:.3f}, center={:.1%})".format(
            score,
            metrics.get('silhouette', -1.0),
            metrics.get('trustworthiness', -1.0),
            metrics.get('frac_center', 1.0),
        ))
        if score > best_score:
            best_score = score
            best_emb = trial.obsm['X_umap'].copy()
            best_cfg = cfg
            best_metrics = metrics

    print("  Best UMAP config: "
          f"n_neighbors={best_cfg['n_neighbors']}, "
          f"min_dist={best_cfg['min_dist']}, spread={best_cfg['spread']} "
          f"(score={best_score:.4f})")
    sub.obsm['X_umap'] = best_emb
    return sub, best_cfg, best_metrics


def _annotate_lineage(adata):
    adata.obs['cell_lineage'] = pd.Categorical(
        adata.obs['cell_type_auto'].astype(str).map(_infer_lineage)
    )


def _derive_leiden_from_existing_umap(adata, seed=42):
    if 'leiden_umap' in adata.obs and adata.obs['leiden_umap'].notna().sum() > 100:
        return
    if 'X_umap' not in adata.obsm:
        return
    valid = ~np.isnan(adata.obsm['X_umap'][:, 0])
    if int(valid.sum()) < 500:
        return
    print("  Deriving leiden_umap from existing UMAP coordinates ...")
    tmp = sc.AnnData(adata.obsm['X_umap'][valid].copy())
    sc.pp.neighbors(tmp, n_neighbors=20, metric='euclidean', random_state=seed)
    try:
        sc.tl.leiden(tmp, resolution=1.0, key_added='leiden_umap', random_state=seed)
    except Exception:
        sc.tl.louvain(tmp, resolution=1.0, key_added='leiden_umap', random_state=seed)
    out = np.array([np.nan] * adata.n_obs, dtype=object)
    out[valid] = tmp.obs['leiden_umap'].astype(str).values
    adata.obs['leiden_umap'] = pd.Categorical(out)


def ensure_umap(adata, force, max_per_ct):
    has = ('X_umap' in adata.obsm and
           not np.all(np.isnan(adata.obsm['X_umap'])))

    if has and not force:
        blob, frac = _is_blob(adata)
        if blob:
            print(f"  WARNING: stored UMAP is a blob ({frac:.0%} in central 40%).")
            print("  Recomputing with fresh PCA (bypassing Harmony) ...")
            force = True
        else:
            print(f"  X_umap looks separated ({frac:.0%} in centre), reusing.")

    if not has or force:
        adata = compute_umap(adata, max_per_ct)

    return adata


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        default=os.path.expanduser('~/stereo-seq/stereoseq-analysis'))
    parser.add_argument('--out_subdir',      default='umap')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--max_per_celltype', type=int, default=5000)
    args = parser.parse_args()

    h5ad = os.path.join(args.input_dir, 'downstream_analysis',
                        'processed_data', 'merged_annotated.h5ad')
    if not os.path.exists(h5ad):
        sys.exit(f"ERROR: {h5ad} not found.")

    out_dir = os.path.join(args.input_dir, 'downstream_analysis',
                           'figures', args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Input:  {h5ad}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    adata = sc.read_h5ad(h5ad)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    print(f"  Cell types ({adata.obs['cell_type_auto'].nunique()}): "
          f"{sorted(adata.obs['cell_type_auto'].unique())}")
    print(f"  Treatments: {sorted(adata.obs['treatment'].unique())}")

    # Report what embeddings are available
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    if 'X_umap' in adata.obsm:
        blob, frac = _is_blob(adata)
        print(f"  Existing X_umap: {'BLOB' if blob else 'OK'} "
              f"({frac:.0%} in central 40%)")

    adata = ensure_umap(adata, args.force_recompute, args.max_per_celltype)
    _annotate_lineage(adata)
    _derive_leiden_from_existing_umap(adata)

    x, y, valid = _umap_xy(adata)
    pad  = 1.0
    xlim = (float(np.nanmin(x[valid])) - pad, float(np.nanmax(x[valid])) + pad)
    ylim = (float(np.nanmin(y[valid])) - pad, float(np.nanmax(y[valid])) + pad)

    cell_types = sorted(ct for ct in adata.obs['cell_type_auto'].unique() if ct != 'Unassigned')
    pal        = _palette(cell_types)
    ct_order   = [ct for ct in adata.obs['cell_type_auto'].value_counts().index if ct != 'Unassigned']
    treats     = [t for t in TREATMENT_ORDER
                  if t in adata.obs['treatment'].values]

    total_figs = 4 + len(treats)
    print(f"\nGenerating {total_figs} figures + metrics ...")
    plot_all_combined(adata, x, y, valid, xlim, ylim, pal, ct_order, out_dir)
    plot_unsupervised_clusters(adata, x, y, valid, xlim, ylim, out_dir)
    plot_lineage_panels(adata, x, y, valid, xlim, ylim, pal, ct_order, out_dir)
    plot_marker_overlays(adata, x, y, valid, xlim, ylim, out_dir, markers=DEFAULT_MARKERS)
    for t in treats:
        plot_treatment(adata, x, y, valid, xlim, ylim, pal, ct_order, t, out_dir)
    write_umap_metrics(adata, out_dir)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    rel = os.path.join('downstream_analysis', 'figures', args.out_subdir)
    collect(args.input_dir, {
        os.path.join(rel, 'umap_lineage_immune'):           'suppl8_immune',
        os.path.join(rel, 'umap_lineage_stromal'):           'suppl8_stromal',
        os.path.join(rel, 'umap_lineage_epithelial_tumor'):  'suppl8_epithelial_tumor',
        os.path.join(rel, 'umap_lineage_other'):             'suppl8_other',
    })

    print()
    print("=" * 60)
    print(f"Done. Figures in: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
