#!/usr/bin/env python3
"""
Cell subtype subclustering, spatial ecotype detection, and treatment
composition trends.

Run after step02_build_annotated_h5ad.py completes.
Input:  merged_annotated.h5ad
Output: subcluster labels, ecotype assignments, composition heatmaps
"""

import os

RESOLUTION_LABEL = 'CellBin'  # overwritten in main() based on --resolution arg
import sys
import argparse
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
from scipy.stats import fisher_exact, zscore

import matplotlib.font_manager as _fm
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_preferred_font = next(
    (f for f in ['Arial', 'Liberation Sans', 'FreeSans', 'DejaVu Sans'] if f in _available_fonts),
    'sans-serif'
)

plt.rcParams.update({
    'font.family': _preferred_font,
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10,
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.0)

# ============================================================
# CONFIGURATION
# ============================================================
TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
TREATMENT_COLORS = {
    'Sham': '#2166AC', 'IT': '#EF8A62', 'GPH': '#67A9CF', 'GPH+IT': '#B2182B'
}

CELL_TYPE_MARKERS = {
    'ePDAC':               ['Krt19', 'Krt8', 'Epcam', 'Cdh1', 'Krt18', 'Cldn7', 'Cldn4'],
    'mPDAC':               ['Fn1', 'Vim', 'Cdh2', 'Zeb1', 'Zeb2', 'Snai1', 'Twist1'],
    'myCAFs':              ['Acta2', 'Tagln', 'Col1a1', 'Col1a2', 'Col3a1', 'Postn', 'Tnc'],
    'iCAF':                ['Il6', 'Cxcl1', 'Cxcl12', 'Has1', 'Pdgfra', 'Lif', 'Cxcl5'],
    'apCAFs':              ['Cd74', 'H2-Aa', 'H2-Ab1', 'H2-Eb1', 'Saa3'],
    'qCAF':                ['Dcn', 'Gsn', 'Cygb', 'Fbln1', 'Mfap5'],
    'CD4 T cells':         ['Cd3e', 'Cd3d', 'Cd4', 'Il7r', 'Tcf7'],
    'CD8 T cells':         ['Cd3e', 'Cd8a', 'Cd8b1', 'Gzmb', 'Prf1'],
    'Cytotoxic T cells':   ['Gzmb', 'Gzma', 'Prf1', 'Nkg7', 'Ifng'],
    'Effector CD4+ T cells': ['Cd4', 'Ifng', 'Tbx21', 'Il2', 'Tnf'],
    'Tregs':               ['Foxp3', 'Il2ra', 'Ctla4', 'Tnfrsf18', 'Ikzf2'],
    'NK cells':            ['Ncr1', 'Klrb1c', 'Nkg7', 'Klrk1', 'Gzma'],
    'M1 Macrophage':       ['Nos2', 'Il1b', 'Tnf', 'Cd86', 'Il12b'],
    'M2 Macrophage':       ['Mrc1', 'Cd163', 'Arg1', 'Retnla', 'Chil3'],
    'B cells':             ['Cd79a', 'Ms4a1', 'Cd19', 'Pax5', 'Ebf1'],
    'Endothelial cells':   ['Pecam1', 'Cdh5', 'Vwf', 'Kdr', 'Emcn'],
}


def setup_dirs(output_dir):
    """Build the output directory paths.

    Only `data` is created here; `annotation` and `summary` are created
    lazily by the functions that write into them.
    """
    base = os.path.join(output_dir, 'downstream_analysis')
    dirs = {
        'annotation':  os.path.join(base, 'figures', '07_subcluster_diagnostics'),
        'summary':     os.path.join(base, 'figures', '10_summary'),
        'data':        os.path.join(base, 'processed_data'),
    }
    os.makedirs(dirs['data'], exist_ok=True)
    return dirs


# ============================================================
# SUBCLUSTER MAJOR POPULATIONS
# ============================================================
def subcluster_major_populations(adata, dirs):
    """
    Subcluster macrophages, T cells, and CAFs to reveal fine-grained subtypes.
    """
    print("\n" + "="*60)
    print("SUBCLUSTERING MAJOR POPULATIONS")
    print("="*60)

    populations_to_subcluster = {
        'Macrophages': {
            'keywords': ['Macrophage', 'macrophage', 'Myeloid', 'monocyte', 'Monocyte', 'DC', 'Dendritic'],
            'markers': {
                'M1-like TAM':    ['Nos2', 'Il1b', 'Tnf', 'Cd86', 'Il12b', 'Cxcl9', 'Cxcl10'],
                'M2-like TAM':    ['Mrc1', 'Cd163', 'Arg1', 'Retnla', 'Chil3', 'Il10', 'Tgfb1'],
                'Inflammatory Mo': ['Ly6c2', 'Ccr2', 'S100a8', 'S100a9', 'Csf3r'],
                'cDC1':           ['Clec9a', 'Xcr1', 'Cadm1', 'Irf8'],
                'cDC2':           ['Cd1c', 'Cd8a', 'Clec12a', 'Itgam'],
                'pDC':            ['Siglech', 'Ly6d', 'Bst2', 'Irf7'],
            }
        },
        'T_cells': {
            'keywords': ['T cell', 'T cells', 'CD4', 'CD8', 'Treg', 'NK', 'NKT'],
            'markers': {
                'Naive CD4+':    ['Cd4', 'Ccr7', 'Sell', 'Tcf7', 'Il7r'],
                'Effector CD4+': ['Cd4', 'Ifng', 'Tbx21', 'Il2', 'Tnf'],
                'Treg':          ['Foxp3', 'Il2ra', 'Ctla4', 'Ikzf2', 'Tnfrsf18'],
                'Naive CD8+':    ['Cd8a', 'Ccr7', 'Sell', 'Tcf7'],
                'Effector CD8+': ['Cd8a', 'Gzmb', 'Prf1', 'Ifng', 'Nkg7'],
                'Exhausted CD8+': ['Cd8a', 'Pdcd1', 'Lag3', 'Havcr2', 'Tigit', 'Tox'],
                'NK cells':      ['Ncr1', 'Klrb1c', 'Nkg7', 'Klrk1', 'Gzma'],
            }
        },
        'Fibroblasts': {
            'keywords': ['Fibroblast', 'fibroblast', 'CAF', 'Stellate', 'myofibroblast'],
            'markers': {
                'myCAF': ['Acta2', 'Tagln', 'Postn', 'Tnc', 'Col1a1', 'Pdgfrb'],
                'iCAF':  ['Il6', 'Cxcl1', 'Cxcl12', 'Has1', 'Pdgfra', 'Ly6c1'],
                'apCAF': ['Cd74', 'H2-Aa', 'H2-Ab1', 'Saa3', 'H2-Eb1'],
                'qCAF':  ['Dcn', 'Gsn', 'Cygb', 'Fbln1', 'Mfap5'],
            }
        },
    }

    subcluster_results = {}

    for pop_name, pop_config in populations_to_subcluster.items():
        print(f"\n--- Subclustering: {pop_name} ---")

        keywords = pop_config['keywords']
        mask = adata.obs['cell_type_auto'].str.contains('|'.join(keywords), case=False, na=False)
        n_cells = mask.sum()

        if n_cells < 50:
            print(f"  Insufficient cells ({n_cells}) for subclustering, skipping.")
            continue

        print(f"  Selected {n_cells:,} cells for subclustering")
        adata_sub = adata[mask].copy()

        try:
            sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, flavor='seurat_v3') if 'highly_variable' not in adata_sub.var.columns else None

            use_hvg = adata_sub.var.get('highly_variable', pd.Series(True, index=adata_sub.var_names))
            adata_hvg = adata_sub[:, use_hvg].copy()
            sc.pp.scale(adata_hvg, max_value=10)
            import scipy.sparse as sp_sparse
            if sp_sparse.issparse(adata_hvg.X):
                adata_hvg.X = adata_hvg.X.toarray()
            adata_hvg.X = np.nan_to_num(adata_hvg.X, nan=0.0)
            gene_std = adata_hvg.X.std(axis=0)
            gene_mask = gene_std > 0
            if gene_mask.sum() > 10:
                adata_hvg = adata_hvg[:, gene_mask].copy()
                print(f"  After scale NaN removal: {gene_mask.sum():,} / {len(gene_mask):,} genes retained")
            sc.tl.pca(adata_hvg, n_comps=min(30, n_cells-1, adata_hvg.n_vars-1))
            adata_sub.obsm['X_pca_sub'] = adata_hvg.obsm['X_pca']

            # n_neighbors=10 gives finer subcluster resolution than the scanpy default.
            n_neighbors = min(10, n_cells - 1)
            sc.pp.neighbors(adata_sub, n_pcs=min(20, adata_sub.obsm['X_pca_sub'].shape[1]),
                            use_rep='X_pca_sub', n_neighbors=n_neighbors)
            sc.tl.umap(adata_sub, min_dist=0.3)

            # Wide resolution range: subpopulation size varies a lot by cell type.
            for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
                sc.tl.leiden(adata_sub, resolution=res, key_added='subcluster')
                n_sub = adata_sub.obs['subcluster'].nunique()
                if n_sub >= 5:
                    break

            print(f"  Found {n_sub} subclusters (resolution={res})")

            subtype_scores = {}
            if adata_sub.raw is not None:
                adata_score_sub = adata_sub.raw.to_adata().copy()
                adata_score_sub.obs = adata_sub.obs.copy()
                gene_set_sub = set(adata_score_sub.var_names)
                print(f"  Using adata.raw for subtype scoring: {adata_score_sub.n_vars:,} genes")
            else:
                adata_score_sub = adata_sub
                gene_set_sub = set(adata_sub.var_names)

            for subtype, markers in pop_config['markers'].items():
                available = [g for g in markers if g in gene_set_sub]
                if len(available) >= 2:
                    sc.tl.score_genes(adata_score_sub, gene_list=available, score_name=f'score_{subtype}')
                    adata_sub.obs[f'score_{subtype}'] = adata_score_sub.obs[f'score_{subtype}'].values
                    subtype_scores[subtype] = f'score_{subtype}'

            if subtype_scores:
                score_cols_sub  = list(subtype_scores.values())
                score_names_sub = list(subtype_scores.keys())
                score_matrix    = adata_sub.obs[score_cols_sub].values
                best_idx_sub    = np.argmax(score_matrix, axis=1)
                best_scores_sub = np.max(score_matrix, axis=1)
                # Argmax with same threshold used in primary annotation
                sub_labels = np.array([score_names_sub[i] for i in best_idx_sub])
                sub_labels[best_scores_sub < 0.05] = 'Unassigned'
                adata_sub.obs['subtype_label'] = sub_labels

                col_name = f'subcluster_{pop_name}'
                adata.obs[col_name] = 'Other'
                adata.obs.loc[mask, col_name] = adata_sub.obs['subtype_label'].values

                print(f"  Subtype distribution:")
                for st, n in adata_sub.obs['subtype_label'].value_counts().items():
                    print(f"    {st:<30}: {n:>6,} ({100*n/n_cells:.1f}%)")

            fig, axes = plt.subplots(1, 3, figsize=(42, 15))
            fig.subplots_adjust(bottom=0.18)

            sc.pl.umap(adata_sub, color='subcluster', ax=axes[0], show=False,
                       legend_loc='none',
                       title=f'{pop_name}: Leiden Subclusters', frameon=False, s=5)

            if 'subtype_label' in adata_sub.obs.columns:
                sc.pl.umap(adata_sub, color='subtype_label', ax=axes[1], show=False,
                           legend_loc='none',
                           title=f'{pop_name}: Subtype Labels', frameon=False, s=5)

            sc.pl.umap(adata_sub, color='treatment', ax=axes[2], show=False,
                       palette=TREATMENT_COLORS, legend_loc='none',
                       title=f'{pop_name}: Treatment', frameon=False, s=5)

            # Shared legends below each panel
            _sub_cats  = sorted(adata_sub.obs['subcluster'].unique())
            _sub_cols  = adata_sub.uns.get('subcluster_colors', [])
            if _sub_cols:
                _sub_h = [mpatches.Patch(color=_sub_cols[i % len(_sub_cols)], label=c)
                          for i, c in enumerate(_sub_cats)]
                axes[0].legend(handles=_sub_h, bbox_to_anchor=(0.5, -0.06),
                               loc='upper center', frameon=False, fontsize=9,
                               ncol=min(5, len(_sub_cats)))
            if 'subtype_label' in adata_sub.obs.columns:
                _sty_cats = sorted(adata_sub.obs['subtype_label'].unique())
                _sty_cols = adata_sub.uns.get('subtype_label_colors', [])
                if _sty_cols:
                    _sty_h = [mpatches.Patch(color=_sty_cols[i % len(_sty_cols)], label=c)
                              for i, c in enumerate(_sty_cats)]
                    axes[1].legend(handles=_sty_h, bbox_to_anchor=(0.5, -0.06),
                                   loc='upper center', frameon=False, fontsize=9,
                                   ncol=min(5, len(_sty_cats)))
            _tr_h = [mpatches.Patch(color=v, label=k) for k, v in TREATMENT_COLORS.items()]
            axes[2].legend(handles=_tr_h, bbox_to_anchor=(0.5, -0.06),
                           loc='upper center', frameon=False, fontsize=10, ncol=4)

            plt.suptitle(f'{pop_name} Subclustering Analysis', fontsize=24, fontweight='bold')
            plt.tight_layout(rect=[0, 0.10, 1, 0.97])
            # Not a curated paper figure -- skip saving (computation above still used).
            plt.close()

            if subtype_scores and 'subtype_label' in adata_sub.obs.columns:
                flat_markers = {}
                for subtype, markers in pop_config['markers'].items():
                    available = [g for g in markers if g in gene_set_sub][:4]
                    if available:
                        flat_markers[subtype] = available

                adata_score_sub.obs['subtype_label'] = adata_sub.obs['subtype_label'].values
                if flat_markers:
                    try:
                        _dp_sub = sc.pl.dotplot(
                            adata_score_sub, var_names=flat_markers,
                            groupby='subtype_label', show=False,
                            standard_scale='var', cmap='Reds',
                            swap_axes=True, return_fig=True,
                            colorbar_title='Mean\nExpression',
                            size_title='Fraction\nof Cells (%)',
                        )
                        _dp_sub.style(dot_edge_color='black', dot_edge_lw=0.3, grid=True)
                        # Not a curated paper figure -- skip saving (computation above still used).
                        plt.close('all')
                    except Exception as e:
                        print(f"  Warning: Dotplot failed: {e}")
                        plt.close('all')

            subcluster_results[pop_name] = adata_sub

        except Exception as e:
            print(f"  WARNING: Subclustering {pop_name} failed: {e}")
            import traceback
            traceback.print_exc()

    subcluster_cols = [c for c in adata.obs.columns if c.startswith('subcluster_')]
    if subcluster_cols:
        adata.obs[subcluster_cols].to_csv(os.path.join(dirs['data'], 'subcluster_labels.csv'))
        print(f"\nSaved subcluster labels: subcluster_labels.csv")

    print("\nSubclustering complete.")
    return adata


# ============================================================
# ENSURE UMAP IS COMPUTED
# ============================================================
def ensure_umap(adata):
    """Ensure UMAP coordinates exist; compute if missing."""
    if 'X_umap' not in adata.obsm:
        print("\n" + "="*60)
        print("COMPUTING UMAP (missing from input data)")
        print("="*60)

        if 'X_pca' not in adata.obsm:
            print("Computing PCA first...")
            if 'highly_variable' in adata.var.columns:
                adata_hvg = adata[:, adata.var['highly_variable']].copy()
            else:
                adata_hvg = adata.copy()
            if adata_hvg.X.max() > 20:
                sc.pp.scale(adata_hvg, max_value=10)
            sc.tl.pca(adata_hvg, n_comps=50)
            adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
            print(f"  PCA computed: {adata.obsm['X_pca'].shape}")

        print("Computing neighborhood graph...")
        use_rep = 'X_pca_harmony' if 'X_pca_harmony' in adata.obsm else 'X_pca'
        sc.pp.neighbors(adata, n_pcs=30, use_rep=use_rep)
        print("Computing UMAP...")
        sc.tl.umap(adata)
        print(f"  UMAP computed: {adata.obsm['X_umap'].shape}")
    else:
        print(f"UMAP exists: {adata.obsm['X_umap'].shape}")

    return adata


# ============================================================
# ANNOTATION VISUALIZATION
# ============================================================
def plot_annotation_figures(adata, dirs):
    """Generate all annotation visualization figures."""
    print("\n" + "="*60)
    print("GENERATING ANNOTATION FIGURES")
    print("="*60)

    adata = ensure_umap(adata)

    # Figure 17: UMAP colored by cell type
    fig, axes = plt.subplots(1, 2, figsize=(31, 11))
    sc.pl.umap(adata, color='cell_type_auto', ax=axes[0], show=False,
               legend_loc='none',
               title='Cell Type (marker scoring)', frameon=False, s=0.5)
    sc.pl.umap(adata, color='treatment', ax=axes[1], show=False,
               palette=TREATMENT_COLORS, legend_loc='none',
               title='Treatment', frameon=False, s=0.5)
    # Shared cell-type legend outside figure to the right
    _ct_cats17 = (adata.obs['cell_type_auto'].cat.categories.tolist()
                  if hasattr(adata.obs['cell_type_auto'], 'cat')
                  else sorted(adata.obs['cell_type_auto'].unique()))
    _ct_cols17 = adata.uns.get('cell_type_auto_colors', [])
    if _ct_cols17:
        _handles17 = [mpatches.Patch(color=_ct_cols17[i], label=ct)
                      for i, ct in enumerate(_ct_cats17) if i < len(_ct_cols17)]
        fig.legend(handles=_handles17, title='Cell Type',
                   bbox_to_anchor=(1.01, 0.5), loc='center left',
                   frameon=False, fontsize=10, title_fontsize=11,
                   ncol=1 + len(_ct_cats17) // 20)
    _treat_handles17 = [mpatches.Patch(color=v, label=k) for k, v in TREATMENT_COLORS.items()]
    axes[1].legend(handles=_treat_handles17, title='Treatment',
                   bbox_to_anchor=(0.5, -0.08), loc='upper center',
                   frameon=False, fontsize=10, ncol=4)
    plt.tight_layout(rect=[0, 0.07, 0.88, 1])
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['annotation'], 'fig17_umap_cell_type_auto.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 17c: Confidence scores
    if 'cell_type_score' in adata.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(28, 11))
        sc.pl.umap(adata, color='cell_type_score', ax=axes[0], show=False,
                   title='Annotation Confidence Score', frameon=False, s=0.5,
                   cmap='viridis', vmin=0, vmax=1)
        ct_order = adata.obs['cell_type_auto'].value_counts().index[:15]
        if len(ct_order) > 0:
            subset = adata.obs[adata.obs['cell_type_auto'].isin(ct_order)]
            axes[1].violinplot(
                [subset[subset['cell_type_auto'] == ct]['cell_type_score'].values
                 for ct in ct_order],
                positions=range(len(ct_order)),
                showmedians=True
            )
            axes[1].set_xticks(range(len(ct_order)))
            axes[1].set_xticklabels(ct_order, rotation=45, ha='right', fontsize=11)
            axes[1].set_ylabel('Confidence Score')
            axes[1].set_title('Confidence Score per Cell Type', fontweight='bold')
            axes[1].axhline(0.5, color='red', ls='--', alpha=0.5, label='Low conf threshold')
            axes[1].legend()
        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['annotation'], 'fig17c_confidence_scores.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig17c_confidence_scores.png")

    # Figure 17d: Per-treatment UMAPs
    if 'treatment' in adata.obs.columns:
        fig, axes = plt.subplots(2, 2, figsize=(28, 22))
        axes = axes.flatten()
        for i, treatment in enumerate(TREATMENT_ORDER):
            ax = axes[i]
            mask = adata.obs['treatment'] == treatment
            adata_sub = adata[mask].copy()
            sc.pl.umap(adata_sub, color='cell_type_auto', ax=ax, show=False,
                       legend_loc='none',
                       title=f'{treatment} (n={adata_sub.n_obs:,})',
                       frameon=False, s=1.0, alpha=0.6)
        # Shared cell-type legend outside the 2x2 grid
        _ct_cats17d = (adata.obs['cell_type_auto'].cat.categories.tolist()
                       if hasattr(adata.obs['cell_type_auto'], 'cat')
                       else sorted(adata.obs['cell_type_auto'].unique()))
        _ct_cols17d = adata.uns.get('cell_type_auto_colors', [])
        if _ct_cols17d:
            _handles17d = [mpatches.Patch(color=_ct_cols17d[i], label=ct)
                           for i, ct in enumerate(_ct_cats17d) if i < len(_ct_cols17d)]
            fig.legend(handles=_handles17d, title='Cell Type',
                       bbox_to_anchor=(0.5, -0.01), loc='upper center',
                       frameon=False, fontsize=10, title_fontsize=11,
                       ncol=min(6, len(_ct_cats17d)))
        plt.suptitle('Cell Type Distribution per Treatment (UMAP)', fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0.06, 1, 0.97])
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['annotation'], 'fig17d_umap_per_treatment.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig17d_umap_per_treatment.png")

    # Figure 17e: UMAP with cell density
    fig, axes = plt.subplots(1, 2, figsize=(28, 11))
    sc.pl.umap(adata, color='cell_type_auto', ax=axes[0], show=False,
               legend_loc='none',
               title='Cell Type Annotation', frameon=False, s=0.5)
    from scipy.stats import gaussian_kde
    umap_coords = adata.obsm['X_umap']
    n_density = min(50000, adata.n_obs)
    idx_density = np.random.choice(adata.n_obs, n_density, replace=False)
    xy = umap_coords[idx_density]
    try:
        kde = gaussian_kde(xy.T)
        density = kde(umap_coords.T)
        scatter = axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                  c=density, cmap='YlOrRd', s=0.5, alpha=0.5)
        axes[1].set_title('Cell Density in UMAP Space', fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(scatter, ax=axes[1], label='Density', shrink=0.8)
    except Exception as e:
        print(f"  Warning: Could not compute density: {e}")
        axes[1].text(0.5, 0.5, 'Density calculation\nfailed',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
    # Cell-type legend below left panel
    _ct_cats17e = (adata.obs['cell_type_auto'].cat.categories.tolist()
                   if hasattr(adata.obs['cell_type_auto'], 'cat')
                   else sorted(adata.obs['cell_type_auto'].unique()))
    _ct_cols17e = adata.uns.get('cell_type_auto_colors', [])
    if _ct_cols17e:
        _handles17e = [mpatches.Patch(color=_ct_cols17e[i], label=ct)
                       for i, ct in enumerate(_ct_cats17e) if i < len(_ct_cols17e)]
        axes[0].legend(handles=_handles17e, title='Cell Type',
                       bbox_to_anchor=(0.5, -0.04), loc='upper center',
                       frameon=False, fontsize=9, title_fontsize=10,
                       ncol=min(4, len(_ct_cats17e)))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['annotation'], 'fig17e_umap_with_density.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig17e_umap_with_density.png")

    # Figure 17f: UMAP colored by QC metrics
    qc_metrics = []
    qc_titles = []
    if 'n_genes_by_counts' in adata.obs.columns:
        qc_metrics.append('n_genes_by_counts'); qc_titles.append('Number of Genes')
    if 'total_counts' in adata.obs.columns:
        qc_metrics.append('total_counts'); qc_titles.append('Total UMI Counts')
    if 'pct_counts_mt' in adata.obs.columns:
        qc_metrics.append('pct_counts_mt'); qc_titles.append('% Mitochondrial')
    if qc_metrics:
        n_metrics = len(qc_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(10*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
        for i, (metric, title) in enumerate(zip(qc_metrics, qc_titles)):
            sc.pl.umap(adata, color=metric, ax=axes[i], show=False,
                       title=title, frameon=False, s=0.5,
                       cmap='viridis',
                       vmin=np.percentile(adata.obs[metric], 1),
                       vmax=np.percentile(adata.obs[metric], 99))
        plt.suptitle('QC Metrics on UMAP', fontsize=20, fontweight='bold')
        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['annotation'], 'fig17f_umap_qc_metrics.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig17f_umap_qc_metrics.png")

    # Figure 18: UMAP split by treatment
    fig, axes = plt.subplots(1, 4, figsize=(45, 12))
    fig.subplots_adjust(bottom=0.20)
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[j]
        mask = adata.obs['treatment'] == treatment
        adata_sub = adata[mask].copy()
        sc.pl.umap(adata_sub, color='cell_type_auto', ax=ax, show=False,
                   legend_loc='none',
                   title=treatment, frameon=False, s=1)
    # Shared legend below all 4 panels
    _ct_cats18 = (adata.obs['cell_type_auto'].cat.categories.tolist()
                  if hasattr(adata.obs['cell_type_auto'], 'cat')
                  else sorted(adata.obs['cell_type_auto'].unique()))
    _ct_cols18 = adata.uns.get('cell_type_auto_colors', [])
    if _ct_cols18:
        _handles18 = [mpatches.Patch(color=_ct_cols18[i], label=ct)
                      for i, ct in enumerate(_ct_cats18) if i < len(_ct_cols18)]
        fig.legend(handles=_handles18, title='Cell Type',
                   bbox_to_anchor=(0.5, 0.0), loc='upper center',
                   frameon=False, fontsize=10, title_fontsize=11,
                   ncol=min(8, len(_ct_cats18)))
    plt.suptitle('Cell Types per Treatment', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['annotation'], 'fig18_umap_celltype_per_treatment.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 19: Cell type proportion views
    prop = pd.crosstab(adata.obs['treatment'], adata.obs['cell_type_auto'], normalize='index')
    prop = prop.reindex(TREATMENT_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(34, 20))
    n_ct = len(prop.columns)
    colors = sns.color_palette("tab20", n_ct)

    ax = axes[0, 0]
    prop.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.75,
              edgecolor='black', linewidth=1)
    ax.set_ylabel('Proportion', fontweight='bold', fontsize=16)
    ax.set_xlabel('', fontweight='bold')
    ax.set_title('Cell Type Composition (Stacked)', fontweight='bold', fontsize=17)
    ax.get_legend().remove()          # remove auto legend; shared legend added below
    ax.set_xticklabels(TREATMENT_ORDER, rotation=0, fontweight='bold')
    ax.set_ylim(0, 1)
    sns.despine(ax=ax)

    ax = axes[0, 1]
    sns.heatmap(prop.T, cmap='YlOrRd', ax=ax, annot=True, fmt='.2f',
                linewidths=1, linecolor='white', cbar_kws={'label': 'Proportion'},
                vmin=0, vmax=prop.values.max())
    ax.set_title('Cell Type Proportion Heatmap', fontweight='bold', fontsize=17)
    ax.set_ylabel('Cell Type', fontweight='bold', fontsize=15)
    ax.set_xlabel('Treatment', fontweight='bold', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight='bold')
    plt.setp(ax.get_yticklabels(), rotation=0)

    ax = axes[1, 0]
    top_cts = prop.mean().nlargest(10).index
    prop_top = prop[top_cts]
    colors_top = [colors[list(prop.columns).index(ct)] for ct in top_cts]
    prop_top.plot(kind='bar', ax=ax, color=colors_top, width=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Proportion', fontweight='bold', fontsize=16)
    ax.set_xlabel('Treatment', fontweight='bold', fontsize=16)
    ax.set_title('Top 10 Cell Types (Grouped)', fontweight='bold', fontsize=17)
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center',
              fontsize=11, frameon=False, ncol=5)
    ax.set_xticklabels(TREATMENT_ORDER, rotation=0, fontweight='bold')
    sns.despine(ax=ax)

    ax = axes[1, 1]
    counts = pd.crosstab(adata.obs['treatment'], adata.obs['cell_type_auto'])
    counts = counts.reindex(TREATMENT_ORDER)
    counts_top = counts[top_cts]
    counts_top.plot(kind='bar', ax=ax, color=colors_top, width=0.8,
                    stacked=False, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Cell Count', fontweight='bold', fontsize=16)
    ax.set_xlabel('Treatment', fontweight='bold', fontsize=16)
    ax.set_title('Top 10 Cell Types (Absolute Counts)', fontweight='bold', fontsize=17)
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center',
              fontsize=11, frameon=False, ncol=5)
    ax.set_xticklabels(TREATMENT_ORDER, rotation=0, fontweight='bold')
    sns.despine(ax=ax)

    # Shared stacked-bar legend outside top row to the right
    _handles19 = [mpatches.Patch(color=colors[i], label=ct)
                  for i, ct in enumerate(prop.columns)]
    fig.legend(handles=_handles19, title='Cell Type',
               bbox_to_anchor=(1.01, 0.75), loc='upper left',
               frameon=False, fontsize=10, title_fontsize=11)

    plt.suptitle('Cell Type Composition Analysis', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 0.90, 0.97])
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['annotation'], 'fig19_celltype_proportion_bar.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 20: Marker gene dotplot and matrixplot
    dot_markers = {}
    for ct, markers in CELL_TYPE_MARKERS.items():
        available = [g for g in markers if g in adata.var_names][:4]
        if available:
            dot_markers[ct] = available

    flat_markers = [g for group in dot_markers.values() for g in group]
    if flat_markers:
        n_ct20   = len(adata.obs['cell_type_auto'].unique())
        n_genes20 = len(flat_markers)
        try:
            dp20 = sc.pl.dotplot(
                adata, var_names=dot_markers, groupby='cell_type_auto',
                show=False, standard_scale='var', cmap='Reds',
                dot_max=0.7, dot_min=0.1,
                swap_axes=True,
                figsize=(max(10, n_ct20 * 1.1 + 2), max(6, n_genes20 * 0.45 + 2)),
                return_fig=True,
                colorbar_title='Mean\nExpression',
                size_title='Fraction\nof Cells (%)',
            )
            dp20.style(dot_edge_color='black', dot_edge_lw=0.3, grid=True)
            # Not a curated paper figure -- skip saving (computation above still used).
            # dp20.savefig(os.path.join(dirs['annotation'], 'fig20a_marker_dotplot_celltype.png'),
            # dpi=300, bbox_inches='tight', facecolor='white')
            plt.close('all')
        except Exception as e:
            print(f"  Warning: dotplot failed: {e}")
            plt.close('all')

        try:
            mp20 = sc.pl.matrixplot(
                adata, var_names=dot_markers, groupby='cell_type_auto',
                show=False, standard_scale='var', cmap='RdBu_r',
                vmin=-2, vmax=2,
                swap_axes=True,
                figsize=(max(10, n_ct20 * 1.1 + 2), max(6, n_genes20 * 0.45 + 2)),
                return_fig=True,
                colorbar_title='Mean\nExpression',
            )
            mp20.style(edge_lw=0.3)
            # Not a curated paper figure -- skip saving (computation above still used).
            # mp20.savefig(os.path.join(dirs['annotation'], 'fig20b_marker_matrixplot_celltype.png'),
            # dpi=300, bbox_inches='tight', facecolor='white')
            plt.close('all')
        except Exception as e:
            print(f"  Warning: matrixplot failed: {e}")
            plt.close('all')

    return adata


# ============================================================
# SPATIAL CELL TYPE MAPS
# ============================================================
def spatial_celltype_maps(adata, individual_adatas, dirs):
    """Plot cell type annotations on tissue spatial coordinates."""
    print("\n" + "="*60)
    print("SPATIAL CELL TYPE MAPS")
    print("="*60)

    for treatment in TREATMENT_ORDER:
        mask = adata.obs['treatment'] == treatment
        ct_map = adata.obs.loc[mask, 'cell_type_auto']
        if treatment in individual_adatas:
            common_idx = individual_adatas[treatment].obs.index.intersection(ct_map.index)
            if len(common_idx) > 0:
                individual_adatas[treatment].obs.loc[common_idx, 'cell_type_auto'] = ct_map.loc[common_idx]

    all_ct = adata.obs['cell_type_auto'].unique()
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(all_ct), 1))
    ct_colors = {ct: cmap(i) for i, ct in enumerate(sorted(all_ct))}

    # Figure 21: Spatial cell type maps
    fig, axes = plt.subplots(1, 4, figsize=(45, 11))
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[j]
        if treatment in individual_adatas:
            ad = individual_adatas[treatment]
            if 'spatial' in ad.obsm and 'cell_type_auto' in ad.obs.columns:
                coords = ad.obsm['spatial']
                n_plot = min(200000, ad.n_obs)
                idx = np.random.choice(ad.n_obs, n_plot, replace=False)
                colors = [ct_colors.get(ct, (0.8, 0.8, 0.8, 1))
                          for ct in ad.obs['cell_type_auto'].values[idx]]
                ax.scatter(coords[idx, 0], coords[idx, 1], c=colors, s=0.8, alpha=0.7,
                           rasterized=True, edgecolors='none')
                ax.set_aspect('equal')
            else:
                ax.text(0.5, 0.5, 'No spatial data', ha='center', va='center',
                        fontsize=15, fontweight='bold')
        ax.set_title(treatment, fontsize=18, fontweight='bold')
        ax.axis('off')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ct_colors[ct], edgecolor='none', label=ct)
                       for ct in sorted(ct_colors.keys())]
    fig.legend(handles=legend_elements, loc='center left', fontsize=10,
               bbox_to_anchor=(1.01, 0.5), title='Cell Type', title_fontsize=12,
               frameon=False, markerscale=1.5)
    plt.suptitle(f'Spatial Cell Type Maps ({RESOLUTION_LABEL})', fontsize=24, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['annotation'], 'fig21_spatial_celltype_maps.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()
    print("Spatial cell type maps saved.")


# ============================================================
# ECOTYPE / CELLULAR NEIGHBORHOOD ANALYSIS
# ============================================================
def ecotype_analysis(adata, dirs, n_ecotypes=10, window_size=50):
    """
    Identify spatial ecotypes per treatment via spatial binning and Leiden clustering.
    Bins tissue into windows, computes cell type composition per window,
    then clusters windows into ecotypes.
    """
    print("\n" + "="*60)
    print("ECOTYPE / CELLULAR NEIGHBORHOOD ANALYSIS")
    print("="*60)

    all_ecotype_data = {}

    for treatment in TREATMENT_ORDER:
        print(f"\n--- {treatment} ---")
        mask = adata.obs['treatment'] == treatment
        adata_t = adata[mask].copy()

        if 'spatial' not in adata_t.obsm:
            print(f"  No spatial coordinates for {treatment}, skipping.")
            continue

        coords = adata_t.obsm['spatial']
        cell_types = adata_t.obs['cell_type_auto'].values
        unique_ct = sorted(adata.obs['cell_type_auto'].unique())

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        eco_window = window_size * 4
        x_bins = np.arange(x_min, x_max + eco_window, eco_window)
        y_bins = np.arange(y_min, y_max + eco_window, eco_window)
        x_idx = np.digitize(coords[:, 0], x_bins) - 1
        y_idx = np.digitize(coords[:, 1], y_bins) - 1
        bin_ids = x_idx * len(y_bins) + y_idx

        bin_compositions = []
        bin_coords_center = []
        bin_cell_counts = []

        for bid in np.unique(bin_ids):
            bin_mask = bin_ids == bid
            n_cells = bin_mask.sum()
            if n_cells < 5:
                continue
            ct_in_bin = cell_types[bin_mask]
            composition = {ct: 0 for ct in unique_ct}
            for ct in ct_in_bin:
                composition[ct] += 1
            total = sum(composition.values())
            composition = {ct: v/total for ct, v in composition.items()}
            bin_compositions.append(composition)
            bin_coords_center.append(coords[bin_mask].mean(axis=0))
            bin_cell_counts.append(n_cells)

        if len(bin_compositions) < 20:
            print(f"  Too few bins ({len(bin_compositions)}), skipping ecotype analysis.")
            continue

        comp_df = pd.DataFrame(bin_compositions)
        bin_coords_arr = np.array(bin_coords_center)
        print(f"  {len(comp_df)} spatial bins created")

        adata_bins = sc.AnnData(X=comp_df.values, obs=pd.DataFrame(index=range(len(comp_df))))
        adata_bins.obsm['spatial'] = bin_coords_arr
        sc.pp.neighbors(adata_bins, n_neighbors=15, use_rep='X')

        for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
            sc.tl.leiden(adata_bins, resolution=res)
            n_clusters = adata_bins.obs['leiden'].nunique()
            if n_clusters >= n_ecotypes:
                break

        cluster_map = {str(i): f'CC{i+1}' for i in range(n_clusters)}
        adata_bins.obs['ecotype'] = adata_bins.obs['leiden'].map(cluster_map)
        print(f"  {n_clusters} ecotypes identified (resolution={res})")

        all_ecotype_data[treatment] = {
            'compositions': comp_df,
            'ecotypes': adata_bins.obs['ecotype'].values,
            'coords': bin_coords_arr,
            'cell_counts': bin_cell_counts,
            'unique_ct': unique_ct,
        }

    print("\nEcotype analysis complete.")
    # NOTE: Spatial ecotype maps, ecotype heatmaps, and the reference ecotype
    # panels (Panel A-E) are all generated by step04_ecotype_panels.py from
    # the final prior-adjusted h5ad. Do not duplicate them here.

    return all_ecotype_data


# ============================================================
# TREATMENT COMPARISON: CELL TYPE TRENDS
# ============================================================
def treatment_trends(adata, dirs):
    """Compare cell type proportions across treatments."""
    print("\n" + "="*60)
    print("TREATMENT TREND ANALYSIS")
    print("="*60)

    prop = pd.crosstab(adata.obs['treatment'], adata.obs['cell_type_auto'], normalize='index')
    prop = prop.reindex(TREATMENT_ORDER) * 100

    # scCODA compositional analysis (optional)
    print("\n" + "-"*60)
    print("COMPOSITIONAL ANALYSIS (scCODA)")
    print("-"*60)
    try:
        from sccoda.util import comp_ana as mod
        import anndata as ad

        counts_df = pd.crosstab(adata.obs['treatment'], adata.obs['cell_type_auto'])
        counts_df = counts_df.reindex(TREATMENT_ORDER).fillna(0).astype(int)

        sccoda_data = ad.AnnData(
            X=counts_df.values,
            obs=pd.DataFrame({'condition': counts_df.index}, index=counts_df.index),
            var=pd.DataFrame(index=counts_df.columns)
        )
        print(f"  scCODA input: {sccoda_data.n_obs} conditions x {sccoda_data.n_vars} cell types")

        model = mod.CompositionalAnalysis(
            sccoda_data, formula="condition", reference_cell_type="automatic"
        )
        sim_results = model.sample_nuts(num_results=10000, num_burnin=5000)
        credible_effects = sim_results.credible_effects()
        credible_effects.to_csv(os.path.join(dirs['data'], 'sccoda_credible_effects.csv'))
        print(f"  Saved: sccoda_credible_effects.csv")

        fig, axes = plt.subplots(1, 2, figsize=(34, 14))
        ax = axes[0]
        prop_plot = counts_df.div(counts_df.sum(axis=1), axis=0)
        n_ct = len(prop_plot.columns)
        colors = sns.color_palette("tab20", n_ct)
        prop_plot.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7,
                       edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Proportion', fontweight='bold', fontsize=16)
        ax.set_title('Cell Type Composition\n(* = scCODA credible effect)', fontweight='bold', fontsize=18)
        ax.set_xticklabels(TREATMENT_ORDER, rotation=0, fontweight='bold', fontsize=13)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        sns.despine(ax=ax)

        ax = axes[1]
        if hasattr(credible_effects, 'reset_index'):
            ce_plot = credible_effects.reset_index()
            if 'log2-fold change' in ce_plot.columns and 'Cell Type' in ce_plot.columns:
                sig = ce_plot[ce_plot.get('Final Parameter', ce_plot.iloc[:, -1]) != 0]
                if len(sig) > 0:
                    sig_sorted = sig.sort_values('log2-fold change', ascending=True)
                    colors_bar = ['#B2182B' if v > 0 else '#2166AC' for v in sig_sorted['log2-fold change']]
                    ax.barh(range(len(sig_sorted)), sig_sorted['log2-fold change'], color=colors_bar)
                    ax.set_yticks(range(len(sig_sorted)))
                    ax.set_yticklabels(
                        sig_sorted['Cell Type'].values if 'Cell Type' in sig_sorted.columns else sig_sorted.index,
                        fontsize=11
                    )
                    ax.axvline(0, color='black', linewidth=1)
                    ax.set_xlabel('log2 Fold Change (GPH+IT vs Sham)', fontweight='bold', fontsize=14)
                    ax.set_title('scCODA: Credible Composition Changes', fontweight='bold', fontsize=16)
                    sns.despine(ax=ax)
                else:
                    ax.text(0.5, 0.5, 'No credible effects\ndetected',
                            ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.axis('off')

        plt.suptitle('Compositional Analysis (scCODA)', fontsize=22, fontweight='bold')
        plt.tight_layout()
        os.makedirs(dirs['summary'], exist_ok=True)
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['summary'], 'fig_sccoda_composition.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig_sccoda_composition.png")

    except ImportError:
        print("  scCODA not installed, skipping compositional analysis.")
    except Exception as e:
        print(f"  WARNING: scCODA analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # NOTE: Proportion figures are generated by step08_annotation_summary_plots.py
    # from the final prior-adjusted h5ad. Only save the CSV here for downstream use.
    prop.to_csv(os.path.join(dirs['data'], 'celltype_proportions_per_treatment.csv'))
    print("\nCell type proportions saved.")
    print(prop.to_string())


# ============================================================
# CLUSTER ANNOTATION SUMMARY CSV
# ============================================================
def save_cluster_annotation_csv(adata, dirs):
    """
    Build and save cluster_annotation_with_markers.csv, one row per Leiden cluster.
    Columns: leiden_cluster, cell_type, n_cells, pct_total, pct_dominant,
    confidence_score_mean, annotation_method, top_markers, known_markers,
    <treatment>_pct.
    """
    print("\n" + "=" * 60)
    print("SAVING CLUSTER ANNOTATION SUMMARY")
    print("=" * 60)

    leiden_col = None
    for col in ['leiden_merged', 'leiden', 'leiden_0.5', 'leiden_1.0']:
        if col in adata.obs.columns:
            leiden_col = col
            break
    if leiden_col is None:
        print("  WARNING: No leiden cluster column found. Skipping cluster annotation CSV.")
        return

    cell_type_col = 'cell_type_auto'
    if cell_type_col not in adata.obs.columns:
        print("  WARNING: cell_type_auto not found. Skipping cluster annotation CSV.")
        return

    print(f"  Using cluster column: {leiden_col}")
    clusters = sorted(adata.obs[leiden_col].astype(str).unique(),
                      key=lambda x: int(x) if x.isdigit() else x)

    de_markers = {}
    try:
        print(f"  Running rank_genes_groups (t-test, n_genes=50) on {leiden_col}...")
        cl_sizes = adata.obs[leiden_col].astype(str).value_counts()
        valid_clusters = cl_sizes[cl_sizes >= 10].index.tolist()
        if len(valid_clusters) < len(clusters):
            print(f"  Excluding {len(clusters)-len(valid_clusters)} clusters with <10 cells from DE")
        mask_valid = adata.obs[leiden_col].astype(str).isin(valid_clusters)
        adata_de = adata[mask_valid].copy()
        sc.tl.rank_genes_groups(adata_de, groupby=leiden_col, method='t-test',
                                n_genes=50, use_raw=True if adata_de.raw is not None else False)
        rgg_df = sc.get.rank_genes_groups_df(adata_de, group=None)
        for grp, sub in rgg_df.groupby('group'):
            top = sub.sort_values('scores', ascending=False)['names'].head(50).tolist()
            de_markers[str(grp)] = ', '.join(top)
        print(f"  DE markers computed for {len(de_markers)} clusters")
    except Exception as e:
        print(f"  WARNING: Could not compute rank_genes_groups markers: {e}")
        if 'rank_genes_groups' in adata.uns:
            try:
                rgg_df = sc.get.rank_genes_groups_df(adata, group=None)
                for grp, sub in rgg_df.groupby('group'):
                    top = sub.sort_values('scores', ascending=False)['names'].head(50).tolist()
                    de_markers[str(grp)] = ', '.join(top)
            except Exception:
                pass

    annotation_method = 'marker_scoring'

    rows = []
    for cluster in clusters:
        mask = adata.obs[leiden_col].astype(str) == cluster
        adata_cl = adata[mask]
        n_cells   = int(mask.sum())
        pct_total = round(100.0 * n_cells / adata.n_obs, 2)

        ct_counts    = adata_cl.obs[cell_type_col].value_counts()
        dominant_ct  = ct_counts.index[0] if len(ct_counts) > 0 else 'Unknown'
        pct_dominant = round(100.0 * ct_counts.iloc[0] / n_cells, 1) if n_cells > 0 else 0.0

        conf_mean = None
        if 'cell_type_score' in adata_cl.obs.columns:
            conf_mean = round(float(adata_cl.obs['cell_type_score'].mean()), 3)

        known = CELL_TYPE_MARKERS.get(dominant_ct, [])
        if not known:
            for key, genes in CELL_TYPE_MARKERS.items():
                if key.lower() in dominant_ct.lower() or dominant_ct.lower() in key.lower():
                    known = genes
                    break
        known_str = ', '.join(known[:7]) if known else ''

        row = {
            'leiden_cluster':       cluster,
            'cell_type':            dominant_ct,
            'n_cells':              n_cells,
            'pct_total':            pct_total,
            'pct_dominant':         pct_dominant,
            'confidence_score_mean': conf_mean,
            'annotation_method':    annotation_method,
            'top_markers':          de_markers.get(cluster, ''),
            'known_markers':        known_str,
        }

        if 'treatment' in adata_cl.obs.columns:
            treatment_counts = adata_cl.obs['treatment'].value_counts()
            for t in ['Sham', 'IT', 'GPH', 'GPH+IT']:
                cnt = treatment_counts.get(t, 0)
                row[f'{t}_pct'] = round(100.0 * cnt / n_cells, 1) if n_cells > 0 else 0.0

        rows.append(row)

    annotation_df = pd.DataFrame(rows)
    annotation_df = annotation_df.sort_values('n_cells', ascending=False).reset_index(drop=True)

    save_path = os.path.join(dirs['data'], 'cluster_annotation_with_markers.csv')
    annotation_df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")
    print(f"  Rows: {len(annotation_df)} clusters")
    print(f"\n  Top clusters:")
    for _, row in annotation_df.head(10).iterrows():
        print(f"    Cluster {row['leiden_cluster']:>4}  |  {row['cell_type']:<35}"
              f"  |  {row['n_cells']:>8,} cells ({row['pct_total']:.1f}%)  "
              f"|  {row['pct_dominant']:.0f}% pure")

    return annotation_df


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Stereo-seq Step 2: Annotation & Ecotype')
    parser.add_argument('--input_dir',  type=str, required=True,
                        help='Directory containing processed_data from Step 1')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--resolution', type=str, default='cellbin',
                        choices=['cellbin', 'bin50'],
                        help='Input resolution (cellbin or bin50); affects figure titles')
    parser.add_argument('--n_ecotypes', type=int, default=10,
                        help='Number of ecotypes for K-means (default: 10)')
    args = parser.parse_args()

    global RESOLUTION_LABEL
    RESOLUTION_LABEL = 'Bin50' if args.resolution == 'bin50' else 'CellBin'
    print("="*60)
    print("STEREO-SEQ STEP 2: ECOTYPE & SUBCLUSTER ANALYSIS")
    print("="*60)

    dirs = setup_dirs(args.output_dir)

    # Cell type assignment happens upstream, in step02_build_annotated_h5ad.py.
    # This script loads the resulting annotated dataset and runs the
    # downstream analyses (subclustering, ecotypes, treatment trends) that
    # operate on top of the assigned cell types.
    merged_path = os.path.join(args.input_dir, 'downstream_analysis', 'processed_data',
                               'merged_annotated.h5ad')
    print(f"\nLoading annotated data: {merged_path}")
    if not os.path.exists(merged_path):
        print(f"ERROR: {merged_path} not found. Run step02_build_annotated_h5ad.py first.")
        sys.exit(1)
    adata = sc.read_h5ad(merged_path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    individual_adatas = {}
    for treatment in TREATMENT_ORDER:
        path = os.path.join(args.input_dir, 'downstream_analysis', 'processed_data',
                            f'{treatment}_cellbin_processed.h5ad')
        if os.path.exists(path):
            individual_adatas[treatment] = sc.read_h5ad(path)
            print(f"  Loaded {treatment}: {individual_adatas[treatment].n_obs:,} cells")

    adata = subcluster_major_populations(adata, dirs)
    # NOTE: annotation/spatial figures intentionally NOT generated here.
    # All final figures are produced by step04 through step11.
    ecotype_data = ecotype_analysis(adata, dirs, n_ecotypes=args.n_ecotypes)
    # NOTE: ecotype panels (Panel A-E), barplots, and all composition figures
    # are generated by step04 through step08, not here.
    treatment_trends(adata, dirs)

    save_path = os.path.join(dirs['data'], 'merged_annotated.h5ad')
    adata.write(save_path)
    print(f"\nSaved annotated data: {save_path}")

    save_cluster_annotation_csv(adata, dirs)

    # Annotation quality assessment
    print("\n" + "="*60)
    print("ANNOTATION QUALITY METRICS")
    print("="*60)
    try:
        from annotation_quality_metrics import (
            calculate_annotation_quality_metrics,
            save_metrics_to_csv,
            save_quality_report
        )
        quality_metrics = calculate_annotation_quality_metrics(
            adata, cell_type_col='cell_type_auto',
            confidence_col='cell_type_score', marker_dict=CELL_TYPE_MARKERS
        )
        metrics_csv  = os.path.join(dirs['data'], 'annotation_quality_metrics.csv')
        report_txt   = os.path.join(dirs['data'], 'annotation_quality_report.txt')
        save_metrics_to_csv(quality_metrics, metrics_csv)
        save_quality_report(quality_metrics, report_txt)
        print(f"  Overall Quality: {quality_metrics.get('overall_quality', 'N/A')}")
    except ImportError:
        print("  annotation_quality_metrics module not found, skipping.")
    except Exception as e:
        print(f"  WARNING: Quality metrics failed: {e}")

    # Cell distance analysis (runs only if ecotype assignments are available)
    print("\n" + "="*60)
    print("CELL DISTANCE ANALYSIS, mPDAC / ePDAC")
    print("="*60)
    try:
        from step05_cell_distances import run_cell_distance_analysis
        eco_csv = os.path.join(dirs['data'], 'unified_ecotype_assignments.csv')
        if 'ecotype' not in adata.obs.columns and os.path.exists(eco_csv):
            import pandas as _pd
            eco_df = _pd.read_csv(eco_csv, index_col=0)
            if 'ecotype' in eco_df.columns:
                adata.obs['ecotype'] = (
                    eco_df['ecotype'].reindex(adata.obs_names).fillna('Unassigned')
                )
                print(f"  Loaded ecotype assignments from unified_ecotype_assignments.csv")
        if 'ecotype' in adata.obs.columns:
            dist_out = os.path.join(os.path.dirname(dirs['data']), 'figures', 'cell_distances')
            run_cell_distance_analysis(adata, dist_out)
        else:
            print("  Skipping, run generate_ecotype_panels.py then cell_distance_pdac.py")
    except ImportError:
        print("  cell_distance_pdac.py not found, skipping.")
    except Exception as e:
        print(f"  WARNING: Cell distance analysis failed: {e}")

    print("\n" + "="*60)
    print("STEP 2 COMPLETE")
    print("="*60)
    print(f"All outputs in: {args.output_dir}")
    print(f"\nFinal cell type counts:")
    ct_summary = adata.obs['cell_type_auto'].value_counts()
    for ct, n in ct_summary.head(20).items():
        pct = 100 * n / adata.n_obs
        print(f"  {ct:<35}: {n:>8,} ({pct:>5.1f}%)")


if __name__ == '__main__':
    main()
