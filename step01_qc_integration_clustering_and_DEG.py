#!/usr/bin/env python3
"""
Stereo-seq CellBin Analysis Pipeline, PDAC 4-Treatment Comparison

Samples: 01SX/B04664A3 → Sham, 02SX/B04751G2 → GPH,
         03SX/B04750G3 → IT,   04SX/B04667G2 → GPH+IT

Input:  *.cellbin_1.0.adjusted.h5ad files from SAW 8.1
Output: Figures and processed h5ad files under downstream_analysis/

"""

import os

RESOLUTION_LABEL = 'CellBin'  # overwritten in main() based on --resolution arg

# DEG figure content. These are DATA choices and must not be quietly reduced to
# solve a layout problem, size the figure to the genes, not the genes to the
# figure. Counts are per direction, so the totals are 2x these.
N_DEG_BAR = 15        # -> "Top 30 DEGs by Significance"
N_VOLCANO_LABEL = 6   # -> 12 labelled genes on the volcano
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from scipy import stats
from statannotations.Annotator import Annotator
import pickle
import json
from pathlib import Path
import gc

sc.settings.n_jobs = -1

# ============================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================
plt.rcParams.update({
    # Arial is not installed on the cluster; Liberation Sans is metric-
    # compatible with it and Nimbus Sans is a Helvetica clone, so the chain
    # lands on an Arial/Helvetica-equivalent wherever it runs. Naming a font
    # that does not exist silently falls back to DejaVu instead.
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica',
                        'Nimbus Sans', 'FreeSans', 'DejaVu Sans'],
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
# Both seaborn calls overwrite font.family/font.sans-serif with seaborn's own
# defaults, which is how DejaVu ended up embedded in the volcano and DEG-bar
# PDFs despite the chain set above. Re-assert it after them.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'Helvetica',
                                   'Nimbus Sans', 'FreeSans', 'DejaVu Sans']

# ============================================================
# CONFIGURATION
# ============================================================
SAMPLES = {
    'Sham':   {'sample': '24201-02-01SX_flip-SAW8.1', 'bgi': 'B04664A3', 'color': '#2166AC'},
    'GPH':    {'sample': '24201-02-02SX_flip-SAW8.1', 'bgi': 'B04751G2', 'color': '#67A9CF'},
    'IT':     {'sample': '24201-02-03SX_flip-SAW8.1', 'bgi': 'B04750G3', 'color': '#EF8A62'},
    'GPH+IT': {'sample': '24201-02-04SX_flip-SAW8.1', 'bgi': 'B04667G2', 'color': '#B2182B'},
}


def sample_h5ad(info, resolution):
    # SAW outputs cellbin with an extra '.adjusted' (post-segmentation correction); bin50 has none.
    suffix = '.adjusted' if resolution == 'cellbin' else ''
    return f"{info['sample']}/analysis/{info['bgi']}.{resolution}_1.0{suffix}.h5ad"


def sample_image(info):
    return f"{info['sample']}/image/{info['bgi']}_ssDNA_regist.tif"

TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']

# NOTE: Stereo-seq CellBin cells have low gene diversity by design (median ~130-200 genes/cell).
# Primary quality filter is UMI-based (MIN_COUNTS), not gene-based.
MIN_GENES = 50         # Keep low: CellBin gene counts are inherently sparse
MAX_GENES = 6000
MIN_COUNTS = 500       # Primary quality gate for CellBin
MAX_MITO_PCT = 20      # Removes stressed/dying cells
MIN_CELLS_PER_GENE = 10


def setup_output(base_dir, output_dir=None):
    """Build the output directory paths.

    Only `data` and `checkpoints` are created here, since every code path
    writes to them right away. Each per-section figure folder
    (01_QC/02_normalization/.../06_composition) is created lazily by the
    function that actually writes into it, so a run that skips a section
    (e.g. --deg_only, --recluster_only) never leaves an empty folder behind.
    """
    if output_dir:
        out = os.path.join(output_dir, 'downstream_analysis')
    else:
        out = os.path.join(base_dir, 'downstream_analysis')
    dirs = {
        'figures': os.path.join(out, 'figures'),
        'qc': os.path.join(out, 'figures', '01_QC'),
        'norm': os.path.join(out, 'figures', '02_normalization'),
        'cluster': os.path.join(out, 'figures', '03_clustering'),
        'spatial': os.path.join(out, 'figures', '04_spatial'),
        'deg': os.path.join(out, 'figures', '05_DEG'),
        'composition': os.path.join(out, 'figures', '06_composition'),
        'data': os.path.join(out, 'processed_data'),
        'checkpoints': os.path.join(out, 'checkpoints'),
    }
    os.makedirs(dirs['data'], exist_ok=True)
    os.makedirs(dirs['checkpoints'], exist_ok=True)
    return dirs


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================
def save_checkpoint(step_name, data, dirs):
    """Save pipeline checkpoint."""
    checkpoint_file = os.path.join(dirs['checkpoints'], f'{step_name}.pkl')
    print(f"  → Saving checkpoint: {step_name}")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_file = os.path.join(dirs['checkpoints'], f'{step_name}.json')
    with open(meta_file, 'w') as f:
        json.dump({'step': step_name, 'timestamp': pd.Timestamp.now().isoformat(), 'completed': True}, f, indent=2)


def load_checkpoint(step_name, dirs):
    """Load pipeline checkpoint. Returns None if not found."""
    checkpoint_file = os.path.join(dirs['checkpoints'], f'{step_name}.pkl')
    if not os.path.exists(checkpoint_file):
        return None
    print(f"  ✓ Loading checkpoint: {step_name}")
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


def checkpoint_exists(step_name, dirs):
    """Check if a checkpoint file exists."""
    return os.path.exists(os.path.join(dirs['checkpoints'], f'{step_name}.pkl'))


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_samples(base_dir, resolution='cellbin'):
    """Load all 4 CellBin h5ad files, convert ENSMUSG IDs to gene symbols, add metadata."""
    adatas = {}
    for treatment, info in SAMPLES.items():
        path = os.path.join(base_dir, sample_h5ad(info, resolution))
        print(f"Loading {treatment}: {path}")
        adata = sc.read_h5ad(path)

        if 'real_gene_name' in adata.var.columns:
            print(f"  Converting ENSMUSG IDs to gene symbols via 'real_gene_name'...")
            adata.var['ensembl_id'] = adata.var_names.copy()
            symbols = adata.var['real_gene_name'].astype(str).copy()
            dup_mask = symbols.duplicated(keep=False)
            n_dups = dup_mask.sum()
            if n_dups > 0:
                symbols[dup_mask] = symbols[dup_mask] + '_' + adata.var_names[dup_mask]
                print(f"  Resolved {n_dups} duplicate gene symbols")
            empty_mask = (symbols == '') | (symbols == 'nan') | symbols.isna()
            symbols[empty_mask] = adata.var_names[empty_mask]
            adata.var_names = symbols.values
            adata.var_names_make_unique()
            print(f"  Gene names converted: ENSMUSG... → {adata.var_names[:3].tolist()} ...")

        adata.obs['treatment'] = treatment
        adata.obs['sample_id'] = info['sample']

        if 'spatial' not in adata.obsm:
            if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
                adata.obsm['spatial'] = adata.obs[['x', 'y']].values
            elif 'spatial_x' in adata.obs.columns:
                adata.obsm['spatial'] = adata.obs[['spatial_x', 'spatial_y']].values

        print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")

        print(f"\n  [Doublet Detection] Running Scrublet...")
        try:
            import scrublet as scr
            scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06, random_state=42)
            doublet_scores, predicted_doublets = scrub.scrub_doublets(
                min_counts=2, min_cells=3,
                min_gene_variability_pctile=85,
                n_prin_comps=min(30, adata.n_vars - 1)
            )
            adata.obs['doublet_score'] = doublet_scores
            adata.obs['predicted_doublet'] = predicted_doublets
            n_doublets = predicted_doublets.sum()
            print(f"  Detected {n_doublets:,} doublets ({100 * n_doublets / adata.n_obs:.1f}%), threshold={scrub.threshold_:.3f}")
        except ImportError:
            print("  WARNING: scrublet not installed. Skipping doublet detection.")
            adata.obs['doublet_score'] = 0.0
            adata.obs['predicted_doublet'] = False
        except Exception as e:
            print(f"  WARNING: Doublet detection failed: {e}")
            adata.obs['doublet_score'] = 0.0
            adata.obs['predicted_doublet'] = False

        adatas[treatment] = adata

    return adatas


# ============================================================
# STEP 2: QUALITY CONTROL
# ============================================================
def run_qc(adatas, dirs):
    """Calculate QC metrics and generate QC figures."""
    os.makedirs(dirs['qc'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 2: QUALITY CONTROL")
    print("="*60)

    for treatment, adata in adatas.items():
        adata.var['mt'] = adata.var_names.str.startswith(('mt-', 'MT-'))
        adata.var['ribo'] = adata.var_names.str.startswith(('Rps', 'Rpl', 'RPS', 'RPL'))
        n_mt = adata.var['mt'].sum()
        n_ribo = adata.var['ribo'].sum()
        print(f"\n{treatment}: detected {n_mt} mito genes, {n_ribo} ribo genes")

        sc.pp.calculate_qc_metrics(
            adata, qc_vars=['mt', 'ribo'],
            percent_top=None, log1p=False, inplace=True
        )

        print(f"\n{treatment} QC summary (before filtering):")
        print(f"  Cells: {adata.n_obs:,}")
        print(f"  Median genes/cell: {adata.obs['n_genes_by_counts'].median():.0f}")
        print(f"  Median UMI/cell: {adata.obs['total_counts'].median():.0f}")
        print(f"  Median %mito: {adata.obs['pct_counts_mt'].median():.1f}%")

    # Figure 1: Violin+Box QC plots (before filtering)
    fig, axes = plt.subplots(3, 4, figsize=(28, 17))
    metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    labels = ['Genes per cell', 'UMI per cell', '% Mitochondrial']

    for j, treatment in enumerate(TREATMENT_ORDER):
        adata = adatas[treatment]
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[i, j]
            data = adata.obs[metric].values
            if metric in ('total_counts', 'n_genes_by_counts'):
                data = np.clip(data, 0, np.percentile(data, 99))

            parts = ax.violinplot([data], positions=[0], showmedians=False,
                                  showextrema=False, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor(SAMPLES[treatment]['color'])
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
                pc.set_alpha(0.6)

            ax.boxplot([data], positions=[0], widths=0.15, patch_artist=True,
                       showfliers=False, showcaps=True,
                       boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2))

            mean_val = np.mean(data)
            ax.plot([0], [mean_val], marker='D', color='blue', markersize=8,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)

            median_val = np.median(data)
            ax.text(0.35, median_val, f'median: {median_val:.0f}', fontsize=11, va='center', ha='left')
            ax.text(0.35, mean_val, f'mean: {mean_val:.0f}', fontsize=11, va='center', ha='left', color='blue')

            ax.set_title(treatment if i == 0 else '', fontsize=15, fontweight='bold')
            ax.set_ylabel(label if j == 0 else '', fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            sns.despine(ax=ax, left=False, bottom=True)

            if metric == 'n_genes_by_counts':
                ax.axhline(MIN_GENES, color='#d62728', ls='--', alpha=0.7, lw=2)
                ax.axhline(MAX_GENES, color='#d62728', ls='--', alpha=0.7, lw=2)
                ax.text(-0.45, MIN_GENES, f'min={MIN_GENES}', fontsize=10, color='#d62728', va='bottom')
                ax.text(-0.45, MAX_GENES, f'max={MAX_GENES}', fontsize=10, color='#d62728', va='top')
            elif metric == 'total_counts':
                ax.axhline(MIN_COUNTS, color='#d62728', ls='--', alpha=0.7, lw=2)
                ax.text(-0.45, MIN_COUNTS, f'min={MIN_COUNTS}', fontsize=10, color='#d62728', va='bottom')
            elif metric == 'pct_counts_mt':
                ax.axhline(MAX_MITO_PCT, color='#d62728', ls='--', alpha=0.7, lw=2)
                ax.text(-0.45, MAX_MITO_PCT, f'max={MAX_MITO_PCT}%', fontsize=10, color='#d62728', va='top')

    plt.suptitle(f'QC Metrics Before Filtering ({RESOLUTION_LABEL})', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['qc'], 'fig01_qc_violin_before_filter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 1b: Doublet Score Distribution
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[j]
        adata = adatas[treatment]
        if 'doublet_score' in adata.obs.columns:
            scores = adata.obs['doublet_score'].values
            predicted = adata.obs['predicted_doublet'].values if 'predicted_doublet' in adata.obs.columns else np.zeros(len(scores), dtype=bool)

            parts = ax.violinplot([scores], positions=[0], showmedians=False,
                                  showextrema=False, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor(SAMPLES[treatment]['color'])
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
                pc.set_alpha(0.6)

            ax.boxplot([scores], positions=[0], widths=0.15, patch_artist=True,
                       showfliers=False, showcaps=True,
                       boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2))

            n_doublets = predicted.sum()
            pct_doublets = 100 * n_doublets / len(scores) if len(scores) > 0 else 0
            ax.set_title(f'{treatment}\n{n_doublets:,} doublets ({pct_doublets:.1f}%)',
                         fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No doublet\nscores', ha='center', va='center',
                    fontsize=14, fontweight='bold', transform=ax.transAxes)

        ax.set_ylabel('Doublet Score' if j == 0 else '', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax, left=False, bottom=True)

    plt.suptitle('Doublet Score Distribution per Treatment (Scrublet)', fontsize=20, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['qc'], 'fig01b_doublet_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: UMI vs Genes hexbin scatter
    fig, axes = plt.subplots(2, 4, figsize=(34, 17))
    for j, treatment in enumerate(TREATMENT_ORDER):
        adata = adatas[treatment]
        n_plot = min(50000, adata.n_obs)
        idx = np.random.choice(adata.n_obs, n_plot, replace=False)

        x = adata.obs['total_counts'].values[idx]
        y = adata.obs['n_genes_by_counts'].values[idx]
        c = adata.obs['pct_counts_mt'].values[idx]

        ax = axes[0, j]
        hexbin = ax.hexbin(x, y, C=c, gridsize=50, cmap='RdYlBu_r',
                           reduce_C_function=np.median, vmin=0, vmax=MAX_MITO_PCT,
                           mincnt=1, linewidths=0.2, edgecolors='face')
        ax.set_xlabel('Total UMI counts', fontweight='bold')
        ax.set_ylabel('Number of genes' if j == 0 else '', fontweight='bold')
        ax.set_title(treatment, fontsize=15, fontweight='bold')
        ax.axhline(MIN_GENES, color='#d62728', ls='--', alpha=0.8, lw=2)
        ax.axvline(MIN_COUNTS, color='#d62728', ls='--', alpha=0.8, lw=2)
        ax.set_xlim(0, np.percentile(adata.obs['total_counts'], 99))
        ax.set_ylim(0, np.percentile(adata.obs['n_genes_by_counts'], 99))
        if j == 3:
            cbar = plt.colorbar(hexbin, ax=ax, label='Median % Mito', shrink=0.8)
            cbar.ax.set_ylabel('Median % Mito', fontweight='bold')
        sns.despine(ax=ax)

        ax = axes[1, j]
        try:
            hexbin2 = ax.hexbin(x, y, gridsize=50, cmap='viridis',
                                mincnt=1, linewidths=0.2, edgecolors='face')
            ax.set_xlabel('Total UMI counts', fontweight='bold')
            ax.set_ylabel('Number of genes' if j == 0 else '', fontweight='bold')
            ax.axhline(MIN_GENES, color='#d62728', ls='--', alpha=0.8, lw=2)
            ax.axvline(MIN_COUNTS, color='#d62728', ls='--', alpha=0.8, lw=2)
            ax.set_xlim(0, np.percentile(adata.obs['total_counts'], 99))
            ax.set_ylim(0, np.percentile(adata.obs['n_genes_by_counts'], 99))
            if j == 3:
                cbar = plt.colorbar(hexbin2, ax=ax, label='Cell Count', shrink=0.8)
                cbar.ax.set_ylabel('Cell Count', fontweight='bold')
            sns.despine(ax=ax)
        except Exception as e:
            ax.scatter(x, y, s=0.5, alpha=0.3, c=SAMPLES[treatment]['color'], rasterized=True)
            ax.set_xlabel('Total UMI counts', fontweight='bold')
            ax.set_ylabel('Number of genes' if j == 0 else '', fontweight='bold')
            ax.axhline(MIN_GENES, color='#d62728', ls='--', alpha=0.8, lw=2)
            ax.axvline(MIN_COUNTS, color='#d62728', ls='--', alpha=0.8, lw=2)
            sns.despine(ax=ax)

    plt.suptitle('UMI vs Genes: Hexbin Density (top) & KDE Density (bottom)',
                 fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['qc'], 'fig02_qc_scatter_umi_vs_genes.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Spatial QC maps
    fig, axes = plt.subplots(2, 4, figsize=(34, 14))
    for j, treatment in enumerate(TREATMENT_ORDER):
        adata = adatas[treatment]
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial']
            n_plot = min(100000, adata.n_obs)
            idx = np.random.choice(adata.n_obs, n_plot, replace=False)

            ax = axes[0, j]
            sc1 = ax.scatter(coords[idx, 0], coords[idx, 1],
                             c=adata.obs['n_genes_by_counts'].values[idx],
                             cmap='plasma', s=0.5, alpha=0.6, rasterized=True, edgecolors='none')
            ax.set_title(f'{treatment}\nnGenes', fontsize=16, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            cbar = plt.colorbar(sc1, ax=ax, shrink=0.7, aspect=20)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label('Genes', fontweight='bold', fontsize=13)

            ax = axes[1, j]
            sc2 = ax.scatter(coords[idx, 0], coords[idx, 1],
                             c=adata.obs['pct_counts_mt'].values[idx],
                             cmap='YlOrRd', s=0.5, alpha=0.6, vmin=0, vmax=MAX_MITO_PCT,
                             rasterized=True, edgecolors='none')
            ax.set_title(f'%mito', fontsize=16, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            cbar = plt.colorbar(sc2, ax=ax, shrink=0.7, aspect=20)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label('% Mito', fontweight='bold', fontsize=13)
        else:
            axes[0, j].text(0.5, 0.5, 'No spatial\ncoordinates', ha='center',
                            va='center', fontsize=15, fontweight='bold')
            axes[1, j].text(0.5, 0.5, 'No spatial\ncoordinates', ha='center',
                            va='center', fontsize=15, fontweight='bold')

    plt.suptitle('Spatial QC Maps', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['qc'], 'fig03_spatial_qc_maps.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return adatas


# ============================================================
# STEP 3: FILTERING
# ============================================================
def filter_cells(adatas, dirs):
    """Apply QC filters. Early bin filtering reduces memory 50% before heavy operations."""
    os.makedirs(dirs['qc'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 3: FILTERING (REMOVING EMPTY BINS)")
    print("="*60)

    filter_stats = []
    for treatment, adata in adatas.items():
        n_before = adata.n_obs
        n_genes_before = adata.n_vars

        print(f"\n{treatment}: {n_before:,} bins, {n_genes_before:,} genes")

        sc.pp.filter_cells(adata, min_genes=MIN_GENES)
        adata = adata[adata.obs['n_genes_by_counts'] < MAX_GENES, :].copy()
        adata = adata[adata.obs['total_counts'] >= MIN_COUNTS, :].copy()
        adata = adata[adata.obs['pct_counts_mt'] < MAX_MITO_PCT, :].copy()

        if 'predicted_doublet' in adata.obs.columns:
            n_before_doublet = adata.n_obs
            adata = adata[~adata.obs['predicted_doublet']].copy()
            n_removed_doublets = n_before_doublet - adata.n_obs
            print(f"  Removed {n_removed_doublets:,} doublets ({100*n_removed_doublets/n_before_doublet:.1f}%)")

        sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)

        n_after = adata.n_obs
        n_genes_after = adata.n_vars
        pct_bins_kept = 100 * n_after / n_before
        pct_genes_kept = 100 * n_genes_after / n_genes_before

        print(f"  Bins: {n_before:,} → {n_after:,} ({pct_bins_kept:.1f}% kept)")
        print(f"  Genes: {n_genes_before:,} → {n_genes_after:,} ({pct_genes_kept:.1f}% kept)")

        filter_stats.append({
            'Treatment': treatment,
            'Bins_Before': n_before,
            'Bins_After': n_after,
            '% Bins_Kept': f'{pct_bins_kept:.1f}',
            'Genes_Before': n_genes_before,
            'Genes_After': n_genes_after,
            '% Genes_Kept': f'{pct_genes_kept:.1f}'
        })
        adatas[treatment] = adata

    pd.DataFrame(filter_stats).to_csv(os.path.join(dirs['qc'], 'filtering_stats.csv'), index=False)
    print(f"\nFiltering complete. Stats saved to filtering_stats.csv")

    # Figure 4: Violin+Box QC plots (after filtering)
    fig, axes = plt.subplots(3, 4, figsize=(28, 17))
    metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    labels = ['Genes per cell', 'UMI per cell', '% Mitochondrial']

    for j, treatment in enumerate(TREATMENT_ORDER):
        adata = adatas[treatment]
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[i, j]
            data = adata.obs[metric].values

            parts = ax.violinplot([data], positions=[0], showmedians=False,
                                  showextrema=False, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor(SAMPLES[treatment]['color'])
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
                pc.set_alpha(0.6)

            ax.boxplot([data], positions=[0], widths=0.15, patch_artist=True,
                       showfliers=False, showcaps=True,
                       boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2))

            mean_val = np.mean(data)
            ax.plot([0], [mean_val], marker='D', color='blue', markersize=8,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)

            median_val = np.median(data)
            ax.text(0.35, median_val, f'median: {median_val:.0f}', fontsize=11, va='center', ha='left')
            ax.text(0.35, mean_val, f'mean: {mean_val:.0f}', fontsize=11, va='center', ha='left', color='blue')

            ax.set_title(treatment if i == 0 else '', fontsize=15, fontweight='bold')
            ax.set_ylabel(label if j == 0 else '', fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
            sns.despine(ax=ax, left=False, bottom=True)

    plt.suptitle(f'QC Metrics After Filtering ({RESOLUTION_LABEL})', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['qc'], 'fig04_qc_violin_after_filter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return adatas


# ============================================================
# STEP 4: NORMALIZE, HVG, PCA, UMAP
# ============================================================
def normalize_and_reduce(adatas, dirs):
    """
    Memory-efficient per-sample normalization and dimensionality reduction.
    Subsets to 5000 HVGs before scaling (~4-8 GB/sample vs 35-60 GB naive).
    """
    os.makedirs(dirs['norm'], exist_ok=True)
    os.makedirs(dirs['cluster'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 4: NORMALIZATION & DIMENSIONALITY REDUCTION")
    print("="*60)

    for treatment, adata in adatas.items():
        print(f"\nProcessing {treatment} ({adata.n_obs:,} cells, {adata.n_vars:,} genes)...")
        import time
        step_start = time.time()

        print("  [1/7] Storing raw counts...")
        adata.layers['counts'] = adata.X.copy()

        print("  [2/7] Normalizing (total counts)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        print("  [3/7] Log-transforming...")
        sc.pp.log1p(adata)

        print("  [4/7] Storing raw copy...")
        adata.raw = adata.copy()

        print("  [5/7] Finding highly variable genes...")
        n_genes_before = adata.n_vars
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        n_hvg = adata.var['highly_variable'].sum()
        print(f"      → {n_hvg} highly variable genes selected")

        print(f"  [6/7] Subsetting to HVGs in-place...")
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"      → Genes: {n_genes_before:,} → {adata.n_vars:,} ({100*adata.n_vars/n_genes_before:.1f}% retained)")

        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.astype(np.float32)
        else:
            adata.X = adata.X.astype(np.float32)

        print("  [7/7] Scaling, PCA, neighbors, UMAP, clustering...")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=30, svd_solver='arpack')
        # n_pcs=30, fewer dims speeds up kNN on large datasets
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=15)
        sc.tl.umap(adata, min_dist=0.3, spread=1.0)

        print("      → Leiden clustering (auto resolution)...")
        chosen_res = 1.0
        # Resolution range extends to 6.0 to reliably reach 15+ clusters.
        for res in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
            sc.tl.leiden(adata, resolution=res, key_added='leiden')
            n_cl = adata.obs['leiden'].nunique()
            print(f"        resolution={res}: {n_cl} clusters")
            chosen_res = res
            if n_cl >= 15:
                break
        print(f"      → {adata.obs['leiden'].nunique()} Leiden clusters at resolution={chosen_res}")

        gc.collect()

        step_time = time.time() - step_start
        print(f"  ✓ {treatment} completed in {step_time/60:.1f} minutes")

        adatas[treatment] = adata

    # Figure 5: HVG scatter and top-ranked genes
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    example = adatas['Sham']
    hvg = example.var

    ax = axes[0]
    mean_col = 'mean_counts' if 'mean_counts' in hvg.columns else 'means'
    disp_col = 'dispersions_norm' if 'dispersions_norm' in hvg.columns else 'dispersions'

    non_hvg = hvg[~hvg['highly_variable']]
    ax.scatter(non_hvg[mean_col], non_hvg[disp_col],
               c='#d3d3d3', s=3, alpha=0.3, label='Non-HVG', rasterized=True)
    hvg_subset = hvg[hvg['highly_variable']]
    ax.scatter(hvg_subset[mean_col], hvg_subset[disp_col],
               c='#e31a1c', s=5, alpha=0.6, label='HVG', rasterized=True)
    ax.set_xlabel('Mean expression', fontweight='bold', fontsize=15)
    ax.set_ylabel('Normalized dispersion', fontweight='bold', fontsize=15)
    ax.set_title(f'Highly Variable Genes (Sham)\n{hvg_subset.shape[0]} HVGs selected',
                 fontweight='bold', fontsize=16)
    ax.legend(loc='upper right', frameon=True, fontsize=13)
    ax.set_xscale('log')
    sns.despine(ax=ax)

    ax = axes[1]
    top_hvgs = hvg_subset.nlargest(20, disp_col)
    y_pos = np.arange(len(top_hvgs))
    ax.barh(y_pos, top_hvgs[disp_col], color='#e31a1c', edgecolor='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_hvgs.index, fontsize=12)
    ax.set_xlabel('Normalized dispersion', fontweight='bold', fontsize=15)
    ax.set_title('Top 20 HVGs by Dispersion', fontweight='bold', fontsize=16)
    ax.invert_yaxis()
    sns.despine(ax=ax)

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['norm'], 'fig05_hvg_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 6: PCA elbow plot with cumulative variance
    fig, axes = plt.subplots(2, 4, figsize=(34, 14))
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[0, j]
        variance = adatas[treatment].uns['pca']['variance_ratio']
        ax.plot(range(1, len(variance)+1), variance, 'o-', markersize=4,
                color=SAMPLES[treatment]['color'], linewidth=2, markeredgecolor='black',
                markeredgewidth=0.5)
        ax.axvline(30, color='#d62728', ls='--', alpha=0.8, linewidth=2, label='n_pcs=30')
        ax.set_xlabel('Principal Component', fontweight='bold', fontsize=14)
        ax.set_ylabel('Variance Ratio' if j == 0 else '', fontweight='bold', fontsize=14)
        ax.set_title(treatment, fontweight='bold', fontsize=15)
        ax.legend(fontsize=12, frameon=True)
        ax.set_xlim(0, 51)
        sns.despine(ax=ax)

        ax = axes[1, j]
        cumsum = np.cumsum(variance)
        ax.plot(range(1, len(cumsum)+1), cumsum, 'o-', markersize=4,
                color=SAMPLES[treatment]['color'], linewidth=2, markeredgecolor='black',
                markeredgewidth=0.5)
        ax.axvline(30, color='#d62728', ls='--', alpha=0.8, linewidth=2)
        ax.axhline(cumsum[29], color='#2ca02c', ls=':', alpha=0.8, linewidth=2,
                   label=f'{cumsum[29]:.2%} var. explained')
        ax.set_xlabel('Principal Component', fontweight='bold', fontsize=14)
        ax.set_ylabel('Cumulative Variance' if j == 0 else '', fontweight='bold', fontsize=14)
        ax.legend(fontsize=12, frameon=True)
        ax.set_xlim(0, 51)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)

    plt.suptitle('PCA Analysis: Variance Explained', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['norm'], 'fig06_pca_elbow.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 7: UMAP per treatment
    fig, axes = plt.subplots(1, 4, figsize=(39, 8))
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[j]
        adata = adatas[treatment]
        clusters = adata.obs['leiden'].unique()
        n_clusters = len(clusters)
        colors = sns.color_palette("tab20", n_clusters)

        for i, cluster in enumerate(sorted(clusters, key=lambda x: int(x))):
            mask = adata.obs['leiden'] == cluster
            umap = adata.obsm['X_umap'][mask]
            ax.scatter(umap[:, 0], umap[:, 1], s=3, alpha=0.7,
                       c=[colors[i]], label=cluster, rasterized=True, edgecolors='none')

        ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
        ax.set_ylabel('UMAP 2' if j == 0 else '', fontweight='bold', fontsize=15)
        ax.set_title(treatment, fontsize=16, fontweight='bold')
        ax.legend(title='Leiden', fontsize=10, title_fontsize=11, loc='best',
                  ncol=2, frameon=True, markerscale=2)
        ax.set_aspect('equal')
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'UMAP — Leiden Clusters per Treatment ({RESOLUTION_LABEL})', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['cluster'], 'fig07_umap_leiden_per_treatment.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return adatas


# ============================================================
# STEP 5: MERGE & INTEGRATE
# ============================================================
def merge_and_integrate(adatas, dirs, args=None):
    """Merge all samples and run Harmony-integrated analysis."""
    os.makedirs(dirs['cluster'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 5: MERGE & INTEGRATE")
    print("="*60)
    import time
    merge_start = time.time()

    print("\n[1/7] Preparing samples for merge...")
    adata_list = []
    all_hvgs = set()

    for treatment in TREATMENT_ORDER:
        ad = adatas[treatment]
        print(f"  {treatment}: {ad.n_obs:,} cells, {ad.n_vars:,} genes")
        if 'highly_variable' in ad.var.columns:
            hvg_genes = ad.var_names[ad.var['highly_variable']].tolist()
            all_hvgs.update(hvg_genes)
            print(f"    → {len(hvg_genes)} HVGs")

    print(f"\n  Total union HVGs: {len(all_hvgs)}")

    print("\n[2/7] Extracting normalized data...")
    for treatment in TREATMENT_ORDER:
        ad = adatas[treatment]

        if ad.raw is not None:
            ad_norm = ad.raw.to_adata()
        else:
            ad_norm = ad.copy()

        if not scipy.sparse.issparse(ad_norm.X):
            ad_norm.X = scipy.sparse.csr_matrix(ad_norm.X, dtype=np.float32)
        else:
            ad_norm.X = ad_norm.X.astype(np.float32)

        ad_norm.obs = ad.obs[['treatment', 'sample_id']].copy()
        if 'spatial' in ad.obsm:
            ad_norm.obsm['spatial'] = ad.obsm['spatial'].copy()

        print(f"  {treatment}: {ad_norm.n_obs:,} cells × {ad_norm.n_vars:,} genes")
        adata_list.append(ad_norm)

    print("\n[3/7] Concatenating samples...")
    adata_merged = sc.concat(
        adata_list,
        label='treatment',
        keys=TREATMENT_ORDER,
        join='inner',
        merge='same'
    )
    print(f"  → Merged: {adata_merged.n_obs:,} cells × {adata_merged.n_vars:,} genes")

    del adata_list

    print("\n[4/7] Storing full gene set as .raw...")
    adata_merged.raw = adata_merged.copy()
    print(f"  → Stored raw: {adata_merged.raw.n_vars:,} genes")

    print("\n[5/7] Finding HVGs on merged data...")
    # 4000 genes captures rare cell-type markers across all 4 treatments.
    sc.pp.highly_variable_genes(adata_merged, n_top_genes=4000,
                                 batch_key='treatment')
    n_merged_hvg = adata_merged.var['highly_variable'].sum()
    print(f"  → {n_merged_hvg} HVGs selected for integration")

    print("\n[6/7] Subsetting merged data to HVGs...")
    n_genes_before = adata_merged.n_vars
    adata_merged = adata_merged[:, adata_merged.var['highly_variable']].copy()
    print(f"  → Genes: {n_genes_before:,} → {adata_merged.n_vars:,}")

    if scipy.sparse.issparse(adata_merged.X):
        adata_merged.X = adata_merged.X.astype(np.float32)

    sc.pp.scale(adata_merged, max_value=10)
    sc.tl.pca(adata_merged, n_comps=50, svd_solver='arpack')
    gc.collect()

    print("\n[7/7] Batch correction with Harmony...")
    try:
        from harmonypy import run_harmony

        pca_matrix = adata_merged.obsm['X_pca']
        n_cells, n_pcs = pca_matrix.shape
        print(f"  Input PCA shape: {pca_matrix.shape}")

        ho = run_harmony(pca_matrix, adata_merged.obs, 'treatment', max_iter_harmony=20)
        z_corr = ho.Z_corr

        if hasattr(z_corr, 'numpy'):
            z_corr = z_corr.detach().cpu().numpy()

        print(f"  Z_corr raw shape: {z_corr.shape}")

        if z_corr.shape == (n_pcs, n_cells):
            z_corr = z_corr.T
        elif z_corr.shape == (n_cells, n_pcs):
            pass
        else:
            raise ValueError(f"Unexpected Z_corr shape: {z_corr.shape}, expected ({n_cells}, {n_pcs}) or ({n_pcs}, {n_cells})")

        adata_merged.obsm['X_pca_harmony'] = z_corr
        use_rep = 'X_pca_harmony'
        print(f"  ✓ Harmony corrected: {adata_merged.obsm['X_pca_harmony'].shape}")

        print("\n  [Integration QC] Computing LISI scores...")
        try:
            from harmonypy import compute_lisi
            lisi_scores = compute_lisi(
                adata_merged.obsm['X_pca_harmony'],
                adata_merged.obs[['treatment']],
                ['treatment']
            )
            adata_merged.obs['LISI_treatment'] = lisi_scores[:, 0]
            mean_lisi = adata_merged.obs['LISI_treatment'].mean()
            print(f"  Mean LISI score (treatment): {mean_lisi:.3f} (max possible = {len(adata_merged.obs['treatment'].unique())})")
            if mean_lisi > 1.5:
                print(f"  EXCELLENT: Good batch correction (LISI = {mean_lisi:.2f})")
            elif mean_lisi > 1.2:
                print(f"  GOOD: Adequate batch correction (LISI = {mean_lisi:.2f})")
            else:
                print(f"  WARNING: Batch correction may be insufficient (LISI = {mean_lisi:.2f})")
        except Exception as e:
            print(f"  WARNING: LISI computation failed: {e}")
            adata_merged.obs['LISI_treatment'] = 1.0

    except ImportError:
        print("  Harmony not installed. Using uncorrected PCA.")
        use_rep = 'X_pca'
    except Exception as e:
        print(f"  Harmony failed: {e}. Falling back to uncorrected PCA.")
        use_rep = 'X_pca'

    print("\n[8/7] Computing neighbors, UMAP, and clustering...")
    _ckpt_dir = dirs['checkpoints']
    _nbr_ckpt  = os.path.join(_ckpt_dir, 'merged_neighbors.npz')
    _umap_ckpt = os.path.join(_ckpt_dir, 'merged_umap.npy')
    _leid_ckpt = os.path.join(_ckpt_dir, 'merged_leiden.csv')

    # ── Neighbors ────────────────────────────────────────────────────────────
    if (args is not None and args.resume) and os.path.exists(_nbr_ckpt):
        print("  ✓ Checkpoint found, loading neighbors graph...")
        import scipy.sparse as sp_sparse
        loaded = np.load(_nbr_ckpt, allow_pickle=True)
        adata_merged.obsp['connectivities'] = sp_sparse.csr_matrix(
            (loaded['data'], loaded['indices'], loaded['indptr']),
            shape=(adata_merged.n_obs, adata_merged.n_obs))
        adata_merged.obsp['distances'] = sp_sparse.csr_matrix(
            (loaded['dist_data'], loaded['dist_indices'], loaded['dist_indptr']),
            shape=(adata_merged.n_obs, adata_merged.n_obs))
        adata_merged.uns['neighbors'] = {'connectivities_key': 'connectivities',
                                          'distances_key': 'distances',
                                          'params': {'n_neighbors': 15, 'method': 'umap'}}
    else:
        t0 = time.time()
        # n_pcs=30, fewer dims speeds up kNN on 1.2M cells dramatically
        sc.pp.neighbors(adata_merged, n_pcs=30, n_neighbors=15, use_rep=use_rep)
        print(f"  Neighbors done in {time.time()-t0:.0f}s, saving checkpoint...")
        conn = adata_merged.obsp['connectivities'].tocsr()
        dist = adata_merged.obsp['distances'].tocsr()
        np.savez(_nbr_ckpt,
                 data=conn.data, indices=conn.indices, indptr=conn.indptr,
                 dist_data=dist.data, dist_indices=dist.indices, dist_indptr=dist.indptr)
        print(f"  Checkpoint saved: {_nbr_ckpt}")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    if (args is not None and args.resume) and os.path.exists(_umap_ckpt):
        print("  ✓ Checkpoint found, loading UMAP coordinates...")
        adata_merged.obsm['X_umap'] = np.load(_umap_ckpt)
    else:
        t0 = time.time()
        sc.tl.umap(adata_merged, min_dist=0.3, spread=1.0)
        print(f"  UMAP done in {time.time()-t0:.0f}s, saving checkpoint...")
        np.save(_umap_ckpt, adata_merged.obsm['X_umap'])
        print(f"  Checkpoint saved: {_umap_ckpt}")

    # ── Leiden ────────────────────────────────────────────────────────────────
    if (args is not None and args.resume) and os.path.exists(_leid_ckpt):
        print("  ✓ Checkpoint found, loading Leiden labels...")
        import pandas as pd
        _ldf = pd.read_csv(_leid_ckpt, index_col=0)
        adata_merged.obs['leiden_merged'] = pd.Categorical(_ldf['leiden_merged'].values)
        chosen_res = float(_ldf.attrs.get('resolution', 1.0))
    else:
        # Resolution search extends to 8.0, targeting 25+ clusters.
        print("  → Leiden clustering (auto resolution targeting 25+ clusters)...")
        chosen_res = 2.0
        for res in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
            sc.tl.leiden(adata_merged, resolution=res, key_added='leiden_merged')
            n_cl = adata_merged.obs['leiden_merged'].nunique()
            print(f"    resolution={res}: {n_cl} clusters")
            chosen_res = res
            if n_cl >= 25:
                break
        import pandas as pd
        _ldf = adata_merged.obs[['leiden_merged']].copy()
        _ldf.to_csv(_leid_ckpt)
        print(f"  Leiden checkpoint saved: {_leid_ckpt}")

    n_clusters = adata_merged.obs['leiden_merged'].nunique()
    print(f"  → Final: {n_clusters} clusters at resolution={chosen_res}")

    cluster_sizes = adata_merged.obs['leiden_merged'].value_counts()
    top5_pct = cluster_sizes.head(5).sum() / len(adata_merged) * 100
    if top5_pct > 70:
        print(f"  WARNING: Top 5 clusters contain {top5_pct:.1f}% of cells, consider higher resolution.")

    merge_time = time.time() - merge_start
    print(f"\nMerge complete: {n_clusters} clusters in {merge_time/60:.1f} minutes")

    # Figure 8: Integrated UMAP
    fig, axes = plt.subplots(1, 3, figsize=(38, 10))

    ax = axes[0]
    for treatment in TREATMENT_ORDER:
        mask = adata_merged.obs['treatment'] == treatment
        umap = adata_merged.obsm['X_umap'][mask]
        ax.scatter(umap[:, 0], umap[:, 1], s=3, alpha=0.6,
                   c=[SAMPLES[treatment]['color']], label=treatment,
                   rasterized=True, edgecolors='none')
    ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
    ax.set_ylabel('UMAP 2', fontweight='bold', fontsize=15)
    ax.set_title('Colored by Treatment', fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, frameon=True, markerscale=3)
    ax.set_aspect('equal')
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    clusters = adata_merged.obs['leiden_merged'].unique()
    n_clusters = len(clusters)
    colors = sns.color_palette("tab20", n_clusters)
    for i, cluster in enumerate(sorted(clusters, key=lambda x: int(x))):
        mask = adata_merged.obs['leiden_merged'] == cluster
        umap = adata_merged.obsm['X_umap'][mask]
        ax.scatter(umap[:, 0], umap[:, 1], s=3, alpha=0.7,
                   c=[colors[i]], label=cluster, rasterized=True, edgecolors='none')
    ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
    ax.set_ylabel('UMAP 2', fontweight='bold', fontsize=15)
    ax.set_title('Leiden Clusters (Merged)', fontsize=16, fontweight='bold')
    ax.legend(title='Cluster', fontsize=10, title_fontsize=12, loc='best',
              ncol=2, frameon=True, markerscale=2)
    ax.set_aspect('equal')
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    try:
        umap_coords = adata_merged.obsm['X_umap']
        n_plot = min(200000, len(umap_coords))
        idx_plot = np.random.choice(len(umap_coords), n_plot, replace=False)
        hexbin = ax.hexbin(umap_coords[idx_plot, 0], umap_coords[idx_plot, 1],
                           gridsize=80, cmap='viridis', mincnt=1,
                           linewidths=0.1, edgecolors='face')
        cbar = plt.colorbar(hexbin, ax=ax, shrink=0.8)
        cbar.set_label('Cell Count', fontweight='bold', fontsize=13)
    except Exception as e:
        print(f"  Warning: Density plot failed: {e}")
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=1, alpha=0.3, c='grey', rasterized=True)

    ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
    ax.set_ylabel('UMAP 2', fontweight='bold', fontsize=15)
    ax.set_title('Cell Density', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.suptitle('Integrated UMAP (All 4 Treatments)', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['cluster'], 'fig08_integrated_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 8b: Integration Quality (LISI on UMAP)
    fig, axes = plt.subplots(1, 3, figsize=(42, 11))

    ax = axes[0]
    for treatment in TREATMENT_ORDER:
        mask = adata_merged.obs['treatment'] == treatment
        pca_coords = adata_merged.obsm['X_pca'][mask]
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], s=1, alpha=0.5,
                   c=[SAMPLES[treatment]['color']], label=treatment,
                   rasterized=True, edgecolors='none')
    ax.set_xlabel('PC 1', fontweight='bold', fontsize=15)
    ax.set_ylabel('PC 2', fontweight='bold', fontsize=15)
    ax.set_title('Before Integration (PCA)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, markerscale=3)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    for treatment in TREATMENT_ORDER:
        mask = adata_merged.obs['treatment'] == treatment
        umap_coords = adata_merged.obsm['X_umap'][mask]
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=1, alpha=0.5,
                   c=[SAMPLES[treatment]['color']], label=treatment,
                   rasterized=True, edgecolors='none')
    ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
    ax.set_ylabel('UMAP 2', fontweight='bold', fontsize=15)
    ax.set_title('After Harmony Integration', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, markerscale=3)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[2]
    if 'LISI_treatment' in adata_merged.obs.columns:
        mean_lisi_val = adata_merged.obs['LISI_treatment'].mean()
        n_treatments = len(adata_merged.obs['treatment'].unique())
        umap_all = adata_merged.obsm['X_umap']
        lisi_vals = adata_merged.obs['LISI_treatment'].values
        sort_idx = lisi_vals.argsort()
        sc_lisi = ax.scatter(umap_all[sort_idx, 0], umap_all[sort_idx, 1],
                             c=lisi_vals[sort_idx], s=1, alpha=0.6,
                             cmap='RdYlGn', vmin=1, vmax=n_treatments,
                             rasterized=True, edgecolors='none')
        cbar = plt.colorbar(sc_lisi, ax=ax, shrink=0.8)
        cbar.set_label('LISI Score', fontweight='bold', fontsize=13)
        ax.set_title(f'Integration Quality (LISI, mean={mean_lisi_val:.2f})',
                     fontsize=16, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'LISI scores\nnot available', ha='center', va='center',
                fontsize=15, fontweight='bold', transform=ax.transAxes)
        ax.set_title('Integration Quality (LISI)', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
    ax.set_ylabel('UMAP 2', fontweight='bold', fontsize=15)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.suptitle('Harmony Integration Quality Assessment', fontsize=24, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['cluster'], 'fig08b_integration_quality_lisi.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 9: UMAP split by treatment
    fig, axes = plt.subplots(1, 4, figsize=(39, 8))
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[j]
        mask = adata_merged.obs['treatment'] == treatment
        ax.scatter(adata_merged.obsm['X_umap'][~mask, 0],
                   adata_merged.obsm['X_umap'][~mask, 1],
                   c='#e0e0e0', s=1, alpha=0.2, rasterized=True, edgecolors='none')
        ax.scatter(adata_merged.obsm['X_umap'][mask, 0],
                   adata_merged.obsm['X_umap'][mask, 1],
                   c=SAMPLES[treatment]['color'], s=3, alpha=0.7,
                   rasterized=True, edgecolors='none')
        ax.set_xlabel('UMAP 1', fontweight='bold', fontsize=15)
        ax.set_ylabel('UMAP 2' if j == 0 else '', fontweight='bold', fontsize=15)
        ax.set_title(treatment, fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('UMAP Split by Treatment', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['cluster'], 'fig09_umap_split_treatment.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return adata_merged


# ============================================================
# STEP 6: MARKER GENES & ANNOTATION
# ============================================================
def find_markers(adata_merged, dirs):
    """Find marker genes per cluster for annotation."""
    os.makedirs(dirs['cluster'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 6: MARKER GENES")
    print("="*60)

    sc.tl.rank_genes_groups(adata_merged, groupby='leiden_merged', method='wilcoxon')

    # Figure 10: Dotplot and heatmap of top markers
    fig, ax = plt.subplots(1, 1, figsize=(28, 14))
    sc.pl.rank_genes_groups_dotplot(adata_merged, n_genes=5, ax=ax, show=False,
                                    dendrogram=False, cmap='Reds',
                                    dot_max=0.7, dot_min=0.1,
                                    standard_scale='var')
    ax.set_title('Top 5 Marker Genes per Cluster', fontweight='bold', fontsize=17)
    plt.suptitle('Marker Gene Analysis', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['cluster'], 'fig10_marker_dotplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    try:
        fig = plt.figure(figsize=(22, 17))
        sc.pl.rank_genes_groups_heatmap(adata_merged, n_genes=5, show=False,
                                        cmap='RdBu_r', vmin=-2, vmax=2,
                                        standard_scale='var', swap_axes=False,
                                        show_gene_labels=True)
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['cluster'], 'fig10_marker_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Generated both dotplot and heatmap")
    except Exception as e:
        print(f"  Heatmap plotting skipped: {e}")

    markers_df = sc.get.rank_genes_groups_df(adata_merged, group=None)
    markers_df.to_csv(os.path.join(dirs['data'], 'marker_genes_per_cluster.csv'), index=False)
    print(f"Saved marker genes to marker_genes_per_cluster.csv")

    # Figure 11: PDAC marker genes on UMAP
    pdac_markers = {
        'Tumor (ductal)': ['Krt19', 'Krt8', 'Epcam', 'Cdh1'],
        'CAFs': ['Col1a1', 'Col3a1', 'Dcn', 'Fn1', 'Acta2'],
        'Macrophage': ['Adgre1', 'Cd68', 'Mrc1', 'Itgam'],
        'T cells': ['Cd3e', 'Cd4', 'Cd8a', 'Foxp3'],
        'B cells': ['Cd79a', 'Ms4a1', 'Cd19'],
        'Endothelial': ['Pecam1', 'Cdh5', 'Vwf'],
        'NK cells': ['Ncr1', 'Klrb1c', 'Nkg7'],
    }

    all_markers = [g for group in pdac_markers.values() for g in group]
    available = [g for g in all_markers if g in adata_merged.var_names]

    if available:
        n_markers = len(available)
        n_cols = 5
        n_rows = int(np.ceil(n_markers / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i, gene in enumerate(available):
            if i < len(axes_flat):
                ax = axes_flat[i]
                gene_expr = adata_merged[:, gene].X.toarray().flatten() if scipy.sparse.issparse(adata_merged.X) else adata_merged[:, gene].X.flatten()

                idx_sort = gene_expr.argsort()
                umap_sorted = adata_merged.obsm['X_umap'][idx_sort]
                expr_sorted = gene_expr[idx_sort]

                scatter = ax.scatter(umap_sorted[:, 0], umap_sorted[:, 1],
                                     c=expr_sorted, s=2, alpha=0.7,
                                     cmap='magma', rasterized=True, edgecolors='none',
                                     vmin=0, vmax=np.percentile(gene_expr, 99))
                ax.set_title(gene, fontweight='bold', fontsize=17)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                sns.despine(ax=ax, left=True, bottom=True)
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('Expression', fontweight='bold', fontsize=12)

        for i in range(len(available), len(axes_flat)):
            axes_flat[i].axis('off')

        plt.suptitle('PDAC Marker Genes on UMAP', fontsize=24, fontweight='bold')
        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['cluster'], 'fig11_pdac_markers_umap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plotted {len(available)} marker genes")

    return adata_merged


# ============================================================
# STEP 7: SPATIAL VISUALIZATION
# ============================================================
def spatial_plots(adatas, dirs):
    """Generate spatial cell type maps per treatment."""
    os.makedirs(dirs['spatial'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 7: SPATIAL PLOTS")
    print("="*60)

    # Figure 12: Spatial Leiden cluster maps
    fig, axes = plt.subplots(1, 4, figsize=(39, 10))
    for j, treatment in enumerate(TREATMENT_ORDER):
        ax = axes[j]
        adata = adatas[treatment]
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial']
            clusters = adata.obs['leiden'].astype(int)
            n_plot = min(200000, adata.n_obs)
            idx = np.random.choice(adata.n_obs, n_plot, replace=False)
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=clusters.values[idx], cmap='tab20',
                       s=0.8, alpha=0.7, rasterized=True, edgecolors='none')
            ax.set_title(treatment, fontsize=17, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No spatial coords', ha='center', va='center',
                    fontsize=15, fontweight='bold')

    plt.suptitle(f'Spatial Leiden Clusters ({RESOLUTION_LABEL})', fontsize=22, fontweight='bold')
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['spatial'], 'fig12_spatial_leiden_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 13: Spatial expression of key markers per treatment
    key_markers = ['Krt19', 'Col1a1', 'Cd3e', 'Adgre1', 'Pecam1', 'Mrc1']

    for treatment in TREATMENT_ORDER:
        adata = adatas[treatment]
        if 'spatial' not in adata.obsm:
            continue

        available_markers = [g for g in key_markers if g in adata.var_names]
        if not available_markers:
            continue

        n_markers = len(available_markers)
        n_cols = max(1, int(np.ceil(n_markers/2)))
        fig, axes = plt.subplots(2, n_cols, figsize=(7*n_cols, 14))

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        coords = adata.obsm['spatial']
        n_plot = min(200000, adata.n_obs)
        idx = np.random.choice(adata.n_obs, n_plot, replace=False)

        expr_data = adata.raw.X if adata.raw is not None else adata.X

        for i, gene in enumerate(available_markers):
            ax = axes_flat[i]
            gene_idx = list(adata.raw.var_names if adata.raw else adata.var_names).index(gene)

            if hasattr(expr_data, 'toarray'):
                expr = np.array(expr_data[idx, gene_idx].toarray()).flatten()
            else:
                expr = np.array(expr_data[idx, gene_idx]).flatten()

            sort_idx = expr.argsort()
            coords_sorted = coords[idx][sort_idx]
            expr_sorted = expr[sort_idx]

            sc_plot = ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                                 c=expr_sorted, cmap='plasma', s=0.8, alpha=0.7,
                                 vmin=0, vmax=np.percentile(expr[expr > 0], 98) if (expr > 0).any() else 1,
                                 rasterized=True, edgecolors='none')
            ax.set_title(gene, fontsize=16, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            cbar = plt.colorbar(sc_plot, ax=ax, shrink=0.7, aspect=20)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label('Expression', fontweight='bold', fontsize=13)

        for i in range(len(available_markers), len(axes_flat)):
            axes_flat[i].axis('off')

        plt.suptitle(f'Spatial Gene Expression — {treatment}', fontsize=22, fontweight='bold')
        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['spatial'], f'fig13_spatial_markers_{treatment}.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()

    print("Spatial plots saved.")


# ============================================================
# STEP 8: CELL TYPE COMPOSITION COMPARISON
# ============================================================
def composition_analysis(adata_merged, dirs):
    """Compare cluster composition across treatments."""
    os.makedirs(dirs['composition'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 8: COMPOSITION ANALYSIS")
    print("="*60)

    prop = pd.crosstab(adata_merged.obs['treatment'], adata_merged.obs['leiden_merged'], normalize='index')
    prop = prop.reindex(TREATMENT_ORDER)

    # Figure 14: Stacked and grouped bar charts
    fig, axes = plt.subplots(1, 2, figsize=(28, 8))

    ax = axes[0]
    colors = sns.color_palette("tab20", len(prop.columns))
    prop.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.75, edgecolor='black', linewidth=1)
    ax.set_ylabel('Proportion', fontweight='bold', fontsize=15)
    ax.set_xlabel('Treatment', fontweight='bold', fontsize=15)
    ax.set_title('Cluster Composition per Treatment (Stacked)', fontweight='bold', fontsize=17)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title='Cluster',
              title_fontsize=12, frameon=True)
    ax.set_xticklabels(TREATMENT_ORDER, rotation=0, fontweight='bold')
    ax.set_ylim(0, 1)
    sns.despine(ax=ax)

    ax = axes[1]
    prop.plot(kind='bar', stacked=False, ax=ax, color=colors, width=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Proportion', fontweight='bold', fontsize=15)
    ax.set_xlabel('Treatment', fontweight='bold', fontsize=15)
    ax.set_title('Cluster Composition per Treatment (Grouped)', fontweight='bold', fontsize=17)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title='Cluster',
              title_fontsize=11, frameon=True, ncol=2)
    ax.set_xticklabels(TREATMENT_ORDER, rotation=0, fontweight='bold')
    sns.despine(ax=ax)

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['composition'], 'fig14_cluster_proportion_stacked_bar.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 15: Proportion heatmap with hierarchical clustering
    fig, axes = plt.subplots(1, 2, figsize=(28, 11))

    ax = axes[0]
    sns.heatmap(prop.T, cmap='YlOrRd', ax=ax, annot=True, fmt='.2f',
                linewidths=1, linecolor='white', cbar_kws={'label': 'Proportion'},
                vmin=0, vmax=prop.values.max(), square=False)
    ax.set_title('Cluster Proportion Heatmap', fontweight='bold', fontsize=17)
    ax.set_ylabel('Cluster', fontweight='bold', fontsize=15)
    ax.set_xlabel('Treatment', fontweight='bold', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight='bold')

    ax = axes[1]
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist

    try:
        row_linkage = linkage(pdist(prop.T, metric='euclidean'), method='ward')
        dendrogram(row_linkage, ax=ax, orientation='left', no_labels=True)
        ax.set_title('Hierarchical Clustering\n(Cluster Similarity)', fontweight='bold', fontsize=17)
        ax.set_xlabel('Distance', fontweight='bold', fontsize=15)
        ax.set_ylabel('Cluster', fontweight='bold', fontsize=15)
        sns.despine(ax=ax)
    except Exception:
        ax.text(0.5, 0.5, 'Clustering failed', ha='center', va='center',
                fontsize=15, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['composition'], 'fig15_cluster_proportion_heatmap.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()

    prop.to_csv(os.path.join(dirs['data'], 'cluster_proportions_per_treatment.csv'))
    print("Composition analysis complete.")
    return adata_merged


# ============================================================
# STEP 9: DIFFERENTIAL EXPRESSION
# ============================================================
def differential_expression(adata_merged, dirs):
    """Treatment-wise DEG analysis (GPH+IT vs Sham)."""
    os.makedirs(dirs['deg'], exist_ok=True)
    print("\n" + "="*60)
    print("STEP 9: DIFFERENTIAL EXPRESSION")
    print("="*60)

    mask = adata_merged.obs['treatment'].isin(['Sham', 'GPH+IT'])
    adata_sub = adata_merged[mask].copy()

    sc.tl.rank_genes_groups(adata_sub, groupby='treatment', groups=['GPH+IT'],
                            reference='Sham', method='wilcoxon')

    result = sc.get.rank_genes_groups_df(adata_sub, group='GPH+IT')
    result['neg_log10_pval'] = -np.log10(result['pvals_adj'].clip(lower=1e-300))
    result['significant'] = (result['pvals_adj'] < 0.05) & (abs(result['logfoldchanges']) > 0.5)

    # Volcano and top-DEG bar chart are saved as separate figures (panels H
    # and I): a shared canvas forces one scale on both, which caps how many
    # genes the bar chart can show legibly.
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)

    ns = result[~result['significant']]
    ax.scatter(ns['logfoldchanges'], ns['neg_log10_pval'], c='#d3d3d3', s=5, alpha=0.3,
               label='Non-significant', rasterized=True, edgecolors='none')

    sig_up = result[(result['significant']) & (result['logfoldchanges'] > 0)]
    ax.scatter(sig_up['logfoldchanges'], sig_up['neg_log10_pval'], c='#d62728', s=10, alpha=0.6,
               label=f'Up ({len(sig_up)})', rasterized=True, edgecolors='none')

    sig_down = result[(result['significant']) & (result['logfoldchanges'] < 0)]
    ax.scatter(sig_down['logfoldchanges'], sig_down['neg_log10_pval'], c='#1f77b4', s=10, alpha=0.6,
               label=f'Down ({len(sig_down)})', rasterized=True, edgecolors='none')

    top_up = sig_up.nlargest(N_VOLCANO_LABEL, 'neg_log10_pval')
    top_down = sig_down.nlargest(N_VOLCANO_LABEL, 'neg_log10_pval')

    ymax_data = result['neg_log10_pval'].max()
    # Only a little headroom: labels are anchored beside their own points now,
    # so a tall reserved band is not needed -- and reserving one is what forced
    # the long diagonal leaders in the first place.
    ax.set_ylim(top=ymax_data * 1.55)
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo - (xhi - xlo) * 0.09, xhi + (xhi - xlo) * 0.09)

    # Deterministic label placement rather than adjustText.
    #
    # Every gene worth labelling sits at the p-value floor, where -log10(padj)
    # saturates, so the top hits form a near-horizontal line of points with
    # nearly identical y. A force-directed solver has no vertical gradient to
    # work with there and leaves them stacked however the forces are tuned.
    #
    # Strategy: seed each label immediately beside its own point, then push
    # apart only where boxes actually collide. Isolated genes therefore keep a
    # leader line of almost zero length, and only the saturated cluster grows a
    # short stack above itself.
    fig.canvas.draw()
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    ext = ax.get_window_extent()
    x_per_px = (x1 - x0) / ext.width
    y_per_px = (y1 - y0) / ext.height
    FS = 26
    lh = FS * fig.dpi / 72.0 * y_per_px            # text height, data units

    def _tw(t):
        return len(str(t)) * FS * 0.58 * fig.dpi / 72.0 * x_per_px

    def _place(df, side):
        """side: +1 label to the right of its point, -1 to the left."""
        if df.empty:
            return
        d = df.sort_values('neg_log10_pval', ascending=False)
        placed = []                                   # (xa, xb, ya, yb)
        pad_x = (x1 - x0) * 0.012
        for _, row in d.iterrows():
            px, py = row['logfoldchanges'], row['neg_log10_pval']
            w, h = _tw(row['names']), lh
            lx = px + side * pad_x
            ly = py                                    # start level with point
            xa = lx if side > 0 else lx - w
            for _ in range(400):
                ya, yb = ly - h * 0.5, ly + h * 0.5
                hit = next((b for b in placed
                            if not (xa + w < b[0] or xa > b[1]
                                    or yb < b[2] or ya > b[3])), None)
                if hit is None:
                    break
                ly = hit[3] + h * 0.62                 # lift just clear of it
            # If the stack ran out of vertical room, step the label outward in
            # x and restart the descent rather than clamping it back down --
            # clamping silently dropped labels onto each other (Onecut2/Chd9).
            if ly > y1 - h * 0.6:
                shift = 0
                while ly > y1 - h * 0.6 and shift < 12:
                    shift += 1
                    lx = px + side * pad_x * (1 + shift * 1.6)
                    xa = lx if side > 0 else lx - w
                    ly = py
                    for _ in range(400):
                        ya, yb = ly - h * 0.5, ly + h * 0.5
                        hit = next((b for b in placed
                                    if not (xa + w < b[0] or xa > b[1]
                                            or yb < b[2] or ya > b[3])), None)
                        if hit is None:
                            break
                        ly = hit[3] + h * 0.62
                ly = min(ly, y1 - h * 0.6)
            placed.append((xa, xa + w, ly - h * 0.5, ly + h * 0.5))
            ann = ax.annotate(
                str(row['names']), xy=(px, py), xytext=(lx, ly),
                fontsize=FS, zorder=12,
                ha='left' if side > 0 else 'right', va='center',
                arrowprops=dict(arrowstyle='-', color='#666666', lw=0.7,
                                alpha=0.55, shrinkA=1, shrinkB=3))
            # Boxes are kept apart above, but a leader line travelling up to the
            # saturated cluster can still pass through a neighbour's glyphs. A
            # white halo lets the text read cleanly wherever a line crosses it.
            ann.set_path_effects([pe.withStroke(linewidth=3.0, foreground='white')])

    _place(top_up, +1)
    _place(top_down, -1)

    ax.axhline(-np.log10(0.05), color='#666666', ls='--', alpha=0.7, lw=2, label='p-adj < 0.05')
    ax.axvline(0.5, color='#666666', ls='--', alpha=0.7, lw=2)
    ax.axvline(-0.5, color='#666666', ls='--', alpha=0.7, lw=2, label='|FC| > 0.5')
    ax.set_xlabel('Log2 Fold Change', fontsize=32)
    ax.set_ylabel('-log10(adjusted p-value)', fontsize=32)
    ax.set_title('DEGs: GPH+IT vs Sham', fontsize=36)
    # Inside the axes, upper-left: up-regulated labels sit upper-right and
    # down-regulated ones lower-left, so upper-left stays clear.
    # markerscale + framealpha=1: the scatter dots are tiny (s=5/10) and
    # faded (alpha=0.3/0.6) in the plot itself so they read at a glance
    # among thousands of points -- blown up in the legend at full opacity,
    # the red/blue otherwise still looked washed out and hard to tell apart.
    leg = ax.legend(loc='upper left', ncol=1, frameon=True, fontsize=26,
                    markerscale=5, framealpha=1)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.tick_params(axis='both', which='major', labelsize=28)
    sns.despine(ax=ax)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(dirs['deg'], f'fig16_volcano_GPH_IT_vs_Sham.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()

    # ── Panel I: top DEG bar chart, its own figure ───────────────────────────
    # Split by direction across two columns rather than stacking all 2*N genes
    # in one axes. A single column of 30 genes is taller than it is wide, and a
    # panel that shape makes the whole composite height-bound, which shrinks
    # every other panel. Two columns keep all 30 genes (the data is the data)
    # while giving the panel a usable aspect -- and separating up- from
    # down-regulated is the clearer presentation anyway.
    up_bar = sig_up.nlargest(N_DEG_BAR, 'neg_log10_pval').sort_values('logfoldchanges')
    dn_bar = sig_down.nlargest(N_DEG_BAR, 'neg_log10_pval').sort_values('logfoldchanges')
    n_total = len(up_bar) + len(dn_bar)
    n_rows = max(len(up_bar), len(dn_bar))

    # Row pitch must exceed one line of label text or the gene names collide.
    # At 24pt labels a line is ~30pt, so ~40pt of pitch per gene is safe.
    bar_h = max(7.5, n_rows * 40 / 72)
    fig, bax = plt.subplots(1, 2, figsize=(16, bar_h))

    for axb, sub, col, sub_title in (
            (bax[0], dn_bar, '#1f77b4', f'Down in GPH+IT (top {len(dn_bar)})'),
            (bax[1], up_bar, '#d62728', f'Up in GPH+IT (top {len(up_bar)})')):
        y_pos = np.arange(len(sub))
        axb.barh(y_pos, sub['logfoldchanges'], color=col,
                 edgecolor='black', linewidth=1)
        axb.set_yticks(y_pos)
        axb.set_yticklabels(sub['names'], fontsize=24)
        axb.set_ylim(-0.8, n_rows - 0.2)
        axb.set_xlabel('Log2 Fold Change', fontsize=28)
        axb.set_title(sub_title, fontsize=28)
        axb.tick_params(axis='x', which='major', labelsize=24)
        axb.axvline(0, color='black', lw=1.5)
        sns.despine(ax=axb)

    fig.suptitle(f'Top {n_total} DEGs by Significance', fontsize=34, y=1.005)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(dirs['deg'], f'fig16b_top_degs_bar.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()

    result.to_csv(os.path.join(dirs['data'], 'DEGs_GPH_IT_vs_Sham.csv'), index=False)
    print(f"DEGs: {len(sig_up)} up, {len(sig_down)} down (GPH+IT vs Sham)")

    return adata_merged


# ============================================================
# STEP 10: SAVE PROCESSED DATA
# ============================================================
def save_data(adatas, adata_merged, dirs):
    """Save processed h5ad files."""
    print("\n" + "="*60)
    print("STEP 10: SAVING DATA")
    print("="*60)

    for treatment, adata in adatas.items():
        path = os.path.join(dirs['data'], f'{treatment}_cellbin_processed.h5ad')
        adata.write(path)
        print(f"Saved {treatment}: {path}")

    path = os.path.join(dirs['data'], 'merged_all_treatments.h5ad')
    adata_merged.write(path)
    print(f"Saved merged: {path}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nAll outputs in: {dirs['figures']}")
    print(f"Processed data in: {dirs['data']}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Stereo-seq CellBin PDAC Analysis Pipeline')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Path to 24201-02-analysis-01062025SX directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: base_dir/downstream_analysis)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint (default: False)')
    parser.add_argument('--force', action='store_true',
                        help='Force rerun all steps, ignore checkpoints (default: False)')
    parser.add_argument('--recluster_only', action='store_true', default=False,
                        help='Fast mode: load existing merged_all_treatments.h5ad and '
                             're-run neighbors + Leiden clustering only (~30 min).')
    parser.add_argument('--deg_only', action='store_true', default=False,
                        help='Fast mode: load existing merged_annotated.h5ad (from Step 2) '
                             'and only run the DEG volcano/bar-chart figures (Fig 6H/6I).')
    parser.add_argument('--leiden_resolution', type=float, default=None,
                        help='Override Leiden resolution for --recluster_only mode.')
    parser.add_argument('--resolution', type=str, default='cellbin',
                        choices=['cellbin', 'bin50'],
                        help='SAW resolution to load: cellbin (per-cell) or bin50 (50um bins).')
    args = parser.parse_args()

    global RESOLUTION_LABEL
    RESOLUTION_LABEL = 'Bin50' if args.resolution == 'bin50' else 'CellBin'

    base_dir = args.base_dir
    output_root = args.output_dir if args.output_dir else base_dir

    # ----------------------------------------------------------
    # FAST PATH: DEG figures only, from Step 2's annotated h5ad
    # ----------------------------------------------------------
    if args.deg_only:
        dirs = setup_output(base_dir, output_root)
        h5ad_path = os.path.join(dirs['data'], 'merged_annotated.h5ad')
        if not os.path.exists(h5ad_path):
            print(f"ERROR: {h5ad_path} not found. Run Step 2 first.")
            sys.exit(1)
        print(f"Loading: {h5ad_path} ...")
        adata_merged = sc.read_h5ad(h5ad_path)
        print(f"  {adata_merged.n_obs:,} cells x {adata_merged.n_vars:,} genes")
        differential_expression(adata_merged, dirs)

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from collect_results import collect
        rel = os.path.join('downstream_analysis', 'figures', '05_DEG')
        collect(output_root, {
            os.path.join(rel, 'fig16_volcano_GPH_IT_vs_Sham'): 'fig6_h',
            os.path.join(rel, 'fig16b_top_degs_bar'):          'fig6_i',
        })
        print("\n  DEG-ONLY COMPLETE.")
        return

    # ----------------------------------------------------------
    # FAST PATH: recluster existing merged h5ad only
    # ----------------------------------------------------------
    if args.recluster_only:
        import time
        print("=" * 60)
        print("MODE: RECLUSTER ONLY (fast, ~30 min)")
        print("=" * 60)
        dirs = setup_output(base_dir, output_root)

        h5ad_path = os.path.join(dirs['data'], 'merged_all_treatments.h5ad')
        if not os.path.exists(h5ad_path):
            print(f"ERROR: {h5ad_path} not found. Run full Step 1 first.")
            sys.exit(1)

        print(f"Loading: {h5ad_path} ...")
        t0 = time.time()
        adata_merged = sc.read_h5ad(h5ad_path)
        print(f"  Loaded {adata_merged.n_obs:,} cells × {adata_merged.n_vars:,} genes in {time.time()-t0:.0f}s")
        print(f"  obsm keys: {list(adata_merged.obsm.keys())}")

        if 'X_pca_harmony' in adata_merged.obsm:
            use_rep = 'X_pca_harmony'
            n_pcs = adata_merged.obsm['X_pca_harmony'].shape[1]
            print(f"  Using Harmony embedding ({n_pcs} PCs)")
        else:
            use_rep = 'X_pca'
            n_pcs = min(50, adata_merged.obsm.get('X_pca', np.zeros((1, 1))).shape[1])
            print(f"  Using PCA embedding ({n_pcs} PCs)")

        # k=15 gives a finer neighbor graph than the scanpy default.
        k = 15
        print(f"\n  Re-computing neighbors (k={k}, n_pcs={n_pcs}, rep={use_rep})...")
        t1 = time.time()
        sc.pp.neighbors(adata_merged, n_pcs=n_pcs, n_neighbors=k, use_rep=use_rep)
        print(f"  Done in {time.time()-t1:.0f}s")

        print("  Re-computing UMAP (min_dist=0.3, spread=1.0)...")
        t2 = time.time()
        sc.tl.umap(adata_merged, min_dist=0.3, spread=1.0)
        print(f"  Done in {time.time()-t2:.0f}s")

        # Resolution search extends to 8.0, targeting 25+ clusters.
        resolutions = [args.leiden_resolution] if args.leiden_resolution is not None else [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

        print("\n  Leiden clustering (targeting 25+ clusters):")
        chosen_res = resolutions[-1]
        for res in resolutions:
            sc.tl.leiden(adata_merged, resolution=res, key_added='leiden_merged')
            n_cl = adata_merged.obs['leiden_merged'].nunique()
            sizes = adata_merged.obs['leiden_merged'].value_counts()
            top5_pct = sizes.head(5).sum() / len(adata_merged) * 100
            print(f"    resolution={res}: {n_cl} clusters  (top-5 clusters = {top5_pct:.1f}% of cells)")
            chosen_res = res
            if n_cl >= 25:
                break

        n_clusters = adata_merged.obs['leiden_merged'].nunique()
        sizes = adata_merged.obs['leiden_merged'].value_counts()
        print(f"\n  FINAL: {n_clusters} clusters at resolution={chosen_res}")
        print(f"  Cluster size range: {sizes.min():,} – {sizes.max():,} cells (median {int(sizes.median()):,})")
        print(f"  Top-5 clusters: {sizes.head(5).sum()/len(adata_merged)*100:.1f}% of cells")

        print(f"\n  Saving updated h5ad: {h5ad_path}")
        adata_merged.write_h5ad(h5ad_path)
        print(f"  Saved. File size: {os.path.getsize(h5ad_path)/1e9:.1f} GB")
        print("\n  RECLUSTER COMPLETE. Re-run Step 2 to update annotations.")
        return

    # ----------------------------------------------------------
    # FULL PIPELINE PATH
    # ----------------------------------------------------------
    print("="*60)
    print("STEREO-SEQ CELLBIN ANALYSIS PIPELINE")
    print("PDAC 4-Treatment Comparison")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Output root: {output_root}")
    print(f"Samples: {', '.join(TREATMENT_ORDER)}")
    if args.resume:
        print("Mode: RESUME (loading checkpoints if available)")
    elif args.force:
        print("Mode: FORCE (ignoring all checkpoints)")
    print()

    dirs = setup_output(base_dir, output_root)

    if not args.force and args.resume and checkpoint_exists('step1_load', dirs):
        print("\n" + "="*60)
        print("STEP 1: LOAD SAMPLES (from checkpoint)")
        print("="*60)
        adatas = load_checkpoint('step1_load', dirs)
    else:
        adatas = load_samples(base_dir, args.resolution)
        save_checkpoint('step1_load', adatas, dirs)

    if not args.force and args.resume and checkpoint_exists('step2_qc', dirs):
        print("\n" + "="*60)
        print("STEP 2: QC (from checkpoint)")
        print("="*60)
        adatas = load_checkpoint('step2_qc', dirs)
    else:
        adatas = run_qc(adatas, dirs)
        save_checkpoint('step2_qc', adatas, dirs)

    if not args.force and args.resume and checkpoint_exists('step3_filter', dirs):
        print("\n" + "="*60)
        print("STEP 3: FILTER CELLS (from checkpoint)")
        print("="*60)
        adatas = load_checkpoint('step3_filter', dirs)
    else:
        adatas = filter_cells(adatas, dirs)
        save_checkpoint('step3_filter', adatas, dirs)

    if not args.force and args.resume and checkpoint_exists('step4_normalize', dirs):
        print("\n" + "="*60)
        print("STEP 4: NORMALIZE & REDUCE (from checkpoint)")
        print("="*60)
        adatas = load_checkpoint('step4_normalize', dirs)
    else:
        adatas = normalize_and_reduce(adatas, dirs)
        save_checkpoint('step4_normalize', adatas, dirs)
        gc.collect()

    if not args.force and args.resume and checkpoint_exists('step5_merge', dirs):
        print("\n" + "="*60)
        print("STEP 5: MERGE & INTEGRATE (from checkpoint)")
        print("="*60)
        adata_merged = load_checkpoint('step5_merge', dirs)
    else:
        adata_merged = merge_and_integrate(adatas, dirs, args)
        save_checkpoint('step5_merge', adata_merged, dirs)
        gc.collect()

    if not args.force and args.resume and checkpoint_exists('step6_markers', dirs):
        print("\n" + "="*60)
        print("STEP 6: FIND MARKERS (from checkpoint)")
        print("="*60)
        adata_merged = load_checkpoint('step6_markers', dirs)
    else:
        adata_merged = find_markers(adata_merged, dirs)
        save_checkpoint('step6_markers', adata_merged, dirs)

    if not args.force and args.resume and checkpoint_exists('step7_spatial', dirs):
        print("\n" + "="*60)
        print("STEP 7: SPATIAL PLOTS (skipped - already complete)")
        print("="*60)
    else:
        spatial_plots(adatas, dirs)
        save_checkpoint('step7_spatial', {'completed': True}, dirs)

    if not args.force and args.resume and checkpoint_exists('step8_composition', dirs):
        print("\n" + "="*60)
        print("STEP 8: COMPOSITION ANALYSIS (from checkpoint)")
        print("="*60)
        adata_merged = load_checkpoint('step8_composition', dirs)
    else:
        adata_merged = composition_analysis(adata_merged, dirs)
        save_checkpoint('step8_composition', adata_merged, dirs)

    if not args.force and args.resume and checkpoint_exists('step9_deg', dirs):
        print("\n" + "="*60)
        print("STEP 9: DIFFERENTIAL EXPRESSION (from checkpoint)")
        print("="*60)
        adata_merged = load_checkpoint('step9_deg', dirs)
    else:
        adata_merged = differential_expression(adata_merged, dirs)
        save_checkpoint('step9_deg', adata_merged, dirs)

    save_data(adatas, adata_merged, dirs)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    rel = os.path.join('downstream_analysis', 'figures', '05_DEG')
    collect(args.output_dir or args.base_dir, {
        os.path.join(rel, 'fig16_volcano_GPH_IT_vs_Sham'): 'fig6_h',
        os.path.join(rel, 'fig16b_top_degs_bar'):          'fig6_i',
    })


if __name__ == '__main__':
    main()
