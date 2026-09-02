#!/usr/bin/env python3
"""
CNV Analysis, Spatial Niche Detection & Tumor Microenvironment Scoring

Run after step02_build_annotated_h5ad.py completes.

Input: merged_annotated.h5ad from Step 2
Output: CNV maps, spatial niches (BANKSY), neighborhood enrichment (squidpy)

Analyses:
  1. Copy Number Variation inference (infercnvpy) - identifies malignant cells
  2. Spatial neighborhood enrichment (squidpy) - cell type co-localization
  3. BANKSY spatial domain identification - data-driven tissue niches
  4. Malignant cell re-annotation and subclonality
  5. Tumor microenvironment (TME) scoring
  6. Integration summary figure

Usage:
  python step13_cnv_spatial_niche_analysis.py \\
    --input_dir /path/to/work_dir \\
    --output_dir /path/to/work_dir
"""

import os
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import zscore
from scipy.spatial.distance import cdist
import pickle

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================
plt.rcParams.update({
    'font.family': 'Arial',
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
# defaults, silently reverting the Arial chain set above to DejaVu. Re-assert.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'Helvetica',
                                   'Nimbus Sans', 'FreeSans', 'DejaVu Sans']

# ============================================================================
# CONSTANTS
# ============================================================================
TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
TREATMENT_COLORS = {
    'Sham': '#2166AC',
    'IT': '#EF8A62',
    'GPH': '#67A9CF',
    'GPH+IT': '#B2182B',
}

# Normal reference cell types for CNV inference
# These are used as baseline (diploid) reference
CNV_REFERENCE_TYPES = [
    'B cells', 'T cells', 'CD4 T cells', 'CD8 T cells', 'NK cells',
    'Tregs', 'Cytotoxic T cells', 'Effector CD4+ T cells',
    'B cell', 'T cell', 'NK cell', 'Plasma cells',
]

# TME signature gene sets (mouse gene symbols)
TME_SIGNATURES = {
    'Cytotoxic_Score':       ['Gzmb', 'Gzma', 'Prf1', 'Nkg7', 'Ifng', 'Fasl', 'Tnf'],
    'Immunosuppression_Score': ['Foxp3', 'Il10', 'Tgfb1', 'Cd274', 'Arg1', 'Il4', 'Il13'],
    'IFNg_Response':         ['Irf1', 'Stat1', 'Cxcl9', 'Cxcl10', 'Mx1', 'Ifit1', 'Oas1a'],
    'EMT_Score':             ['Vim', 'Fn1', 'Cdh2', 'Zeb1', 'Zeb2', 'Snai1', 'Snai2', 'Twist1'],
    'Angiogenesis':          ['Vegfa', 'Kdr', 'Pecam1', 'Ang', 'Angpt2', 'Tie1', 'Flt1'],
    'Stromal_Score':         ['Col1a1', 'Col3a1', 'Acta2', 'Postn', 'Fn1', 'Tnc', 'Dcn'],
    'Proliferation':         ['Mki67', 'Top2a', 'Pcna', 'Ccnb1', 'Cdk1', 'Mcm2', 'Birc5'],
    'Antigen_Presentation':  ['H2-Aa', 'H2-Ab1', 'H2-Eb1', 'Cd74', 'B2m', 'Tap1', 'Nlrc5'],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_dirs(output_dir):
    """Build the output directory paths for Step 7.

    Only `data` is created here; the per-analysis figure folders are created
    lazily by the function that writes into them, since --skip_cnv/
    --skip_squidpy/--skip_banksy/--skip_tme can each skip one entirely.
    """
    base = os.path.join(output_dir, 'downstream_analysis')
    dirs = {
        'cnv':          os.path.join(base, 'figures', '20_cnv_malignant'),
        'spatial_niche': os.path.join(base, 'figures', '21_spatial_niches'),
        'tme':          os.path.join(base, 'figures', '22_tme_scoring'),
        'summary':      os.path.join(base, 'figures', '23_step7_summary'),
        'data':         os.path.join(base, 'processed_data'),
    }
    os.makedirs(dirs['data'], exist_ok=True)
    return dirs


# ============================================================================
# ANALYSIS 1: COPY NUMBER VARIATION (infercnvpy)
# ============================================================================

def run_infercnv_analysis(adata, dirs):
    """
    Infer copy number variations to identify malignant (tumor) cells.
    Uses immune cells as diploid reference population.
    This is critical for PDAC to distinguish tumor from normal epithelial cells.
    """
    print("\n" + "=" * 60)
    print("COPY NUMBER VARIATION ANALYSIS (infercnvpy)")
    print("=" * 60)

    try:
        import infercnvpy as cnv
        print(f"  infercnvpy version: {cnv.__version__}")
    except ImportError:
        print("  WARNING: infercnvpy not installed.")
        print("  Install with: pip install infercnvpy")
        print("  Skipping CNV analysis.")
        return adata

    if 'cell_type_auto' not in adata.obs.columns:
        print("  ERROR: No cell type annotation found. Run Step 2 first.")
        return adata

    # Identify reference (normal) cells
    cell_types = adata.obs['cell_type_auto'].unique()
    reference_cats = [ct for ct in cell_types
                      if any(ref.lower() in ct.lower() for ref in CNV_REFERENCE_TYPES)]

    if len(reference_cats) == 0:
        reference_cats = [ct for ct in cell_types
                          if any(kw in ct.lower() for kw in
                                 ['t cell', 'b cell', 'nk', 'lymph', 'plasma'])]

    if len(reference_cats) == 0:
        print("  WARNING: No reference cell types found. Using non-epithelial cells.")
        reference_cats = [ct for ct in cell_types
                          if not any(kw in ct.lower() for kw in
                                     ['tumor', 'cancer', 'epithelial', 'pdac', 'ductal'])]

    print(f"  Reference cell types ({len(reference_cats)}): {reference_cats[:5]}...")
    print(f"  Reference cells: {adata.obs['cell_type_auto'].isin(reference_cats).sum():,}")

    # Use raw counts if available: use ALL genes (not just HVGs) for CNV
    # CNV inference needs chromosome-distributed genes, not just HVGs
    if adata.raw is not None:
        adata_cnv = adata.raw.to_adata().copy()
        # Copy obs columns (treatment, cell_type_auto, etc.) from main adata
        for col in ['cell_type_auto', 'treatment', 'sample']:
            if col in adata.obs.columns:
                adata_cnv.obs[col] = adata.obs[col].values
        print(f"  Using raw counts for CNV inference ({adata_cnv.n_vars:,} genes)")
    elif 'counts' in adata.layers:
        adata_cnv = adata.copy()
        adata_cnv.X = adata.layers['counts'].copy()
        print("  Using counts layer for CNV inference")
    else:
        adata_cnv = adata.copy()
        print("  WARNING: No raw counts found. Using current X (may be normalized).")

    try:
        print("\n  Running infercnvpy...")
        print("  (This may take 10-30 minutes for large datasets)")

        # Subsample if too large for speed
        subsampled_for_cnv = False
        if adata_cnv.n_obs > 100000:
            print(f"  Subsampling to 80,000 cells (was {adata_cnv.n_obs:,})")
            sc.pp.subsample(adata_cnv, n_obs=80000, random_state=42)
            subsampled_for_cnv = True

        # infercnvpy requires chromosome/start/end columns in adata.var
        needs_genomic = not all(c in adata_cnv.var.columns for c in ('chromosome', 'start', 'end'))
        if needs_genomic:
            print("  Annotating gene genomic positions for infercnvpy (offline GTF)...")
            import urllib.request, gzip, tempfile, re
            gtf_cache = os.path.expanduser('~/.cache/stereoseq/Mus_musculus.GRCm39.111.gtf.gz')
            if not os.path.exists(gtf_cache):
                os.makedirs(os.path.dirname(gtf_cache), exist_ok=True)
                gtf_url = 'https://ftp.ensembl.org/pub/release-111/gtf/mus_musculus/Mus_musculus.GRCm39.111.gtf.gz'
                print(f"  Downloading mouse GTF: {gtf_url}")
                urllib.request.urlretrieve(gtf_url, gtf_cache)
                print(f"  Cached: {gtf_cache}")

            # Parse GTF for gene-level chromosome/start/end
            gene_pos = {}
            gname_re = re.compile(r'gene_name "([^"]+)"')
            with gzip.open(gtf_cache, 'rt') as fh:
                for line in fh:
                    if line.startswith('#'): continue
                    f = line.split('\t')
                    if len(f) < 9 or f[2] != 'gene': continue
                    m = gname_re.search(f[8])
                    if not m: continue
                    name = m.group(1)
                    chrom = f[0] if f[0].startswith('chr') else f'chr{f[0]}'
                    if name not in gene_pos:
                        gene_pos[name] = (chrom, int(f[3]), int(f[4]))
            print(f"  Parsed {len(gene_pos):,} mouse genes from GTF")

            chroms, starts, ends = [], [], []
            for g in adata_cnv.var_names:
                pos = gene_pos.get(g)
                if pos is None:
                    chroms.append(None); starts.append(None); ends.append(None)
                else:
                    chroms.append(pos[0]); starts.append(pos[1]); ends.append(pos[2])
            adata_cnv.var['chromosome'] = chroms
            adata_cnv.var['start'] = starts
            adata_cnv.var['end']   = ends
            keep = adata_cnv.var['chromosome'].notna()
            adata_cnv = adata_cnv[:, keep].copy()
            adata_cnv.var['start'] = adata_cnv.var['start'].astype(int)
            adata_cnv.var['end']   = adata_cnv.var['end'].astype(int)
            print(f"  Genomic positions annotated for {keep.sum():,} / {len(keep):,} genes")

        cnv.tl.infercnv(
            adata_cnv,
            reference_key='cell_type_auto',
            reference_cat=reference_cats,
            window_size=100,
            step=5,
        )
        print("  CNV inference complete!")

        cnv.tl.pca(adata_cnv)
        cnv.pp.neighbors(adata_cnv)
        cnv.tl.leiden(adata_cnv, key_added='cnv_leiden')
        cnv.tl.umap(adata_cnv)

        # Compute CNV score
        if 'cnv' in adata_cnv.obsm:
            cnv_matrix = adata_cnv.obsm['cnv']
            cnv_score = np.mean(np.abs(cnv_matrix), axis=1)
            adata_cnv.obs['cnv_score'] = cnv_score

            cnv_threshold = np.percentile(cnv_score, 75)
            epithelial_kws = ['epithelial', 'tumor', 'pdac', 'ductal', 'cancer', 'epdac', 'mpdac']
            is_epithelial = adata_cnv.obs['cell_type_auto'].str.lower().str.contains(
                '|'.join(epithelial_kws), na=False)
            adata_cnv.obs['is_malignant'] = (
                (cnv_score > cnv_threshold) & is_epithelial
            ).astype(str)
            adata_cnv.obs['is_malignant'] = adata_cnv.obs['is_malignant'].map(
                {'True': 'Malignant', 'False': 'Non-malignant'})

            n_malignant = (adata_cnv.obs['is_malignant'] == 'Malignant').sum()
            print(f"\n  CNV Score Statistics:")
            print(f"    Mean CNV score:          {cnv_score.mean():.4f}")
            print(f"    Median CNV score:        {np.median(cnv_score):.4f}")
            print(f"    Threshold (75th pctile): {cnv_threshold:.4f}")
            print(f"    Malignant cells:         {n_malignant:,} ({100 * n_malignant / len(adata_cnv):.1f}%)")

            # Transfer results back to main adata
            if len(adata_cnv) == len(adata):
                adata.obs['cnv_score'] = adata_cnv.obs['cnv_score'].values
                adata.obs['is_malignant'] = adata_cnv.obs['is_malignant'].values
                adata.obs['cnv_inferred'] = True
            else:
                # Use reindex instead of join to avoid AnnData obs replacement issues
                adata.obs['cnv_score'] = adata_cnv.obs['cnv_score'].reindex(adata.obs_names)
                adata.obs['is_malignant'] = adata_cnv.obs['is_malignant'].reindex(adata.obs_names).fillna('Unknown')
                adata.obs['cnv_inferred'] = adata.obs['cnv_score'].notna()
                if subsampled_for_cnv:
                    print("  NOTE: CNV computed on subsample; non-computed cells kept as NaN/Unknown.")

        # --- Figure 1: CNV Chromosome Heatmap ---
        try:
            # cnv.pl.chromosome_heatmap creates its own figure, DO NOT pass ax=
            cnv.pl.chromosome_heatmap(adata_cnv, groupby='cell_type_auto', show=False,
                                      figsize=(34, 17), vmin=-0.3, vmax=0.3)
            plt.suptitle('Copy Number Variation Heatmap by Cell Type',
                         fontsize=20)
            # Not a curated paper figure -- skip saving (computation above still used).
            # plt.savefig(os.path.join(dirs['cnv'], 'fig_cnv_chromosome_heatmap.png'),
            # dpi=300, bbox_inches='tight')
            plt.close('all')
            print("  Saved: fig_cnv_chromosome_heatmap.png")
        except Exception as e:
            print(f"  Note: Chromosome heatmap failed ({e}), trying alternative...")
            plt.close('all')
            if 'cnv' in adata_cnv.obsm:
                cnv_mat = adata_cnv.obsm['cnv']
                n_show = min(500, adata_cnv.n_obs)
                idx_show = np.random.choice(adata_cnv.n_obs, n_show, replace=False)
                fig, ax = plt.subplots(figsize=(34, 14))
                im = ax.imshow(cnv_mat[idx_show], aspect='auto', cmap='RdBu_r',
                               vmin=-0.5, vmax=0.5, interpolation='none')
                plt.colorbar(im, ax=ax, label='CNV Score', shrink=0.5)
                ax.set_xlabel('Genomic Position (genes)', fontsize=15)
                ax.set_ylabel('Cells (sampled)', fontsize=15)
                ax.set_title('Copy Number Variation Matrix', fontsize=20)
                plt.tight_layout()
                # Not a curated paper figure -- skip saving (computation above still used).
                # plt.savefig(os.path.join(dirs['cnv'], 'fig_cnv_matrix.png'),
                # dpi=300, bbox_inches='tight')
                plt.close()
                print("  Saved: fig_cnv_matrix.png")

        # --- Figure 2: CNV Score Distribution ---
        if 'cnv_score' in adata.obs.columns and adata.obs['cnv_score'].notna().sum() > 0:
            fig, axes = plt.subplots(1, 3, figsize=(42, 13))

            if 'X_umap' in adata.obsm:
                sc.pl.umap(adata, color='cnv_score', ax=axes[0], show=False,
                           title='CNV Score on UMAP', frameon=False, s=1,
                           cmap='YlOrRd',
                           vmin=0,
                           vmax=np.nanpercentile(adata.obs['cnv_score'], 99))
            else:
                axes[0].text(0.5, 0.5, 'No UMAP available', ha='center', va='center',
                             transform=axes[0].transAxes, fontsize=14)
                axes[0].axis('off')

            ct_cnv = adata.obs.groupby('cell_type_auto')['cnv_score'].mean().sort_values(
                ascending=False)
            top15 = ct_cnv.head(15)
            malignant_kws = ['tumor', 'pdac', 'epithelial', 'cancer']
            colors_bar = [
                '#B2182B' if any(k in ct.lower() for k in malignant_kws) else '#4393C3'
                for ct in top15.index
            ]
            axes[1].barh(range(len(top15)), top15.values[::-1], color=colors_bar[::-1])
            axes[1].set_yticks(range(len(top15)))
            axes[1].set_yticklabels(top15.index[::-1], fontsize=11)
            axes[1].set_xlabel('Mean CNV Score', fontsize=15)
            axes[1].set_title('Mean CNV Score per Cell Type\n(Red = likely malignant)',
                              fontsize=16)
            sns.despine(ax=axes[1])

            if 'is_malignant' in adata.obs.columns and 'X_umap' in adata.obsm:
                sc.pl.umap(adata, color='is_malignant', ax=axes[2], show=False,
                           title='Malignant Cell Classification',
                           palette={'Malignant': '#B2182B', 'Non-malignant': '#4393C3', 'Unknown': '#BDBDBD'},
                           frameon=False, s=1)
            else:
                axes[2].axis('off')

            plt.suptitle('Copy Number Variation Analysis: Malignant Cell Identification',
                         fontsize=24)
            plt.tight_layout()
            # Not a curated paper figure -- skip saving (computation above still used).
            # plt.savefig(os.path.join(dirs['cnv'], 'fig_cnv_score_distribution.png'),
            # dpi=300, bbox_inches='tight')
            plt.close()
            print("  Saved: fig_cnv_score_distribution.png")

        # --- Figure 3: CNV per treatment ---
        if 'cnv_score' in adata.obs.columns and adata.obs['cnv_score'].notna().sum() > 0 and 'treatment' in adata.obs.columns:
            treatments_present = [t for t in TREATMENT_ORDER
                                  if t in adata.obs['treatment'].unique()]
            n_t = len(treatments_present)
            fig, axes = plt.subplots(1, n_t, figsize=(11 * n_t, 11))
            if n_t == 1:
                axes = [axes]
            for ax, treatment in zip(axes, treatments_present):
                mask = adata.obs['treatment'] == treatment
                adata_t = adata[mask]
                if 'X_umap' in adata.obsm:
                    sc.pl.umap(adata_t, color='cnv_score', ax=ax, show=False,
                               title=f'{treatment}\nCNV Score',
                               frameon=False, s=1, cmap='YlOrRd',
                               vmin=0,
                               vmax=np.nanpercentile(adata.obs['cnv_score'], 99))
                else:
                    ax.text(0.5, 0.5, f'{treatment}\nNo UMAP', ha='center', va='center',
                            transform=ax.transAxes, fontsize=13)
                    ax.axis('off')
            plt.suptitle('CNV Score per Treatment', fontsize=24)
            plt.tight_layout()
            # Not a curated paper figure -- skip saving (computation above still used).
            # plt.savefig(os.path.join(dirs['cnv'], 'fig_cnv_per_treatment.png'),
            # dpi=300, bbox_inches='tight')
            plt.close()
            print("  Saved: fig_cnv_per_treatment.png")

        # Save CNV results table
        obs_cols = [c for c in ['cell_type_auto', 'treatment', 'cnv_score', 'is_malignant', 'cnv_inferred']
                    if c in adata.obs.columns]
        if obs_cols:
            adata.obs[obs_cols].to_csv(
                os.path.join(dirs['data'], 'cnv_malignant_classification.csv'))
            print("  Saved: cnv_malignant_classification.csv")

        print("\n  CNV analysis complete!")

    except Exception as e:
        print(f"  ERROR: infercnvpy analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return adata


# ============================================================================
# ANALYSIS 2: SQUIDPY SPATIAL NEIGHBORHOOD ENRICHMENT
# ============================================================================

def run_squidpy_analysis(adata, dirs):
    """
    Run squidpy spatial analyses:
    - Spatial neighborhood graph construction
    - Cell type neighborhood enrichment / depletion
    - Co-occurrence scores between cell types
    - Spatial autocorrelation (Moran's I)
    """
    os.makedirs(dirs['spatial_niche'], exist_ok=True)
    print("\n" + "=" * 60)
    print("SQUIDPY: SPATIAL NEIGHBORHOOD ANALYSIS")
    print("=" * 60)

    try:
        import squidpy as sq
        print(f"  squidpy version: {sq.__version__}")
    except ImportError:
        print("  WARNING: squidpy not installed.")
        print("  Install with: pip install squidpy")
        print("  Skipping squidpy analysis.")
        return adata

    if 'spatial' not in adata.obsm:
        print("  ERROR: No spatial coordinates found in adata.obsm['spatial']")
        print("  Skipping squidpy analysis.")
        return adata

    if 'cell_type_auto' not in adata.obs.columns:
        print("  ERROR: No cell type annotation found. Run Step 2 first.")
        return adata

    all_nhood_results = {}

    for treatment in TREATMENT_ORDER:
        print(f"\n--- {treatment} ---")
        mask = adata.obs['treatment'] == treatment
        adata_t = adata[mask].copy()
        n_cells = adata_t.n_obs

        if n_cells < 200:
            print(f"  Too few cells ({n_cells}), skipping.")
            continue

        print(f"  Cells: {n_cells:,}")

        if n_cells > 50000:
            print(f"  Subsampling to 50,000 cells for performance...")
            sc.pp.subsample(adata_t, n_obs=50000, random_state=42)
            n_cells = adata_t.n_obs

        try:
            sq.gr.spatial_neighbors(
                adata_t,
                coord_type='generic',
                spatial_key='spatial',
                n_neighs=6,
                delaunay=True,
            )
            print(f"  Spatial graph built")

            sq.gr.nhood_enrichment(adata_t, cluster_key='cell_type_auto', seed=42)
            print(f"  Neighborhood enrichment computed")

            try:
                sq.gr.co_occurrence(
                    adata_t,
                    cluster_key='cell_type_auto',
                    spatial_key='spatial',
                    interval=np.linspace(0, 500, 50),
                    n_splits=10,
                )
                print(f"  Co-occurrence scores computed")
            except Exception as e:
                print(f"  Note: Co-occurrence failed ({e})")

            # Moran's I on top variable genes
            if 'highly_variable' in adata_t.var.columns:
                hvg_genes = adata_t.var_names[adata_t.var['highly_variable']].tolist()[:50]
            else:
                hvg_genes = adata_t.var_names[:50].tolist()

            try:
                sq.gr.spatial_autocorr(
                    adata_t,
                    genes=hvg_genes,
                    mode='moran',
                )
                print(f"  Moran's I computed for {len(hvg_genes)} genes")
            except Exception as e:
                print(f"  Note: Moran's I failed ({e})")

            all_nhood_results[treatment] = adata_t

        except Exception as e:
            print(f"  WARNING: squidpy analysis failed for {treatment}: {e}")
            import traceback
            traceback.print_exc()

    if not all_nhood_results:
        print("\n  No squidpy results to plot.")
        return adata

    # --- Figure: Neighborhood enrichment heatmaps (publication quality) ---
    treatment_order = ['Sham', 'IT', 'GPH', 'GPH+IT']
    ordered_results = {t: all_nhood_results[t] for t in treatment_order if t in all_nhood_results}
    if not ordered_results:
        ordered_results = all_nhood_results

    # Extract zscore matrices
    plot_data = {}
    for treatment, adata_t in ordered_results.items():
        try:
            z = adata_t.uns['cell_type_auto_nhood_enrichment']['zscore']
            if hasattr(adata_t.obs['cell_type_auto'], 'cat'):
                ct_names = list(adata_t.obs['cell_type_auto'].cat.categories)
            else:
                ct_names = sorted(adata_t.obs['cell_type_auto'].unique())
            n = min(len(ct_names), z.shape[0])
            plot_data[treatment] = pd.DataFrame(z[:n, :n],
                                                index=ct_names[:n],
                                                columns=ct_names[:n])
            print(f"  Extracted zscore: {treatment} ({n}x{n})")
        except Exception as e:
            print(f"  WARNING: {treatment}: {e}")

    if not plot_data:
        print("\n  No zscore data to plot.")
        return adata

    all_vals = np.concatenate([df.values.ravel() for df in plot_data.values()])
    vmax = max(float(np.nanpercentile(np.abs(all_vals), 98)), 5)

    # Font sizes: 3× publication quality
    FS_TICK  = 144  # cell type tick labels
    FS_TITLE = 192  # panel title
    FS_AXIS  = 168  # per-panel axis labels
    FS_CBAR  = 156  # colorbar
    FS_SUPT  = 204  # suptitle

    # 2×2 layout, large panels to accommodate 3× fonts
    pw, ph = 52, 60
    fig, axes2d = plt.subplots(
        2, 2,
        figsize=(pw * 2 + 8, ph * 2 + 8),
        gridspec_kw={'wspace': 1.20, 'hspace': 1.40},
        facecolor='white',
    )

    flat_axes = [axes2d[0, 0], axes2d[0, 1], axes2d[1, 0], axes2d[1, 1]]
    treatments_ordered = list(plot_data.keys())
    for i in range(len(treatments_ordered), 4):
        flat_axes[i].set_visible(False)

    mappable = None
    for ax, treatment in zip(flat_axes, treatments_ordered):
        df = plot_data[treatment]
        sns.heatmap(
            df,
            cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
            ax=ax, cbar=False,
            linewidths=0.6, linecolor='#e0e0e0',
            xticklabels=True, yticklabels=True,
        )
        ax.set_title(treatment, fontsize=FS_TITLE, pad=24)

        # X-axis: vertical, not 45°. At 45° a label's horizontal footprint grows
        # with the length of the name, and "Effector CD4+ T cells" across 15
        # columns overran its neighbours into an illegible smear. Upright labels
        # occupy only one line-height each regardless of how long the name is,
        # so the column pitch alone decides whether they fit -- and it does.
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=90, ha='center', va='top',
            fontsize=FS_TICK
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0, fontsize=FS_TICK
        )
        # Per-panel axis labels (always visible, never clipped by tight_layout)
        ax.set_xlabel('Neighborhood cell type', fontsize=FS_AXIS, labelpad=16)
        ax.set_ylabel('Reference cell type', fontsize=FS_AXIS, labelpad=16)
        mappable = ax.collections[0]

    # Shared colorbar: added after tight_layout so positions are stable
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.018, 0.72])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Neighborhood enrichment\nZ-score',
                   fontsize=FS_CBAR, labelpad=14)
    cbar.ax.tick_params(labelsize=FS_CBAR - 4)

    fig.suptitle(
        'Neighborhood Enrichment Z-scores per Treatment\n'
        'Red = enriched co-localization   |   Blue = depletion',
        fontsize=FS_SUPT, y=0.99
    )

    for _ext in ('png', 'pdf'):
        plt.savefig(os.path.join(dirs['spatial_niche'],
                                 f'fig_squidpy_nhood_enrichment.{_ext}'),
                    dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    plt.close()
    print("\n  Saved: fig_squidpy_nhood_enrichment.png")

    # --- Figure: Co-occurrence scores ---
    cooc_available = {t: ad for t, ad in all_nhood_results.items()
                      if 'cell_type_auto_co_occurrence' in ad.uns}
    if cooc_available:
        n_cooc = len(cooc_available)
        fig, axes = plt.subplots(1, n_cooc, figsize=(22 * n_cooc, 17))
        if n_cooc == 1:
            axes = [axes]
        for ax, (treatment, adata_t) in zip(axes, cooc_available.items()):
            try:
                top_cts = list(adata_t.obs['cell_type_auto'].value_counts().index[:6])
                sq.pl.co_occurrence(
                    adata_t,
                    cluster_key='cell_type_auto',
                    clusters=top_cts,
                    ax=ax,
                    show=False,
                )
                ax.set_title(f'{treatment}', fontsize=18)
            except Exception as e:
                ax.text(0.5, 0.5,
                        f'Co-occurrence\nplot failed:\n{str(e)[:40]}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.suptitle('Cell Type Co-occurrence by Distance (squidpy)',
                     fontsize=24)
        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['spatial_niche'], 'fig_squidpy_co_occurrence.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig_squidpy_co_occurrence.png")

    # Save neighborhood enrichment z-score tables
    for treatment, adata_t in all_nhood_results.items():
        key = 'cell_type_auto_nhood_enrichment'
        if key in adata_t.uns:
            zscore_mat = adata_t.uns[key].get('zscore', None)
            if zscore_mat is not None:
                ct_names = sorted(adata_t.obs['cell_type_auto'].unique())
                n = min(len(ct_names), zscore_mat.shape[0])
                zscore_df = pd.DataFrame(zscore_mat[:n, :n],
                                         index=ct_names[:n], columns=ct_names[:n])
                zscore_df.to_csv(
                    os.path.join(dirs['data'], f'nhood_enrichment_{treatment}.csv'))

    print("\n  squidpy analysis complete!")
    return adata


# ============================================================================
# ANALYSIS 3: BANKSY SPATIAL DOMAIN IDENTIFICATION
# ============================================================================

def run_banksy_analysis(adata, dirs):
    """
    Run BANKSY for data-driven spatial domain identification.
    BANKSY incorporates cell neighborhood information into clustering,
    producing more spatially coherent domains than standard Leiden clustering.

    Falls back to a spatially-weighted PCA approach if banksy-py is not installed.
    """
    os.makedirs(dirs['spatial_niche'], exist_ok=True)
    print("\n" + "=" * 60)
    print("BANKSY: SPATIAL DOMAIN IDENTIFICATION")
    print("=" * 60)

    try:
        import banksy
        BANKSY_AVAILABLE = True
        print("  BANKSY imported successfully")
    except ImportError:
        BANKSY_AVAILABLE = False
        print("  WARNING: BANKSY not installed.")
        print("  Install with: pip install banksy-py")
        print("  Falling back to squidpy-based spatial clustering...")

    if 'spatial' not in adata.obsm:
        print("  No spatial coordinates found. Skipping spatial domain analysis.")
        return adata

    # --- Spatially-weighted clustering fallback (used when banksy-py absent) ---
    try:
        import squidpy as sq

        banksy_results = {}
        for treatment in TREATMENT_ORDER:
            print(f"\n--- {treatment} ---")
            mask = adata.obs['treatment'] == treatment
            adata_t = adata[mask].copy()

            if adata_t.n_obs < 100:
                print(f"  Too few cells ({adata_t.n_obs}), skipping.")
                continue

            if adata_t.n_obs > 50000:
                print(f"  Subsampling to 50,000 cells...")
                sc.pp.subsample(adata_t, n_obs=50000, random_state=42)

            sq.gr.spatial_neighbors(adata_t, coord_type='generic',
                                    spatial_key='spatial', n_neighs=6, delaunay=True)

            # Build spatial PCA (approximate BANKSY: lambda-weighted neighborhood aggregation)
            if 'X_pca' not in adata_t.obsm:
                if 'X_pca_harmony' in adata_t.obsm:
                    adata_t.obsm['X_pca'] = adata_t.obsm['X_pca_harmony']
                else:
                    sc.pp.pca(adata_t, n_comps=30)

            from scipy.sparse import issparse
            conn_key = 'spatial_connectivities'
            if conn_key in adata_t.obsp and issparse(adata_t.obsp[conn_key]):
                W = adata_t.obsp[conn_key]
                pca_coords = adata_t.obsm['X_pca']
                lambda_param = 0.2
                neighbor_pca = W.dot(pca_coords)
                spatial_pca = (1 - lambda_param) * pca_coords + lambda_param * neighbor_pca
                adata_t.obsm['X_spatial_pca'] = spatial_pca
                print(f"  Spatial-weighted PCA computed (lambda={lambda_param})")

                sc.pp.neighbors(adata_t, use_rep='X_spatial_pca', n_neighbors=15, n_pcs=30)
                sc.tl.umap(adata_t)

                n_domains = 0
                for res in [0.3, 0.5, 0.8]:
                    sc.tl.leiden(adata_t, resolution=res, key_added='spatial_domain')
                    n_domains = adata_t.obs['spatial_domain'].nunique()
                    if n_domains >= 5:
                        break

                print(f"  Found {n_domains} spatial domains")

                col_name = f'spatial_domain_{treatment}'
                adata.obs[col_name] = 'Unknown'
                common_idx = adata.obs.index.intersection(adata_t.obs.index)
                adata.obs.loc[common_idx, col_name] = adata_t.obs.loc[
                    common_idx, 'spatial_domain'].values
                banksy_results[treatment] = adata_t
            else:
                print(f"  WARNING: Spatial graph not available for {treatment}.")

        # Plot spatial domains
        if banksy_results:
            n_t = len(banksy_results)
            fig, axes = plt.subplots(2, n_t, figsize=(8 * n_t, 14))
            if n_t == 1:
                axes = axes.reshape(2, 1)

            for j, (treatment, adata_t) in enumerate(banksy_results.items()):
                # Top row: UMAP coloured by domain
                sc.pl.umap(adata_t, color='spatial_domain',
                           ax=axes[0, j], show=False,
                           legend_loc='right margin', legend_fontsize=11,
                           title=f'{treatment}: Spatial Domains (UMAP)',
                           frameon=False, s=3)

                # Bottom row: tissue map coloured by domain
                ax = axes[1, j]
                if 'spatial' in adata_t.obsm:
                    coords = adata_t.obsm['spatial']
                    domains = adata_t.obs['spatial_domain'].astype(str)
                    unique_domains = sorted(domains.unique())
                    cmap_d = matplotlib.colormaps.get_cmap('tab10').resampled(
                        max(len(unique_domains), 1))
                    d_colors = {d: cmap_d(i) for i, d in enumerate(unique_domains)}
                    colors_arr = [d_colors[d] for d in domains]

                    n_plot = min(50000, adata_t.n_obs)
                    idx = np.random.choice(adata_t.n_obs, n_plot, replace=False)
                    ax.scatter(coords[idx, 0], coords[idx, 1],
                               c=[colors_arr[i] for i in idx],
                               s=0.5, alpha=0.6, rasterized=True)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.set_title(f'{treatment}: Spatial Domains (Tissue)',
                                 fontsize=16)

                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor=d_colors[d], label=f'Domain {d}')
                                       for d in unique_domains]
                    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                              title='Domain', title_fontsize=11)

            plt.suptitle('Spatial Domain Identification (Spatially-Weighted Clustering)',
                         fontsize=24)
            plt.tight_layout()
            # Not a curated paper figure -- skip saving (computation above still used).
            # plt.savefig(os.path.join(dirs['spatial_niche'], 'fig_spatial_domains_banksy.png'),
            # dpi=150, bbox_inches='tight')
            plt.close()
            print("  Saved: fig_spatial_domains_banksy.png")

    except Exception as e:
        print(f"  Spatial domain analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n  Spatial domain analysis complete!")
    return adata


# ============================================================================
# ANALYSIS 4: TUMOR MICROENVIRONMENT (TME) SCORING
# ============================================================================

def run_tme_scoring(adata, dirs):
    """
    Score cells for key TME functional states using curated gene signatures.
    Provides a functional view of the tumor microenvironment across treatments.
    """
    os.makedirs(dirs['tme'], exist_ok=True)
    print("\n" + "=" * 60)
    print("TUMOR MICROENVIRONMENT (TME) SCORING")
    print("=" * 60)

    # Use raw gene space (all genes) when adata.X only has HVG subset
    gene_universe = set(adata.var_names)
    if adata.raw is not None and len(adata.raw.var_names) > len(adata.var_names):
        print(f"  Using adata.raw ({len(adata.raw.var_names):,} genes) for signature lookup")
        gene_universe = set(adata.raw.var_names)
        adata_for_score = adata.raw.to_adata()
        # Copy obs metadata needed for downstream plots
        adata_for_score.obs = adata.obs.copy()
        if 'X_umap' in adata.obsm:
            adata_for_score.obsm['X_umap'] = adata.obsm['X_umap']
    else:
        adata_for_score = adata

    scored_sigs = []
    for sig_name, genes in TME_SIGNATURES.items():
        available = [g for g in genes if g in gene_universe]
        if len(available) >= 2:
            score_key = f'TME_{sig_name}'
            sc.tl.score_genes(adata_for_score, gene_list=available,
                              score_name=score_key, random_state=42)
            # Copy score back to main adata
            adata.obs[score_key] = adata_for_score.obs[score_key].values
            scored_sigs.append(sig_name)
            print(f"  {sig_name}: {len(available)}/{len(genes)} genes scored")
        else:
            print(f"  {sig_name}: too few genes available ({len(available)}), skipping")

    if not scored_sigs:
        print("  WARNING: No TME signatures could be scored.")
        return adata

    # --- Figure 1: TME Scores on UMAP ---
    n_sigs = len(scored_sigs)
    n_cols = 4
    n_rows = int(np.ceil(n_sigs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, sig in enumerate(scored_sigs):
        ax = axes_flat[i]
        score_col = f'TME_{sig}'
        if 'X_umap' in adata.obsm:
            sc.pl.umap(adata, color=score_col, ax=ax, show=False,
                       title=sig.replace('_', ' '), frameon=False, s=0.5,
                       cmap='RdYlBu_r',
                       vmin=np.percentile(adata.obs[score_col], 2),
                       vmax=np.percentile(adata.obs[score_col], 98))
        else:
            ax.text(0.5, 0.5, f'{sig}\n(no UMAP)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.axis('off')

    for i in range(len(scored_sigs), len(axes_flat)):
        axes_flat[i].axis('off')

    plt.suptitle('Tumor Microenvironment (TME) Functional Scores on UMAP',
                 fontsize=24)
    plt.tight_layout()
    _dual_save(plt.gcf(), dirs['tme'], 'fig_tme_scores_umap.png')
    print("\n  Saved: fig_tme_scores_umap.png + small_2x2")

    # --- Figure 2: TME Scores per Treatment (Violin) ---
    treatments_present = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    n_cols_v = 4
    n_rows_v = int(np.ceil(n_sigs / n_cols_v))
    fig, axes = plt.subplots(n_rows_v, n_cols_v, figsize=(8 * n_cols_v, 6 * n_rows_v))
    axes_flat_v = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, sig in enumerate(scored_sigs):
        if i >= len(axes_flat_v):
            break
        ax = axes_flat_v[i]
        score_col = f'TME_{sig}'
        data_plot = [adata.obs.loc[adata.obs['treatment'] == t, score_col].values
                     for t in treatments_present]
        parts = ax.violinplot(data_plot, positions=range(len(data_plot)),
                              showmedians=True, showextrema=False)
        for pc, color in zip(parts['bodies'],
                             [TREATMENT_COLORS[t] for t in treatments_present]):
            pc.set_facecolor(color)
            pc.set_alpha(0.8)
        ax.set_xticks(range(len(data_plot)))
        ax.set_xticklabels(treatments_present, rotation=30, ha='right', fontsize=11)
        ax.set_title(sig.replace('_', ' '), fontsize=14)
        ax.set_ylabel('Score', fontsize=12)
        sns.despine(ax=ax)

    for i in range(len(scored_sigs), len(axes_flat_v)):
        axes_flat_v[i].axis('off')

    plt.suptitle('TME Functional Scores per Treatment', fontsize=24)
    plt.tight_layout()
    _dual_save(fig, dirs['tme'], 'fig_tme_scores_per_treatment.png')
    print("  Saved: fig_tme_scores_per_treatment.png + small_2x2")

    # --- Figure 3: TME Score Heatmap (mean per cell type per treatment) ---
    rows = []
    for treatment in TREATMENT_ORDER:
        if treatment not in adata.obs['treatment'].unique():
            continue
        mask_t = adata.obs['treatment'] == treatment
        for ct in sorted(adata.obs['cell_type_auto'].unique()):
            mask_ct = mask_t & (adata.obs['cell_type_auto'] == ct)
            if mask_ct.sum() < 10:
                continue
            row = {'treatment': treatment, 'cell_type': ct}
            for sig in scored_sigs:
                row[sig] = adata.obs.loc[mask_ct, f'TME_{sig}'].mean()
            rows.append(row)

    if rows:
        tme_df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, len(TREATMENT_ORDER),
                                 figsize=(11 * len(TREATMENT_ORDER), 22))
        if len(TREATMENT_ORDER) == 1:
            axes = [axes]

        for ax, treatment in zip(axes, TREATMENT_ORDER):
            df_t = tme_df[tme_df['treatment'] == treatment].set_index('cell_type')[scored_sigs]
            if df_t.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                ax.axis('off')
                continue
            df_z = df_t.apply(zscore, axis=0).clip(-2, 2).fillna(0)
            sns.heatmap(
                df_z,
                cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Z-score', 'shrink': 0.5},
                xticklabels=[s.replace('_', '\n') for s in scored_sigs],
            )
            ax.set_title(f'{treatment}', fontsize=18)
            ax.set_xlabel('TME Signature', fontsize=14)
            ax.set_ylabel('Cell Type', fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=11)
            ax.tick_params(axis='y', rotation=0, labelsize=10)

        plt.suptitle('TME Functional State per Cell Type per Treatment (Z-score)',
                     fontsize=24)
        plt.tight_layout()
        _dual_save(fig, dirs['tme'], 'fig_tme_heatmap_celltype_treatment.png')
        print("  Saved: fig_tme_heatmap_celltype_treatment.png + small_2x2")
        tme_df.to_csv(
            os.path.join(dirs['data'], 'tme_scores_per_celltype_treatment.csv'), index=False)
        print("  Saved: tme_scores_per_celltype_treatment.csv")

    # --- Figure 4: Spatial TME scores on tissue ---
    if 'spatial' in adata.obsm:
        key_sigs = ['Cytotoxic_Score', 'Immunosuppression_Score', 'IFNg_Response', 'Proliferation']
        key_sigs_avail = [s for s in key_sigs if s in scored_sigs]

        if key_sigs_avail:
            treatments_present = [t for t in TREATMENT_ORDER
                                  if t in adata.obs['treatment'].unique()]
            n_rows_sp = len(key_sigs_avail)
            n_cols_sp = len(treatments_present)
            fig, axes = plt.subplots(n_rows_sp, n_cols_sp,
                                     figsize=(22 * n_cols_sp, 14 * n_rows_sp))
            if axes.ndim == 1:
                axes = axes.reshape(-1, 1) if n_cols_sp == 1 else axes.reshape(1, -1)

            for row_i, sig in enumerate(key_sigs_avail):
                score_col = f'TME_{sig}'
                vmin = np.percentile(adata.obs[score_col], 2)
                vmax = np.percentile(adata.obs[score_col], 98)

                for col_j, treatment in enumerate(treatments_present):
                    ax = axes[row_i, col_j]
                    mask_t = adata.obs['treatment'] == treatment
                    adata_t = adata[mask_t]

                    if adata_t.n_obs == 0:
                        ax.axis('off')
                        continue

                    coords = adata_t.obsm['spatial']
                    scores = adata_t.obs[score_col].values
                    n_plot = min(30000, adata_t.n_obs)
                    idx = np.random.choice(adata_t.n_obs, n_plot, replace=False)

                    scatter = ax.scatter(
                        coords[idx, 0], coords[idx, 1],
                        c=scores[idx], cmap='RdYlBu_r',
                        s=0.3, alpha=0.6, vmin=vmin, vmax=vmax,
                        rasterized=True,
                    )
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.set_title(f'{treatment}\n{sig.replace("_", " ")}',
                                 fontsize=14)

                    if col_j == n_cols_sp - 1:
                        plt.colorbar(scatter, ax=ax, label='Score', shrink=0.6)

            plt.suptitle('Spatial TME Functional Scores', fontsize=24)
            plt.tight_layout()
            _dual_save(plt.gcf(), dirs['tme'], 'fig_tme_spatial_scores.png')
            print("  Saved: fig_tme_spatial_scores.png")

    # --- Figure 5: GSEA-style TME signature enrichment by treatment (slide 17) ---
    plot_tme_gsea_by_treatment(adata, scored_sigs, dirs)

    print("\n  TME scoring complete!")
    return adata


def _save_small(fig, path, target_inches=2.0, min_font=5):
    """Save a 2×2 inch version of fig with scaled fonts."""
    import matplotlib as _mpl
    orig_size = fig.get_size_inches()
    scale = target_inches / max(orig_size)
    texts = list(fig.findobj(_mpl.text.Text))
    orig_fs = [t.get_fontsize() for t in texts]
    for t, sz in zip(texts, orig_fs):
        t.set_fontsize(max(min_font, sz * scale))
    fig.set_size_inches(target_inches, target_inches)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.set_size_inches(orig_size)
    for t, sz in zip(texts, orig_fs):
        t.set_fontsize(sz)


# The only TME figure this script builds that ends up in the paper
# (suppl10_b); every other one below is still computed but not saved.
_CURATED_STEMS = {'fig_tme_gsea_heatmap.png'}


def _dual_save(fig, out_dir, fname, close=True):
    """Save normal + 2×2 small version, if it's a curated paper figure."""
    full_path = os.path.join(out_dir, fname)
    if fname not in _CURATED_STEMS:
        if close:
            plt.close(fig)
        return full_path
    fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    # PDF twin so these panels can go into the vector publication composite
    # without being rasterised.
    fig.savefig(os.path.splitext(full_path)[0] + '.pdf',
                bbox_inches='tight', facecolor='white')
    small_dir = os.path.join(out_dir, 'small_2x2')
    os.makedirs(small_dir, exist_ok=True)
    try:
        _save_small(fig, os.path.join(small_dir, fname))
    except Exception as e:
        print(f"  [warn] small version failed for {fname}: {e}")
    if close:
        plt.close(fig)
    return full_path


def plot_tme_gsea_by_treatment(adata, scored_sigs, dirs):
    """
    True ssGSEA per cell using gseapy.ssgsea().
    Each cell is ranked by gene expression and a proper enrichment score is
    computed for every TME gene set. Scores are then compared across
    treatments with Mann-Whitney U + BH-FDR.
    Outputs: violin figure, boxplot figure, companion heatmap.
    """

    try:
        import gseapy as gp
        HAS_GSEAPY = True
    except ImportError:
        HAS_GSEAPY = False
        print("  [tme-gsea] gseapy not installed, pip install gseapy. "
              "Falling back to sc.tl.score_genes module scores.")

    if 'treatment' not in adata.obs.columns:
        print("  [tme-gsea] missing treatment column, skipping"); return

    treatments_present = [t for t in TREATMENT_ORDER
                          if t in adata.obs['treatment'].unique()]
    if 'Sham' not in treatments_present or len(treatments_present) < 2:
        print("  [tme-gsea] need Sham + ≥1 other treatment, skipping"); return

    # ── Expression source ────────────────────────────────────────────────────
    src = adata.raw.to_adata() if adata.raw is not None else adata
    gene_names = list(src.var_names)

    TREAT_PAL = {
        'Sham': '#4393C3', 'IT': '#F4A582',
        'GPH':  '#92C5DE', 'GPH+IT': '#D6604D',
    }

    N_CELLS = 600   # cells per treatment per signature for ssGSEA
    rng = np.random.default_rng(42)

    # Cell types relevant to each signature: score within biologically correct populations
    ct_key = next((c for c in ['cell_type', 'cell_type_auto', 'cell_type_marker_scoring',
                                'leiden_merged'] if c in adata.obs.columns), None)
    SIG_CELLTYPES = {
        'Cytotoxic_Score':         ['Cytotoxic T cells', 'CD8 T cells', 'NK cells'],
        'Immunosuppression_Score': ['Tregs', 'M2 Macrophage', 'M2 Macrophages'],
        'IFNg_Response':           ['Cytotoxic T cells', 'CD8 T cells', 'NK cells',
                                    'M1 Macrophage', 'M1 Macrophages'],
        'EMT_Score':               ['mPDAC', 'ePDAC'],
        'Angiogenesis':            ['Endothelial cells', 'Endothelial'],
        'Stromal_Score':           ['myCAFs', 'iCAF', 'apCAFs', 'qCAF', 'CAFs'],
        'Proliferation':           None,   # all cell types
        'Antigen_Presentation':    ['M1 Macrophage', 'M2 Macrophage', 'B cells',
                                    'M1 Macrophages', 'M2 Macrophages'],
    }

    # ── Score each signature within relevant cell types ───────────────────────
    score_rows = []

    if HAS_GSEAPY:
        print("  [tme-gsea] running ssGSEA per relevant cell type via gseapy …")
        for sig, gene_set in TME_SIGNATURES.items():
            relevant_cts = SIG_CELLTYPES.get(sig)
            if relevant_cts and ct_key:
                ct_vals = adata.obs[ct_key].astype(str)
                ct_mask = ct_vals.str.contains('|'.join(relevant_cts), case=False, regex=True)
                sig_adata = src[ct_mask]
            else:
                sig_adata = src
            if sig_adata.n_obs < 20:
                print(f"    [tme-gsea] {sig}: too few cells ({sig_adata.n_obs}), skipping")
                continue
            print(f"    {sig}: {sig_adata.n_obs} cells from relevant types")
            for t in treatments_present:
                t_mask = sig_adata.obs['treatment'] == t
                t_sub = sig_adata[t_mask]
                if t_sub.n_obs < 10:
                    continue
                samp = rng.choice(t_sub.n_obs, min(N_CELLS, t_sub.n_obs), replace=False)
                sub = t_sub[samp]
                X = sub.X.toarray() if hasattr(sub.X, 'toarray') else np.asarray(sub.X)
                gene_names_sub = list(sub.var_names)
                expr_df = pd.DataFrame(X.T, index=gene_names_sub,
                                       columns=[f'c{i}' for i in range(len(samp))])
                try:
                    ss = gp.ssgsea(data=expr_df, gene_sets={sig: gene_set},
                                   outdir=None, no_plot=True, min_size=3,
                                   processes=2, verbose=False)
                    res = ss.res2d
                    if 'Term' in res.columns:
                        nes_vals = res.loc[res['Term'] == sig, 'NES'].values
                    elif sig in res.index:
                        nes_vals = res.loc[sig].values
                    else:
                        nes_vals = []
                    for v in nes_vals:
                        score_rows.append({'treatment': t, 'signature': sig,
                                           'score': float(v)})
                except Exception as e:
                    print(f"    ssGSEA failed for {sig}/{t}: {e}")
                    HAS_GSEAPY = False
                    break
            if not HAS_GSEAPY:
                break

    if not HAS_GSEAPY or not score_rows:
        # Fallback: sc.tl.score_genes delta-mean (already in adata.obs)
        print("  [tme-gsea] using sc.tl.score_genes scores (fallback)")
        for sig in TME_SIGNATURES:
            col = f'TME_{sig}'
            if col not in adata.obs.columns:
                continue
            for t in treatments_present:
                vals = adata.obs.loc[adata.obs['treatment'] == t, col].dropna().values
                for v in vals:
                    score_rows.append({'treatment': t, 'signature': sig,
                                       'score': float(v)})

    if not score_rows:
        print("  [tme-gsea] no scores produced, skipping"); return

    df_all = pd.DataFrame(score_rows)
    sig_list = [s for s in TME_SIGNATURES if s in df_all['signature'].unique()]
    df_all.to_csv(os.path.join(dirs['data'], 'tme_ssgsea_scores.csv'), index=False)
    print(f"  Saved: tme_ssgsea_scores.csv ({len(df_all)} rows)")

    # ── Shared plot helper ────────────────────────────────────────────────────
    plt.rcParams.update({'axes.linewidth': 1.3})
    n_sigs = len(sig_list)
    n_cols  = min(3, n_sigs)
    n_rows  = int(np.ceil(n_sigs / n_cols))
    FIG_W, FIG_H = 9.0 * n_cols, 8.0 * n_rows

    # Significance brackets removed: n=1 per treatment, cell-level stats invalid

    def _finish_ax(ax, sig):
        ax.axhline(0, color='#BBBBBB', lw=1.1, ls='--', zorder=0)
        ax.set_xticks(range(len(treatments_present)))
        ax.set_xticklabels(treatments_present, fontsize=18, rotation=30, ha='right')
        ax.tick_params(axis='y', labelsize=16)
        ylabel = 'ssGSEA score' if HAS_GSEAPY else 'Gene module score'
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(sig.replace('_', ' '), fontsize=20, pad=8)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.9, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    # ── Boxplot figure ────────────────────────────────────────────────────────
    fig_b, axes_b = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H), squeeze=False)
    for idx, sig in enumerate(sig_list):
        ax = axes_b.flatten()[idx]
        data = [df_all.loc[(df_all['signature'] == sig) &
                           (df_all['treatment'] == t), 'score'].values
                for t in treatments_present]
        bp = ax.boxplot(data, positions=range(len(treatments_present)),
                        widths=0.55, patch_artist=True, showfliers=False, zorder=3,
                        medianprops=dict(color='black', linewidth=3.0, zorder=5),
                        whiskerprops=dict(color='#222222', linewidth=2.0),
                        capprops=dict(color='#222222', linewidth=2.0),
                        boxprops=dict(linewidth=2.0))
        for patch, t in zip(bp['boxes'], treatments_present):
            patch.set_facecolor(TREAT_PAL.get(t, '#888')); patch.set_alpha(0.65)
        all_v = df_all.loc[df_all['signature'] == sig, 'score'].values
        ylo, yhi = np.nanpercentile(all_v, 1), np.nanpercentile(all_v, 99)
        yspan = yhi - ylo
        q25, q75 = np.nanpercentile(all_v, 25), np.nanpercentile(all_v, 75)
        iqr = q75 - q25
        whisker_top = min(q75 + 1.5 * iqr, yhi)
        ax.set_ylim(ylo - iqr * 0.3, whisker_top + iqr * 0.5)
        _finish_ax(ax, sig)
    for ax in axes_b.flatten()[n_sigs:]:
        ax.set_visible(False)
    score_label = 'ssGSEA' if HAS_GSEAPY else 'Gene Module'
    plt.suptitle(f'TME Functional Signatures — {score_label} Score by Treatment\n(computed within biologically relevant cell types)',
                 fontsize=24, y=1.01)
    plt.tight_layout(h_pad=4.0, w_pad=3.0)
    p = os.path.join(dirs['tme'], 'fig_tme_gsea_by_treatment.png')
    _dual_save(fig_b, dirs['tme'], 'fig_tme_gsea_by_treatment.png')
    print(f"  Saved: fig_tme_gsea_by_treatment.png + small_2x2")

    # ── Violin figure ─────────────────────────────────────────────────────────
    fig_v, axes_v = plt.subplots(n_rows, n_cols, figsize=(FIG_W, FIG_H), squeeze=False)
    for idx, sig in enumerate(sig_list):
        ax = axes_v.flatten()[idx]
        all_v = df_all.loc[df_all['signature'] == sig, 'score'].values
        ylo, yhi = np.nanpercentile(all_v, 1), np.nanpercentile(all_v, 99)
        yspan = yhi - ylo
        clipped = [np.clip(
            df_all.loc[(df_all['signature'] == sig) &
                       (df_all['treatment'] == t), 'score'].values, ylo, yhi)
                   for t in treatments_present]
        vp = ax.violinplot(clipped, positions=range(len(treatments_present)),
                           widths=0.72, showmedians=False, showextrema=False)
        for body, t in zip(vp['bodies'], treatments_present):
            body.set_facecolor(TREAT_PAL.get(t, '#888'))
            body.set_alpha(0.60); body.set_edgecolor('white'); body.set_linewidth(0.8)
        for xi, t in enumerate(treatments_present):
            med = np.median(df_all.loc[(df_all['signature'] == sig) &
                                       (df_all['treatment'] == t), 'score'].values)
            ax.hlines(med, xi - 0.12, xi + 0.12, colors='black',
                      linewidth=2.5, zorder=5)
        ax.set_ylim(ylo - yspan * 0.08, yhi + yspan * 0.12)
        _finish_ax(ax, sig)
    for ax in axes_v.flatten()[n_sigs:]:
        ax.set_visible(False)
    plt.suptitle(f'TME Functional Signatures — {score_label} Score by Treatment (Violin)\n(computed within biologically relevant cell types)',
                 fontsize=24, y=1.01)
    plt.tight_layout(h_pad=4.0, w_pad=3.0)
    pv = os.path.join(dirs['tme'], 'fig_tme_gsea_by_treatment_violin.png')
    _dual_save(fig_v, dirs['tme'], 'fig_tme_gsea_by_treatment_violin.png')
    print(f"  Saved: fig_tme_gsea_by_treatment_violin.png + small_2x2")

    # ── Heatmap of median ssGSEA per (signature × treatment) ─────────────────
    med_mat = (df_all.groupby(['signature', 'treatment'])['score']
               .median().unstack('treatment')
               .reindex(columns=treatments_present))
    fig_h, ax_h = plt.subplots(figsize=(1.6 * len(treatments_present) + 4,
                                        0.6 * len(sig_list) + 2.5))
    # Sized against the canvas (10.4in wide here) so this panel's text lands at
    # the same physical size as the GSEA bar chart it shares Suppl. Fig. 10
    # with: that panel runs ~0.021 points of type per point of figure width,
    # and at the old 10-15pt this one sat at ~0.013 and came out visibly
    # smaller on the page.
    sns.heatmap(med_mat, cmap='RdBu_r', center=0, annot=True, fmt='.3f',
                linewidths=0.4, ax=ax_h,
                annot_kws={'fontsize': 15},
                cbar_kws={'label': f'Median {score_label} score'})
    ax_h.set_xlabel('Treatment', fontsize=20)
    ax_h.set_ylabel('TME signature', fontsize=20)
    ax_h.tick_params(axis='both', labelsize=16)
    cb = ax_h.collections[0].colorbar
    cb.set_label(f'Median {score_label} score', fontsize=17)
    cb.ax.tick_params(labelsize=15)
    ax_h.set_title(f'TME {score_label} score heatmap',
                   fontsize=22)
    plt.tight_layout()
    _dual_save(fig_h, dirs['tme'], 'fig_tme_gsea_heatmap.png')
    print("  Saved: fig_tme_gsea_heatmap.png + small_2x2")

    # ── Trend summary: median score per signature across treatments ────────────
    treat_order = [t for t in ['Sham', 'IT', 'GPH', 'GPH+IT'] if t in treatments_present]
    med_trend = (df_all.groupby(['signature', 'treatment'])['score']
                 .median().unstack('treatment').reindex(columns=treat_order))

    fig_t, ax_t = plt.subplots(figsize=(10, 7))
    cmap_sigs = plt.cm.get_cmap('tab10', len(sig_list))
    for k, sig in enumerate(sig_list):
        vals = med_trend.loc[sig, treat_order].values
        sham_val = vals[0] if not np.isnan(vals[0]) else 0
        color = '#d62728' if vals[-1] > sham_val else '#1f77b4'
        ax_t.plot(treat_order, vals, marker='o', linewidth=2.5,
                  markersize=9, color=cmap_sigs(k),
                  label=sig.replace('_', ' '), zorder=3)
        # arrow at end showing direction vs Sham
        dy = vals[-1] - sham_val
        ax_t.annotate('', xy=(len(treat_order) - 1 + 0.15, vals[-1]),
                      xytext=(len(treat_order) - 1, vals[-1]),
                      arrowprops=dict(arrowstyle='->', color=cmap_sigs(k),
                                      lw=1.5), annotation_clip=False)

    ax_t.axhline(0, color='#BBBBBB', lw=1.2, ls='--', zorder=0)
    ax_t.set_xticks(range(len(treat_order)))
    ax_t.set_xticklabels(treat_order, fontsize=16)
    ax_t.tick_params(axis='y', labelsize=14)
    ax_t.set_ylabel(f'Median {score_label} score', fontsize=16)
    ax_t.set_title(f'TME Signature Trends Across Treatments', fontsize=18)
    ax_t.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12,
                frameon=False, title='Signature', title_fontsize=13)
    ax_t.yaxis.grid(True, linestyle=':', linewidth=0.9, alpha=0.5)
    ax_t.set_axisbelow(True)
    sns.despine(ax=ax_t)
    plt.tight_layout()
    _dual_save(fig_t, dirs['tme'], 'fig_tme_gsea_trends.png')
    print(f"  Saved: fig_tme_gsea_trends.png + small_2x2")


def main():
    parser = argparse.ArgumentParser(
        description='Stereo-seq Step 7: CNV, Spatial Niche & TME Analysis')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Working directory (same as output_dir from Steps 1-2)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--skip_cnv', action='store_true',
                        help='Skip infercnvpy CNV analysis')
    parser.add_argument('--skip_squidpy', action='store_true',
                        help='Skip squidpy neighborhood analysis')
    parser.add_argument('--skip_banksy', action='store_true',
                        help='Skip BANKSY spatial domain analysis')
    parser.add_argument('--skip_tme', action='store_true',
                        help='Skip TME scoring')
    parser.add_argument('--resume_checkpoint', action='store_true',
                        help='If merged_step7_complete.h5ad already exists (a prior run '
                             'finished all four analyses and saved it before failing later, '
                             'e.g. while rendering the summary figure), load it directly and '
                             'skip straight to the summary figure instead of recomputing '
                             'CNV/squidpy/BANKSY/TME from scratch.')
    args = parser.parse_args()

    print("=" * 60)
    print("STEREO-SEQ STEP 7: CNV, SPATIAL NICHE & TME ANALYSIS")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    dirs = setup_dirs(args.output_dir)

    checkpoint_path = os.path.join(dirs['data'], 'merged_step7_complete.h5ad')
    if args.resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        adata = sc.read_h5ad(checkpoint_path)
        print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    else:
        # Load annotated data from Step 2
        annotated_path = os.path.join(
            args.input_dir, 'downstream_analysis', 'processed_data', 'merged_annotated.h5ad')
        if not os.path.exists(annotated_path):
            fallback_path = os.path.join(
                args.input_dir, 'downstream_analysis', 'processed_data',
                'merged_all_treatments.h5ad')
            if os.path.exists(fallback_path):
                annotated_path = fallback_path
                print(f"  WARNING: merged_annotated.h5ad not found. Using merged_all_treatments.h5ad")
            else:
                print(f"ERROR: No input h5ad file found at:")
                print(f"  {annotated_path}")
                print(f"  {fallback_path}")
                sys.exit(1)

        print(f"\nLoading data: {annotated_path}")
        adata = sc.read_h5ad(annotated_path)
        print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"  Columns: {list(adata.obs.columns)[:10]}...")

        if 'treatment' not in adata.obs.columns:
            print("ERROR: 'treatment' column not found in adata.obs")
            print(f"Available columns: {list(adata.obs.columns)}")
            sys.exit(1)

        print(f"\nTreatment counts:")
        for t, n in adata.obs['treatment'].value_counts().items():
            print(f"  {t}: {n:,} cells")

        # =========================================================================
        # ANALYSIS 1: CNV inference (infercnvpy)
        # =========================================================================
        if not args.skip_cnv:
            adata = run_infercnv_analysis(adata, dirs)
        else:
            print("\n[SKIPPED] CNV analysis (--skip_cnv)")

        # =========================================================================
        # ANALYSIS 2: Squidpy spatial neighborhoods
        # =========================================================================
        if not args.skip_squidpy:
            adata = run_squidpy_analysis(adata, dirs)
        else:
            print("\n[SKIPPED] squidpy analysis (--skip_squidpy)")

        # =========================================================================
        # ANALYSIS 3: BANKSY spatial domains
        # =========================================================================
        if not args.skip_banksy:
            adata = run_banksy_analysis(adata, dirs)
        else:
            print("\n[SKIPPED] BANKSY analysis (--skip_banksy)")

        # =========================================================================
        # ANALYSIS 4: TME scoring
        # =========================================================================
        if not args.skip_tme:
            adata = run_tme_scoring(adata, dirs)
        else:
            print("\n[SKIPPED] TME scoring (--skip_tme)")

        # Save updated adata with all new annotations
        print(f"\nSaving updated adata: {checkpoint_path}")
        adata.write(checkpoint_path)
        print(f"Saved: {checkpoint_path}")

    # =========================================================================
    # SUMMARY FIGURE: Step 7 integrated overview
    # =========================================================================
    try:
        print("\nGenerating Step 7 summary figure...")
        n_panels = 4
        fig, axes = plt.subplots(2, 2, figsize=(28, 20))
        axes = axes.flatten()

        # Panel 1: CNV score by cell type
        if 'cnv_score' in adata.obs.columns and adata.obs['cnv_score'].notna().sum() > 0:
            ct_cnv = adata.obs.groupby('cell_type_auto')['cnv_score'].mean().sort_values(ascending=False).head(12)
            malignant_kws = ['tumor', 'pdac', 'epithelial', 'cancer']
            colors = ['#B2182B' if any(k in ct.lower() for k in malignant_kws) else '#4393C3'
                      for ct in ct_cnv.index]
            axes[0].barh(range(len(ct_cnv)), ct_cnv.values[::-1], color=colors[::-1])
            axes[0].set_yticks(range(len(ct_cnv)))
            axes[0].set_yticklabels(ct_cnv.index[::-1], fontsize=10)
            axes[0].set_xlabel('Mean CNV Score', fontsize=12)
            axes[0].set_title('CNV Score by Cell Type\n(Red = malignant candidates)', fontsize=13)
            sns.despine(ax=axes[0])
        else:
            axes[0].text(0.5, 0.5, 'CNV analysis skipped', ha='center', va='center',
                         transform=axes[0].transAxes, fontsize=13)
            axes[0].axis('off')

        # Panel 2: Malignant cell fraction per treatment
        if 'is_malignant' in adata.obs.columns and 'treatment' in adata.obs.columns:
            treat_mal = adata.obs.groupby('treatment')['is_malignant'].apply(
                lambda x: (x == 'Malignant').mean() * 100)
            treat_order = [t for t in TREATMENT_ORDER if t in treat_mal.index]
            treat_mal = treat_mal.reindex(treat_order)
            axes[1].bar(treat_mal.index, treat_mal.values,
                        color=['#4D9221', '#1A6FBF', '#D73027', '#762A83'])
            axes[1].set_ylabel('% Malignant Cells', fontsize=12)
            axes[1].set_title('Malignant Cell Fraction\nper Treatment', fontsize=13)
            axes[1].set_ylim(0, max(treat_mal.values) * 1.2 if treat_mal.notna().any() else 100)
            for i, (t, v) in enumerate(treat_mal.items()):
                if not np.isnan(v):
                    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
            sns.despine(ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, 'Malignant classification\nnot available', ha='center', va='center',
                         transform=axes[1].transAxes, fontsize=13)
            axes[1].axis('off')

        # Panel 3: TME score summary by treatment
        tme_cols = [c for c in adata.obs.columns if c.endswith('_score') and c != 'cnv_score']
        if tme_cols and 'treatment' in adata.obs.columns:
            tme_means = adata.obs.groupby('treatment')[tme_cols].mean()
            tme_means = tme_means.reindex([t for t in TREATMENT_ORDER if t in tme_means.index])
            tme_means.columns = [c.replace('_score', '') for c in tme_means.columns]
            im = axes[2].imshow(tme_means.T.values, aspect='auto', cmap='RdBu_r')
            axes[2].set_xticks(range(len(tme_means)))
            axes[2].set_xticklabels(tme_means.index, fontsize=10, rotation=15)
            axes[2].set_yticks(range(len(tme_means.columns)))
            axes[2].set_yticklabels(tme_means.columns, fontsize=10)
            plt.colorbar(im, ax=axes[2], label='Mean Score', shrink=0.7)
            axes[2].set_title('TME Functional Scores\nby Treatment', fontsize=13)
        else:
            axes[2].text(0.5, 0.5, 'TME scoring skipped', ha='center', va='center',
                         transform=axes[2].transAxes, fontsize=13)
            axes[2].axis('off')

        # Panel 4: Cell type composition with spatial domain if available
        if 'spatial_domain' in adata.obs.columns and 'cell_type_auto' in adata.obs.columns:
            domain_ct = adata.obs.groupby('spatial_domain')['cell_type_auto'].value_counts(normalize=True).unstack(fill_value=0)
            top_cts = adata.obs['cell_type_auto'].value_counts().head(8).index
            domain_ct = domain_ct[[c for c in top_cts if c in domain_ct.columns]]
            domain_ct.plot(kind='bar', stacked=True, ax=axes[3], colormap='tab10', legend=True)
            axes[3].set_xlabel('Spatial Domain', fontsize=12)
            axes[3].set_ylabel('Cell Type Fraction', fontsize=12)
            axes[3].set_title('Cell Type Composition\nby Spatial Domain (BANKSY)', fontsize=13)
            axes[3].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=30, ha='right')
            sns.despine(ax=axes[3])
        elif 'cell_type_auto' in adata.obs.columns and 'treatment' in adata.obs.columns:
            ct_treat = adata.obs.groupby('treatment')['cell_type_auto'].value_counts(normalize=True).unstack(fill_value=0)
            top_cts = adata.obs['cell_type_auto'].value_counts().head(8).index
            ct_treat = ct_treat[[c for c in top_cts if c in ct_treat.columns]]
            ct_treat = ct_treat.reindex([t for t in TREATMENT_ORDER if t in ct_treat.index])
            ct_treat.plot(kind='bar', stacked=True, ax=axes[3], colormap='tab10', legend=True)
            axes[3].set_xlabel('Treatment', fontsize=12)
            axes[3].set_ylabel('Cell Type Fraction', fontsize=12)
            axes[3].set_title('Cell Type Composition\nby Treatment', fontsize=13)
            axes[3].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=15)
            sns.despine(ax=axes[3])
        else:
            axes[3].axis('off')

        plt.suptitle('Step 7 Integrated Summary: CNV · Spatial Niches · TME',
                     fontsize=20, y=1.01)
        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        # plt.savefig(os.path.join(dirs['summary'], 'fig_step7_integrated_summary.png'),
        # dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fig_step7_integrated_summary.png")
    except Exception as e:
        print(f"  WARNING: Summary figure failed: {e}")
        plt.close('all')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    collect(args.output_dir, {
        os.path.join('downstream_analysis', 'figures', '22_tme_scoring', 'fig_tme_gsea_heatmap'): 'suppl10_b',
        os.path.join('downstream_analysis', 'figures', '21_spatial_niches', 'fig_squidpy_nhood_enrichment'): 'suppl13_b',
    })

    print("\n" + "=" * 60)
    print("STEP 7 COMPLETE")
    print("=" * 60)
    print(f"Outputs in: {args.output_dir}/downstream_analysis/")
    print(f"  figures/20_cnv_malignant/        - CNV analysis figures")
    print(f"  figures/21_spatial_niches/        - Neighborhood enrichment, co-occurrence")
    print(f"  figures/22_tme_scoring/           - TME functional scores")
    print(f"  processed_data/cnv_malignant_classification.csv")
    print(f"  processed_data/nhood_enrichment_*.csv")
    print(f"  processed_data/tme_scores_per_celltype_treatment.csv")
    print(f"  processed_data/merged_step7_complete.h5ad")


if __name__ == '__main__':
    main()
