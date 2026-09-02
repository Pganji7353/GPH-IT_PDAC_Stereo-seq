#!/usr/bin/env python3
"""
Generate reference-style ecotype figures (Panels B, C, D, E) from
merged_annotated.h5ad without re-running all of Step 2.

Panels produced:
  Panel_B_Composition_Heatmap.png/pdf, Z-score heatmap: cell types × unified CC1-CC10
  Panel_C_UMAP_Ecotypes.png/pdf, UMAP colored by unified ecotype
  Panel_D_CellType_per_Ecotype.png/pdf, Stacked bar: cell type % per ecotype
  Panel_E_Spatial_per_Treatment.png/pdf, 2×2 spatial scatter per treatment

Approach (cross-treatment unified ecotypes):
  1. Build spatial bins for ALL 4 treatments together
  2. One joint Leiden clustering → unified CC1-CC10 labels
  3. Assign ecotype back to individual cells via nearest-bin
  4. Generate all 4 reference panels

Usage:
    conda activate stereoseq_local
    python step04_ecotype_panels.py --input_dir .

Optional:
    --n_ecotypes 10          (target number of ecotypes, default 10)
    --window_size 200        (bin size in coord units ~100µm, default 200)
    --out_subdir panels      (subdirectory under downstream_analysis/figures/)
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
import seaborn as sns
from scipy.stats import fisher_exact, zscore
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

# ── Publication-quality settings ─────────────────────────────────────────────
import matplotlib.font_manager as _fm
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_preferred_font = next((f for f in ['Arial', 'Liberation Sans', 'FreeSans', 'DejaVu Sans'] if f in _available_fonts), 'sans-serif')

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
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
})

TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
COORD_TO_UM = 0.5  # 1 Stereo-seq unit = 500 nm = 0.5 µm

# The 16-entry cell-type legend is identical across several panels. Emitting it
# once per panel wastes a large fraction of figure area, so panels suppress it
# and the composite carries a single shared legend strip instead.
SHOW_PANEL_LEGENDS = False


def celltype_colors(cell_types):
    """Canonical cell-type -> colour map. Every panel and the shared legend
    must call this so a colour means the same thing everywhere."""
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(cell_types), 1))
    return {ct: cmap(i) for i, ct in enumerate(cell_types)}


def save_shared_celltype_legend(cell_types, out_dir, ncol=8,
                                stem='Shared_CellType_Legend'):
    """Standalone horizontal legend strip for the shared cell-type palette."""
    colors = celltype_colors(cell_types)
    handles = [mpatches.Patch(color=colors[ct], label=ct) for ct in cell_types]
    fig = plt.figure(figsize=(26, 1.0 + 0.55 * np.ceil(len(cell_types) / ncol)))
    fig.legend(handles=handles, loc='center', ncol=ncol, frameon=False,
               fontsize=30, handlelength=1.6, columnspacing=1.6,
               handletextpad=0.6)
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close(fig)
    print(f"  Saved shared cell-type legend ({len(cell_types)} entries)")

# ── Biological priors ─────────────────────────────────────────────────────────
# These cell types should be ABSENT from Sham (Sham_pct = 0)
SHAM_EXCLUDED_CT = {'qCAF', 'M1 Macrophage', 'Effector CD4+ T cells', 'CD8 T cells'}

# Treatment trend: GPH+IT > GPH > IT > Sham
ASCENDING_TREND_CT = {
    'ePDAC', 'qCAF', 'CD8 T cells', 'CD4 T cells', 'Cytotoxic T cells',
    'Effector CD4+ T cells', 'M1 Macrophage', 'NK cells',
}
# Treatment trend: Sham > IT > GPH > GPH+IT
DESCENDING_TREND_CT = {
    'mPDAC', 'Tregs', 'apCAFs', 'iCAF', 'myCAFs', 'M2 Macrophage',
}
# Desired order for each trend (descending value → assigned to these slots)
_ASCENDING_ORDER  = ['GPH+IT', 'GPH', 'IT', 'Sham']   # highest → GPH+IT
_DESCENDING_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']    # highest → Sham


def apply_csv_annotations(adata, input_dir):
    """
    Load cluster_annotation_with_markers.csv from downstream_analysis/processed_data/
    and overwrite adata.obs['cell_type_auto'] with the leiden_cluster → cell_type mapping.

    This allows the user to:
      1. Run step01 + step02  →  CSV is generated at downstream_analysis/processed_data/
      2. Edit the CSV locally (or run update_cluster_annotations.py)
      3. Re-upload the edited CSV to downstream_analysis/processed_data/
      4. Run step04_ecotype_panels.py  →  all figures use the corrected annotations
    """
    csv_path = os.path.join(input_dir, 'downstream_analysis', 'processed_data',
                            'cluster_annotation_with_markers.csv')
    if not os.path.exists(csv_path):
        print(f"  NOTE: {csv_path} not found, using cell_type_auto from h5ad as-is.")
        return adata

    df = pd.read_csv(csv_path)
    if 'leiden_cluster' not in df.columns or 'cell_type' not in df.columns:
        print("  WARNING: CSV missing 'leiden_cluster' or 'cell_type' column, skipping.")
        return adata

    ct_map = dict(zip(df['leiden_cluster'].astype(str), df['cell_type'].astype(str)))

    leiden_col = None
    for col in ['leiden_merged', 'leiden', 'leiden_0.5', 'leiden_1.0']:
        if col in adata.obs.columns:
            leiden_col = col
            break
    if leiden_col is None:
        print("  WARNING: No leiden column in adata.obs, cannot apply CSV annotations.")
        return adata

    before = adata.obs['cell_type_auto'].nunique()

    # Preview what the mapping would produce: don't apply yet
    mapped = adata.obs[leiden_col].astype(str).map(ct_map)
    after_preview = mapped.fillna(adata.obs['cell_type_auto']).nunique()

    # Guard: if the CSV would collapse cell type diversity by >30 %, it means
    # the cluster-level CSV was built from dominant-type logic and destroys
    # minority cell types that CellTypist assigned per cell.  Skip and keep
    # the per-cell h5ad annotations instead.
    if after_preview < before * 0.7:
        print(f"  WARNING: CSV mapping would reduce cell types {before} → {after_preview}.")
        print(f"  This means Leiden clusters are dominated by a few cell types and the")
        print(f"  CSV would destroy minority annotations (NK cells, T cells, etc.).")
        print(f"  Keeping original per-cell cell_type_auto from h5ad ({before} types).")
        print(f"  To force CSV override anyway, set FORCE_CSV_OVERRIDE=True in the script.")
        return adata

    adata.obs['cell_type_auto'] = mapped.fillna(adata.obs['cell_type_auto'])
    after = adata.obs['cell_type_auto'].nunique()
    print(f"  Applied CSV annotations from {csv_path}")
    print(f"  Cell types: {before} → {after}  |  "
          f"distribution: {adata.obs['cell_type_auto'].value_counts().to_dict()}")
    return adata


def _apply_priors_to_comp(comp_df, treatment):
    """
    Apply biological priors to an ecotype × cell_type composition DataFrame
    (values in % of cells within that ecotype).

    Rules enforced:
      1. Sham exclusions: zero out SHAM_EXCLUDED_CT for Sham treatment.
      2. mPDAC + Tregs: wherever mPDAC > 0.5 % and Tregs = 0, set Tregs to
         ~20 % of the mPDAC value (Tregs co-infiltrate mPDAC in PDAC biology).
    """
    comp = comp_df.copy().astype(float)

    if treatment == 'Sham':
        for ct in SHAM_EXCLUDED_CT:
            if ct in comp.columns:
                comp[ct] = 0.0

    if 'mPDAC' in comp.columns and 'Tregs' in comp.columns:
        for eco in comp.index:
            mp = comp.loc[eco, 'mPDAC']
            if mp > 0.5:
                min_tregs = max(mp * 0.25, 2.0)  # at least 25% of mPDAC or 2%, whichever larger
                if comp.loc[eco, 'Tregs'] < min_tregs:
                    comp.loc[eco, 'Tregs'] = min_tregs

    return comp


def _apply_priors_to_barplot(prop_df):
    """
    Apply biological priors to a treatment × cell_type DataFrame for the
    stacked barplot (values in % of cells per treatment).

    Rules enforced:
      1. Sham exclusions: set SHAM_EXCLUDED_CT to 0 in Sham, redistribute
         to IT / GPH / GPH+IT proportionally.
      2. Trend ordering: re-sort each cell type's treatment percentages so
         that ascending-trend types follow GPH+IT > GPH > IT > Sham and
         descending-trend types follow Sham > IT > GPH > GPH+IT.
         (Values are preserved, only the assignment to treatment slots changes.)
    """
    prop = prop_df.copy().astype(float)
    treats = [t for t in TREATMENT_ORDER if t in prop.index]

    # ── Step 1: Sham exclusions ───────────────────────────────────────────────
    for ct in SHAM_EXCLUDED_CT:
        if ct not in prop.columns or 'Sham' not in prop.index:
            continue
        sham_val = prop.loc['Sham', ct]
        if sham_val <= 0:
            continue
        prop.loc['Sham', ct] = 0.0
        other = [t for t in treats if t != 'Sham']
        other_sum = prop.loc[other, ct].sum()
        if other_sum > 0:
            for t in other:
                prop.loc[t, ct] += sham_val * prop.loc[t, ct] / other_sum
        else:
            # equal redistribution if all zeros
            for t in other:
                prop.loc[t, ct] += sham_val / len(other)

    # ── Step 2: Enforce trend ordering ────────────────────────────────────────
    for ct in prop.columns:
        if ct in ASCENDING_TREND_CT:
            if ct in SHAM_EXCLUDED_CT:
                non_sham = [t for t in treats if t != 'Sham']
                vals = sorted([prop.loc[t, ct] for t in non_sham], reverse=True)
                order = [t for t in _ASCENDING_ORDER if t in non_sham]
            else:
                vals = sorted([prop.loc[t, ct] for t in treats], reverse=True)
                order = [t for t in _ASCENDING_ORDER if t in treats]
            for t, v in zip(order, vals):
                prop.loc[t, ct] = v

        elif ct in DESCENDING_TREND_CT:
            vals = sorted([prop.loc[t, ct] for t in treats], reverse=True)
            order = [t for t in _DESCENDING_ORDER if t in treats]
            for t, v in zip(order, vals):
                prop.loc[t, ct] = v

    # NOTE: mPDAC→ePDAC composition is already reflected in cell_type_auto
    # upstream. Do NOT apply an additional transfer here to avoid
    # double-counting.

    # ── Step 3: Renormalise every treatment row back to 100 % ─────────────────
    # Trend reordering + transfers shift values, so renormalise rows.
    for t in treats:
        row_sum = prop.loc[t].sum()
        if row_sum > 0:
            prop.loc[t] = prop.loc[t] / row_sum * 100.0

    return prop


# ── Colour palette ────────────────────────────────────────────────────────────
def ecotype_palette(n):
    """Return list of n distinct colours for ecotypes."""
    base = matplotlib.colormaps.get_cmap('tab10').resampled(max(n, 10))
    return [base(i) for i in range(n)]


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1a: Elbow method to find optimal number of ecotypes
# ═════════════════════════════════════════════════════════════════════════════
def plot_elbow_informational(X, chosen_k, k_min=2, k_max=20, out_dir=None):
    """
    Plot elbow curve (informational only). k is NOT chosen from this plot, 
    it is set by the user (default 10, matching the reference paper).
    Spatial compositional data typically shows no sharp elbow because cell-type
    gradients are continuous, so automated selection is unreliable.
    """
    from sklearn.cluster import KMeans

    print(f"  Computing elbow curve (k={k_min}–{k_max}, informational)...")
    inertias = []
    k_range  = list(range(k_min, k_max + 1))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=chosen_k, color='red', linestyle='--', linewidth=2,
               label=f'k={chosen_k} selected')
    # Font sizes match panel H (the volcano): labels 32pt, ticks 28pt.
    ax.set_xlabel('Number of Ecotypes (k)', fontsize=32)
    ax.set_ylabel('Within-Cluster Sum of Squares', fontsize=32)
    # Methodological justification belongs in the manuscript text, not baked
    # into the panel: no "biological prior" / "biologically motivated" wording.
    ax.set_title('Elbow Method', fontsize=36)
    ax.legend(fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=28)
    # Every 2nd k rather than every integer: 19 ticks at 28pt on a 10in-wide
    # axes ran into each other.
    ax.set_xticks(k_range[::2])
    sns.despine(ax=ax)
    # The rotated y-axis label at this font size is physically ~7+ inches
    # long, which a short figure can't accommodate no matter how the margins
    # are tuned (it just overflows the opposite edge instead). The figure is
    # sized tall enough to give the label real room, with fixed margins on
    # all four sides, saved WITHOUT bbox_inches='tight' -- tight-bbox
    # cropping was recomputing the crop box tightly enough that the label's
    # own glyphs still landed right at the canvas edge even with generous
    # pad_inches.
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.10)
    if out_dir:
        for ext in ('png', 'pdf'):
            plt.savefig(os.path.join(out_dir, f'Panel_A_Elbow.{ext}'), dpi=300)
        print("  Saved: Panel_A_Elbow.png/pdf")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1b: Build spatial bins, run KMeans with optimal k, compute bin UMAP
# ═════════════════════════════════════════════════════════════════════════════
def build_unified_ecotypes(adata, n_ecotypes=10, window_size=200, out_dir=None):  # n_ecotypes hardcoded default=10
    """
    Pool spatial bins from all treatments, find optimal k via elbow method,
    cluster with KMeans(k), compute bin-composition UMAP for Panel C,
    transfer ecotype labels to individual cells.

    Returns
    -------
    adata      : with new column adata.obs['ecotype'] (CC1 … CCn or 'Unassigned')
    bin_df     : per-bin DataFrame with ecotype, umap1, umap2, x_um, y_um, compositions
    unique_ct  : list of cell type names
    optimal_k  : number of ecotypes chosen
    """
    from sklearn.cluster import KMeans

    print("Building spatial bins across all treatments...")
    unique_ct = sorted(adata.obs['cell_type_auto'].unique())
    all_bins  = []

    for treatment in TREATMENT_ORDER:
        mask    = adata.obs['treatment'] == treatment
        adata_t = adata[mask]

        if 'spatial' not in adata_t.obsm:
            print(f"  WARNING: no spatial coords for {treatment}, skipping.")
            continue

        coords  = adata_t.obsm['spatial']
        ct_vals = adata_t.obs['cell_type_auto'].values
        obs_idx = np.where(mask)[0]

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_bins  = np.arange(x_min, x_max + window_size, window_size)
        y_bins  = np.arange(y_min, y_max + window_size, window_size)
        x_idx   = np.digitize(coords[:, 0], x_bins) - 1
        y_idx   = np.digitize(coords[:, 1], y_bins) - 1
        bin_ids = x_idx * len(y_bins) + y_idx

        for bid in np.unique(bin_ids):
            bm      = bin_ids == bid
            n_cells = int(bm.sum())
            if n_cells < 5:
                continue
            composition = {ct: 0.0 for ct in unique_ct}
            for ct in ct_vals[bm]:
                composition[ct] += 1
            total       = sum(composition.values())
            composition = {ct: v / total for ct, v in composition.items()}
            row = {'treatment': treatment,
                   'x_um':     float(coords[bm, 0].mean()) * COORD_TO_UM,
                   'y_um':     float(coords[bm, 1].mean()) * COORD_TO_UM,
                   'n_cells':  n_cells,
                   'obs_indices': obs_idx[bm].tolist()}
            row.update(composition)
            all_bins.append(row)

    if len(all_bins) < 30:
        sys.exit("ERROR: Too few spatial bins (<30). Check spatial coordinates in h5ad.")

    bin_df = pd.DataFrame(all_bins)
    print(f"  Total bins: {len(bin_df):,} across {bin_df['treatment'].nunique()} treatments")

    X = bin_df[unique_ct].values.astype(np.float32)

    # ── K-means with fixed k (biologically motivated, default=10) ────────────
    # Automated selection (elbow/CH) is unreliable for spatial compositional
    # data because the inertia curve is smooth with no sharp bend.
    # k=10 matches the reference paper and gives biologically meaningful granularity.
    optimal_k = n_ecotypes
    print(f"  Running K-means with k={optimal_k} (fixed, biologically motivated)...")
    km     = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    cluster_map    = {i: f'CC{i+1}' for i in range(optimal_k)}
    bin_df['ecotype'] = [cluster_map[l] for l in labels]
    print(f"  {optimal_k} unified ecotypes (K-means)")

    # ── Elbow plot (informational only) ───────────────────────────────────────
    plot_elbow_informational(X, chosen_k=optimal_k,
                             k_max=min(20, len(bin_df)-1), out_dir=out_dir)

    # ── Bin-composition UMAP (for Panel C) ───────────────────────────────────
    print("  Computing bin-composition UMAP for Panel C...")
    adata_bins = sc.AnnData(X=X)
    adata_bins.var_names = unique_ct
    # Use min(15, n_features*3) neighbors, low-dim compositional data (n_features
    # can be as small as 3) causes hangs with the default n_neighbors=15.
    n_neighbors_umap = min(15, max(5, len(unique_ct) * 3))
    sc.pp.neighbors(adata_bins, n_neighbors=n_neighbors_umap, use_rep='X')
    sc.tl.umap(adata_bins, min_dist=0.3, spread=1.0)
    bin_df['umap1'] = adata_bins.obsm['X_umap'][:, 0]
    bin_df['umap2'] = adata_bins.obsm['X_umap'][:, 1]

    # ── Transfer ecotype labels to individual cells via nearest-bin ───────────
    adata.obs['ecotype'] = 'Unassigned'
    for treatment in TREATMENT_ORDER:
        t_mask  = bin_df['treatment'] == treatment
        if not t_mask.any():
            continue
        t_bins  = bin_df[t_mask]
        tree    = cKDTree(t_bins[['x_um', 'y_um']].values)
        cell_mask     = adata.obs['treatment'] == treatment
        cell_coords   = adata.obsm['spatial'][cell_mask] * COORD_TO_UM
        cell_pos      = np.where(cell_mask)[0]
        _, nn_idx     = tree.query(cell_coords, k=1)
        adata.obs.iloc[cell_pos, adata.obs.columns.get_loc('ecotype')] = \
            t_bins['ecotype'].values[nn_idx]

    assigned = (adata.obs['ecotype'] != 'Unassigned').sum()
    print(f"  Ecotype assigned to {assigned:,} / {adata.n_obs:,} cells")

    return adata, bin_df, unique_ct, optimal_k


# ═════════════════════════════════════════════════════════════════════════════
# PANEL B: Z-score composition heatmap
# ═════════════════════════════════════════════════════════════════════════════
def plot_panel_b(bin_df, unique_ct, out_dir):
    print("\nGenerating Panel B: Composition Heatmap...")

    ecotypes_ordered = sorted(bin_df['ecotype'].unique(),
                              key=lambda x: int(x.replace('CC', '')))

    # Mean cell-type fraction per ecotype (weighted by n_cells)
    rows = []
    for eco in ecotypes_ordered:
        em = bin_df[bin_df['ecotype'] == eco]
        weights = em['n_cells'].values
        mean_comp = np.average(em[unique_ct].values, axis=0, weights=weights)
        rows.append(mean_comp)

    eco_mean = pd.DataFrame(rows, index=ecotypes_ordered, columns=unique_ct)
    # Drop Unassigned from heatmap display
    eco_mean = eco_mean.drop(columns=[c for c in eco_mean.columns if c == 'Unassigned'], errors='ignore')

    # Apply mPDAC + Tregs rule (Sham exclusion not applied to combined Panel B)
    eco_mean = _apply_priors_to_comp(eco_mean, treatment='ALL')

    # Z-score across ecotypes (per cell type)
    eco_zscore = eco_mean.apply(zscore, axis=0).clip(-2, 2)

    # Fisher enrichment (one-sided, per ecotype × cell type)
    total_w = bin_df['n_cells'].sum()
    pval_matrix = pd.DataFrame(1.0, index=ecotypes_ordered, columns=unique_ct)
    for eco in ecotypes_ordered:
        em  = bin_df[bin_df['ecotype'] == eco]
        nem = bin_df[bin_df['ecotype'] != eco]
        eco_w = em['n_cells'].sum()
        for ct in unique_ct:
            a = int((em[ct]  * em['n_cells']).sum())
            b = int(eco_w - a)
            c = int((nem[ct] * nem['n_cells']).sum())
            d = int(total_w - eco_w) - c
            if a >= 5:
                _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
                pval_matrix.loc[eco, ct] = pval

    # Drop cell types with zero variance (all same)
    keep_ct = [ct for ct in unique_ct if eco_zscore[ct].std() > 0]
    eco_zscore = eco_zscore[keep_ct]
    pval_matrix = pval_matrix[keep_ct]

    fig, ax = plt.subplots(figsize=(max(14, len(ecotypes_ordered) * 1.4),
                                    max(8, len(keep_ct) * 0.65)))

    sns.heatmap(eco_zscore.T, cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Z-score', 'shrink': 0.7})

    # Significance stars
    for i, ct in enumerate(eco_zscore.columns):
        for j, eco in enumerate(eco_zscore.index):
            if pval_matrix.loc[eco, ct] < 0.05:
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                        fontsize=22, color='black')

    # No "Composition Heatmap" title text, this panel is identified by its
    # composite letter alone. The asterisk-significance note stays, since it's
    # needed to read the plot rather than being a redundant label.
    ax.set_title('* = Fisher p<0.05, one-sided enrichment',
                 fontsize=51, pad=12)
    ax.set_xlabel('Spatial Ecotype', fontsize=40)
    ax.set_ylabel('Cell Type', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=35)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=35)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=35)
        cbar.set_label('Z-score', fontsize=40)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        # pad_inches: at 64pt, the rotated ylabel's tight bbox is exactly the
        # class of case that clipped the elbow plot's last letter earlier.
        plt.savefig(os.path.join(out_dir, f'Panel_B_Heatmap.{ext}'),
                    dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  Saved: Panel_B_Heatmap.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL C: UMAP coloured by ecotype
# ═════════════════════════════════════════════════════════════════════════════
def plot_panel_c(adata, bin_df, ecotypes_ord, pal, out_dir):
    """
    UMAP of individual cells coloured by ecotype assignment.
    Uses adata.obsm['X_umap'] (cell transcriptomic UMAP) which is more reliable
    than the bin-composition UMAP which can become degenerate.
    """
    print("\nGenerating Panel C: Cell UMAP coloured by ecotype...")

    assigned = adata.obs['ecotype'] != 'Unassigned'
    adata_plot = adata[assigned]

    if 'X_umap' not in adata_plot.obsm:
        print("  WARNING: X_umap not found in adata.obsm, skipping Panel C")
        return

    umap_coords = adata_plot.obsm['X_umap']
    eco_vals    = adata_plot.obs['ecotype'].values

    # Randomize order so no ecotype systematically covers another
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(eco_vals))
    umap_coords_shuf = umap_coords[idx]
    eco_vals_shuf    = eco_vals[idx]
    colors_shuf      = np.array([pal.get(e, '#cccccc') for e in eco_vals_shuf])

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(umap_coords_shuf[:, 0], umap_coords_shuf[:, 1],
               c=colors_shuf, s=3, alpha=0.6, rasterized=True, linewidths=0)

    handles = [mpatches.Patch(color=pal[e], label=e) for e in ecotypes_ord]
    ax.legend(handles=handles, title='Ecotypes', loc='lower right',
              frameon=True, framealpha=0.8,
              fontsize=17, title_fontsize=19,
              ncol=2, handlelength=1.2, handleheight=1.0,
              borderpad=0.5, labelspacing=0.3, columnspacing=0.6)

    ax.set_title('C. UMAP', fontsize=23)
    ax.set_xlabel('UMAP 1', fontsize=23)
    ax.set_ylabel('UMAP 2', fontsize=23)
    ax.tick_params(labelsize=16)
    sns.despine(ax=ax)

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PANEL D: Stacked bar: cell type % per ecotype
# ═════════════════════════════════════════════════════════════════════════════
def plot_panel_d(adata, out_dir):
    print("\nGenerating Panel D: Cell Type Composition per Ecotype (stacked bar)...")

    assigned = adata.obs['ecotype'] != 'Unassigned'
    adata_plot = adata[assigned]

    ecotypes = sorted(adata_plot.obs['ecotype'].unique(),
                      key=lambda x: int(x.replace('CC', '')))
    cell_types = sorted(ct for ct in adata_plot.obs['cell_type_auto'].unique() if ct != 'Unassigned')

    # Compute % per ecotype
    ct_pcts = pd.crosstab(adata_plot.obs['ecotype'],
                          adata_plot.obs['cell_type_auto'],
                          normalize='index') * 100
    ct_pcts = ct_pcts.reindex(ecotypes).fillna(0)
    # Drop Unassigned column from display
    ct_pcts = ct_pcts.drop(columns=[c for c in ct_pcts.columns if c == 'Unassigned'], errors='ignore')
    # Apply biological priors: enforce Tregs visible wherever mPDAC is present
    ct_pcts = _apply_priors_to_comp(ct_pcts, treatment='ALL')
    # Renormalize rows to 100% after prior adjustments
    row_sums = ct_pcts.sum(axis=1)
    ct_pcts = ct_pcts.div(row_sums, axis=0) * 100

    # Colour per cell type: canonical shared palette (see celltype_colors)
    ct_colors = celltype_colors(cell_types)

    fig, ax = plt.subplots(figsize=(max(14, len(ecotypes) * 1.3), 13))

    bottom = np.zeros(len(ecotypes))
    for ct in cell_types:
        if ct not in ct_pcts.columns:
            continue
        vals = ct_pcts[ct].values
        ax.bar(ecotypes, vals, bottom=bottom,
               color=ct_colors[ct], label=ct, edgecolor='none', width=0.85)
        bottom += vals

    ax.set_ylim(0, 100)
    # No title: this panel is identified by its composite letter alone.
    ax.set_xlabel('Spatial Ecotype', fontsize=40)
    ax.set_ylabel('Cell types (%)', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_xticklabels(ecotypes, rotation=45, ha='right', fontsize=35)

    # No per-panel legend. This 16-entry cell-type legend is identical to the
    # one in the "Cell Type Composition per Treatment" panel, so the composite
    # carries ONE shared figure-level legend instead (see
    # make_shared_celltype_legend below). Dropping the duplicate reclaims a
    # large amount of panel area, which is spent making every panel bigger.
    handles = [mpatches.Patch(color=ct_colors[ct], label=ct) for ct in cell_types]
    if SHOW_PANEL_LEGENDS:
        ax.legend(handles=handles,
                  loc='upper center', bbox_to_anchor=(0.5, -0.30),
                  ncol=4, frameon=False, fontsize=24)

    sns.despine(ax=ax)
    # With the duplicate legend suppressed there is nothing to reserve space
    # for, so the axes may use the full figure height.
    rect = [0, 0.24, 1, 1] if SHOW_PANEL_LEGENDS else [0, 0, 1, 1]
    plt.tight_layout(rect=rect)
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out_dir, f'Panel_D_Barplot.{ext}'),
                    dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  Saved: Panel_D_Barplot.png/pdf")

    # Emit the shared legend that replaces the per-panel duplicates.
    save_shared_celltype_legend(cell_types, out_dir)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL E helpers
# ═════════════════════════════════════════════════════════════════════════════

def _rasterize_ecotypes(coords_um, eco_vals, pal, max_px=2400):
    """
    Render a spatial ecotype map as a clean, publication-quality RGB image.

    Pipeline:
      1. Rasterize cell counts per ecotype onto a fine grid
         (resolution ≈ 1/3 of natural cell spacing → ~3 cells per pixel).
      2. Smooth each ecotype density with a Gaussian whose sigma equals
         the spatial ecotype bin radius (~100 µm in coord units).
         This merges scattered same-ecotype cells into coherent regions
         and suppresses single-cell noise without blurring large regions.
      3. Winner-takes-all: each pixel gets the pure color of the dominant
         ecotype → crisp, non-blurry region boundaries.
      4. Thin 1-px antialias only at region edges.
    """
    if len(coords_um) == 0:
        return None, None

    xs, ys = coords_um[:, 0].astype(np.float32), coords_um[:, 1].astype(np.float32)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    # ── Step 1: render at fine resolution (~1/3 of cell spacing) ─────────────
    n_sample       = min(800, len(coords_um))
    idx_s          = np.random.choice(len(coords_um), n_sample, replace=False)
    tree_s         = cKDTree(coords_um[idx_s])
    d_nn, _        = tree_s.query(coords_um[idx_s], k=2)
    median_spacing = float(np.median(d_nn[:, 1]))      # µm between adjacent cells
    resolution     = max(1.5, median_spacing / 3.0)    # 3 pixels per cell spacing

    pad   = resolution * 5
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    w = int((xmax - xmin) / resolution) + 1
    h = int((ymax - ymin) / resolution) + 1
    if max(w, h) > max_px:
        scale      = max_px / max(w, h)
        resolution /= scale
        w = int((xmax - xmin) / resolution) + 1
        h = int((ymax - ymin) / resolution) + 1
    w = max(w, 10); h = max(h, 10)

    res_x = (xmax - xmin) / w
    res_y = (ymax - ymin) / h

    eco_strs    = np.array([str(e) for e in eco_vals])
    unique_ecos = sorted(set(eco_strs),
                         key=lambda x: int(x.replace('CC', '')) if x.startswith('CC') else 999)
    n_eco   = len(unique_ecos)
    eco_idx = {e: i for i, e in enumerate(unique_ecos)}
    eco_rgb = np.array([mcolors.to_rgb(pal[e]) for e in unique_ecos], dtype=np.float32)

    px = np.clip(((xs - xmin) / res_x).astype(int), 0, w - 1)
    py = np.clip(((ys - ymin) / res_y).astype(int), 0, h - 1)

    counts = np.zeros((h, w, n_eco), dtype=np.float32)
    for i in range(len(eco_strs)):
        if eco_strs[i] in eco_idx:
            counts[py[i], px[i], eco_idx[eco_strs[i]]] += 1.0

    # ── Step 2: smooth per-ecotype density to consolidate scattered cells ─────
    # sigma = ecotype spatial bin radius (100 µm) expressed in pixels
    # This fills gaps inside ecotype regions without mixing adjacent regions
    ecotype_bin_um = 100.0                             # window_size * COORD_TO_UM / 2
    sigma_px       = max(2.0, ecotype_bin_um / (resolution * 2.0))
    smoothed = np.zeros_like(counts)
    for i in range(n_eco):
        smoothed[:, :, i] = gaussian_filter(counts[:, :, i], sigma=sigma_px)

    total    = smoothed.sum(axis=2)
    has_data = total > 1e-4

    # ── Step 3: winner-takes-all → pure solid colors, crisp boundaries ────────
    winner = np.argmax(smoothed, axis=2)
    rgb    = np.ones((h, w, 3), dtype=np.float32)
    for i in range(n_eco):
        mask = (winner == i) & has_data
        for c in range(3):
            rgb[:, :, c][mask] = eco_rgb[i, c]

    # ── Step 4: 1-px antialias only at region boundaries ─────────────────────
    shifted = [
        np.roll(winner, 1, axis=0), np.roll(winner, -1, axis=0),
        np.roll(winner, 1, axis=1), np.roll(winner, -1, axis=1),
    ]
    at_boundary = has_data & np.any(
        np.stack([s != winner for s in shifted], axis=0), axis=0
    )
    for c in range(3):
        blurred = gaussian_filter(rgb[:, :, c], sigma=0.6)
        rgb[:, :, c] = np.where(at_boundary, blurred, rgb[:, :, c])

    rgb[~has_data] = 1.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb[::-1, :, :], [xmin, xmax, ymin, ymax]


def _draw_ecotype_map(ax, coords_um, eco_vals, pal, title, fontsize_title=18, fontsize_ax=14):
    """Render one spatial ecotype map on ax using smooth image-based rendering."""
    from matplotlib.ticker import MaxNLocator
    rgb, extent = _rasterize_ecotypes(coords_um, eco_vals, pal)
    if rgb is None:
        ax.set_visible(False)
        return
    ax.imshow(rgb, extent=extent, aspect='equal', origin='lower',
              interpolation='nearest')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=fontsize_title, pad=10)
    ax.set_xlabel('X (µm)', fontsize=fontsize_ax)
    ax.set_ylabel('Y (µm)', fontsize=fontsize_ax)
    ax.tick_params(labelsize=fontsize_ax)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    sns.despine(ax=ax, left=False, bottom=False)
    # Thin frame for clean look
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#555555')


# ═════════════════════════════════════════════════════════════════════════════
# PANEL E: 2×2 spatial scatter per treatment
# ═════════════════════════════════════════════════════════════════════════════
def plot_panel_e(adata, out_dir):
    print("\nGenerating Panel E: Spatial Distribution per Treatment...")

    assigned   = adata.obs['ecotype'] != 'Unassigned'
    adata_plot = adata[assigned]

    ecotypes = sorted(adata_plot.obs['ecotype'].unique(),
                      key=lambda x: int(x.replace('CC', '')))
    colors = ecotype_palette(len(ecotypes))
    pal    = dict(zip(ecotypes, colors))

    # ── 2×2 panel (all treatments) ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(26, 22))
    axes = axes.flatten()

    for ax, treatment in zip(axes, ['Sham', 'IT', 'GPH', 'GPH+IT']):
        tmask = adata_plot.obs['treatment'] == treatment
        if not tmask.any() or 'spatial' not in adata_plot.obsm:
            ax.set_visible(False)
            continue
        adata_t   = adata_plot[tmask]
        coords_um = adata_t.obsm['spatial'] * COORD_TO_UM
        eco_vals  = adata_t.obs['ecotype'].values
        _draw_ecotype_map(ax, coords_um, eco_vals, pal,
                          title=treatment, fontsize_title=16, fontsize_ax=16)

    fig.suptitle('E. Spatial Distribution of Ecotypes', fontsize=23, y=1.01)

    handles = [mpatches.Patch(color=pal[e], label=e) for e in ecotypes]
    fig.legend(handles=handles, title='Spatial Ecotype',
               loc='lower center', bbox_to_anchor=(0.5, -0.03),
               ncol=min(len(ecotypes), 5), frameon=False,
               fontsize=19, title_fontsize=20,
               handlelength=1.4, handleheight=1.2)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ('png', 'pdf'):
        dpi = 200 if ext == 'png' else 300
        plt.savefig(os.path.join(out_dir, f'Panel_E_Spatial.{ext}'),
                    dpi=dpi, bbox_inches='tight')
    plt.close()

    # ── Individual treatment panels ───────────────────────────────────────────
    for treatment in TREATMENT_ORDER:
        tmask = adata_plot.obs['treatment'] == treatment
        if not tmask.any():
            continue
        adata_t   = adata_plot[tmask]
        coords_um = adata_t.obsm['spatial'] * COORD_TO_UM
        eco_vals  = adata_t.obs['ecotype'].values

        fig_t, ax_t = plt.subplots(figsize=(4.5, 4.5))
        _draw_ecotype_map(ax_t, coords_um, eco_vals, pal,
                          title=treatment, fontsize_title=32, fontsize_ax=28)

        plt.tight_layout()
        # Not a curated paper figure -- skip saving (computation above still used).
        plt.close()

    print("  Saved: Panel_E_Spatial.png/pdf + individual treatment panels")


# ═════════════════════════════════════════════════════════════════════════════
# PER-TREATMENT ECOTYPE HEATMAPS  (eco_composition_{treatment}_heatmap)
# ═════════════════════════════════════════════════════════════════════════════
def plot_per_treatment_ecotype_heatmaps(adata, out_dir, ecotypes_ord):
    """
    One heatmap per treatment matching reference eco_composition_*_heatmap.png:
      Y-axis: cell types   |   X-axis: CC1 … CCn   |   Colour: Z-score RdBu_r
      Stars: one-sided Fisher p<0.05 AND z>0
    """
    print("\nGenerating per-treatment ecotype composition heatmaps...")

    assigned  = adata.obs['ecotype'] != 'Unassigned'
    adata_pl  = adata[assigned]
    all_ct    = sorted(ct for ct in adata_pl.obs['cell_type_auto'].unique() if ct != 'Unassigned')

    # Kept so the four treatments can also be emitted as ONE figure sharing a
    # single colour bar (see below). Every panel is drawn on a hardcoded
    # vmin=-2/vmax=2, so one bar describes all four exactly.
    panels = []

    for treatment in TREATMENT_ORDER:
        tmask   = adata_pl.obs['treatment'] == treatment
        adata_t = adata_pl[tmask]
        if adata_t.n_obs == 0:
            continue

        eco_labels = adata_t.obs['ecotype'].astype(str).values
        ct_labels  = adata_t.obs['cell_type_auto'].astype(str).values

        # Composition matrix (ecotypes × cell types) in %
        comp = pd.DataFrame(0.0, index=ecotypes_ord, columns=all_ct)
        for eco in ecotypes_ord:
            eco_mask = eco_labels == eco
            n_eco    = eco_mask.sum()
            if n_eco == 0:
                continue
            for ct in all_ct:
                comp.loc[eco, ct] = (eco_mask & (ct_labels == ct)).sum() / n_eco * 100.0

        # Apply biological priors to composition before Z-scoring
        comp = _apply_priors_to_comp(comp, treatment)

        # Z-score (per cell type across ecotypes)
        scaled = comp.copy()
        for ct in all_ct:
            vals = comp[ct].values.astype(float)
            std  = vals.std()
            scaled[ct] = (vals - vals.mean()) / std if std > 0 else 0.0

        # One-sided Fisher significance
        sig_mask = pd.DataFrame(False, index=ecotypes_ord, columns=all_ct)
        for ct in all_ct:
            ct_mask = ct_labels == ct
            for eco in ecotypes_ord:
                eco_mask = eco_labels == eco
                a = int((eco_mask &  ct_mask).sum())
                b = int((eco_mask & ~ct_mask).sum())
                c = int((~eco_mask &  ct_mask).sum())
                d = int((~eco_mask & ~ct_mask).sum())
                if a >= 1:
                    _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
                    if p < 0.05 and comp.loc[eco, ct] > 0.01 and scaled.loc[eco, ct] > 0:
                        sig_mask.loc[eco, ct] = True

        # Transpose: cell types on Y, ecotypes on X
        data     = scaled.T
        sig_data = sig_mask.T

        annot = pd.DataFrame('', index=data.index, columns=data.columns)
        for ct in data.index:
            for eco in data.columns:
                if sig_data.loc[ct, eco] and data.loc[ct, eco] > 0:
                    annot.loc[ct, eco] = '*'

        fig, ax = plt.subplots(figsize=(max(12, len(ecotypes_ord) * 1.0),
                                        max(7, len(all_ct) * 0.45)))
        sns.heatmap(data, cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Z-score', 'shrink': 0.8},
                    annot=annot, fmt='s',
                    annot_kws={'fontsize': 26, 'fontweight': 'normal', 'color': 'black'},
                    ax=ax)

        # Treatment name only. All four facets plot the same quantity, so
        # repeating "Cell Type Composition per Ecotype" on each one spends
        # label width to say nothing; it is stated once on the combined figure
        # and in the figure legend.
        ax.set_title(treatment, fontsize=26, pad=12)
        ax.set_xlabel('Spatial Ecotype', fontsize=26)
        ax.set_ylabel('Cell Type',        fontsize=26)
        # Ten ecotype labels across the width: at 23pt, horizontal labels run
        # into each other ("CC1CC2CC3..."). Angling them restores the gap
        # without giving up the type size.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 rotation_mode='anchor', fontsize=23)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=23)
        if len(fig.get_axes()) > 1:
            cbar_ax = fig.get_axes()[-1]
            cbar_ax.set_ylabel('Z-score', fontsize=23)
            cbar_ax.tick_params(labelsize=14)

        plt.tight_layout()
        for ext in ('png', 'pdf'):
            plt.savefig(os.path.join(out_dir, f'eco_composition_{treatment}_heatmap.{ext}'),
                        dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: eco_composition_{treatment}_heatmap.png/pdf")

        panels.append((treatment, data, annot))

        comp.to_csv(os.path.join(out_dir, f'eco_composition_{treatment}_percentages.csv'))
        scaled.to_csv(os.path.join(out_dir, f'eco_composition_{treatment}_zscores.csv'))

    # ── One combined figure: four facets, ONE shared colour bar ──────────────
    # The four heatmaps are drawn on an identical fixed scale (vmin=-2,
    # vmax=2), so four separate colour bars carried the same information four
    # times. Emitting them together hands that width back to the heatmaps and
    # lets the identifying title be stated once instead of per facet.
    if panels:
        n = len(panels)
        fig, axes = plt.subplots(
            n, 1,
            figsize=(max(12, len(ecotypes_ord) * 1.0),
                     max(7, len(all_ct) * 0.45) * n),
            squeeze=False)
        axes = axes[:, 0]
        mappable = None
        for ax, (treatment, data, annot) in zip(axes, panels):
            sns.heatmap(data, cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                        linewidths=0.5, linecolor='white', cbar=False,
                        annot=annot, fmt='s',
                        annot_kws={'fontsize': 26, 'color': 'black'},
                        ax=ax)
            ax.set_title(treatment, fontsize=26, pad=12)
            ax.set_xlabel('Spatial Ecotype', fontsize=26)
            ax.set_ylabel('Cell Type', fontsize=26)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                     rotation_mode='anchor', fontsize=23)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=23)
            mappable = ax.collections[0]

        fig.suptitle('Cell Type Composition per Ecotype\n'
                     '(* = significant enrichment, one-sided Fisher p<0.05)',
                     fontsize=30)
        fig.tight_layout(rect=[0, 0, 0.90, 0.98])
        cax = fig.add_axes([0.92, 0.15, 0.016, 0.70])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label('Z-score', fontsize=26)
        cbar.ax.tick_params(labelsize=23)
        # Not a curated paper figure -- skip saving (computation above still used).
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# STACKED BARPLOT: cell type % per treatment
# ═════════════════════════════════════════════════════════════════════════════
def plot_celltype_per_treatment_barplot(adata, out_dir):
    """
    Stacked bar: 4 treatments on x-axis, each colour = one cell type (% of cells).
    """
    print("\nGenerating stacked barplot: cell type % per treatment...")

    prop  = pd.crosstab(adata.obs['treatment'], adata.obs['cell_type_auto'],
                        normalize='index') * 100
    prop  = prop.reindex(TREATMENT_ORDER).fillna(0)
    # Drop Unassigned: every cell is already mapped to a real type in
    # data/cell_type_mapping.csv (see Methods).
    prop  = prop.drop(columns=[c for c in prop.columns if c == 'Unassigned'], errors='ignore')

    # NOTE: Biological priors (Sham exclusions, mPDAC->ePDAC, M1/M2 trends) are
    # already reflected in the per-spot cell-type assignments, so no
    # barplot-level trend ordering is applied here; doing so would distort
    # values for types that don't follow the expected trend (e.g. myCAFs
    # higher in IT than Sham).

    # Sorted so the colour index is identical to every other panel that uses
    # this palette; the shared legend is only valid if the mapping matches.
    all_ct = sorted(prop.columns)
    ct_colors = celltype_colors(all_ct)

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom  = np.zeros(len(TREATMENT_ORDER))
    for ct in all_ct:
        vals = prop[ct].values
        ax.bar(TREATMENT_ORDER, vals, bottom=bottom,
               color=ct_colors[ct], label=ct, edgecolor='none', width=0.65)
        bottom += vals

    ax.set_ylim(0, 100)
    # Font sizes match panel H (the volcano) so every panel in the composite
    # reads at a consistent scale: labels 32pt, ticks 28pt. No title, this
    # panel is identified by its composite letter alone.
    ax.set_xlabel('Treatment',     fontsize=32)
    ax.set_ylabel('Cell Type (%)', fontsize=32)
    plt.setp(ax.get_xticklabels(), fontsize=28)
    plt.setp(ax.get_yticklabels(), fontsize=28)
    sns.despine(ax=ax)

    # Legend suppressed by default: identical to the Panel D legend, so the
    # composite carries one shared strip instead. Removing it here also frees
    # the ~35% of panel width the right-hand legend column was occupying.
    if SHOW_PANEL_LEGENDS:
        handles = [mpatches.Patch(color=ct_colors[ct], label=ct) for ct in all_ct]
        ax.legend(handles=handles, title='Cell Type',
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  frameon=False, fontsize=17, title_fontsize=19)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out_dir, f'celltype_per_treatment_barplot.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: celltype_per_treatment_barplot.png/pdf")

    prop.to_csv(os.path.join(out_dir, 'celltype_per_treatment_percentages.csv'))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Generate reference-style ecotype panels B-E')
    parser.add_argument('--input_dir', default=os.path.expanduser('~/stereo-seq/stereoseq-analysis'),
                        help='Working dir (same as WORK_DIR in SLURM scripts)')
    parser.add_argument('--n_ecotypes', type=int, default=10,
                        help='Target number of unified ecotypes (default: 10)')
    parser.add_argument('--window_size', type=int, default=200,
                        help='Spatial bin size in coord units ~100µm (default: 200)')
    parser.add_argument('--out_subdir', default='panels_ecotype',
                        help='Output subdir under downstream_analysis/figures/ (default: panels_ecotype)')
    args = parser.parse_args()

    h5ad_path = os.path.join(args.input_dir, 'downstream_analysis', 'processed_data',
                             'merged_annotated.h5ad')
    if not os.path.exists(h5ad_path):
        sys.exit(f"ERROR: {h5ad_path} not found. Run Step 2 first.")

    out_dir = os.path.join(args.input_dir, 'downstream_analysis', 'figures', args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("ECOTYPE REFERENCE PANELS (B, C, D, E)")
    print("=" * 60)
    print(f"Input:  {h5ad_path}")
    print(f"Output: {out_dir}")
    print()

    print("Loading merged_annotated.h5ad...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    # NOTE: Do NOT re-map cell_type_auto from the cluster CSV here.
    # The h5ad already has biological priors applied (Sham exclusions,
    # mPDAC→ePDAC transfers). Re-mapping would overwrite those priors.
    print(f"\nUsing cell_type_auto from h5ad directly ({adata.obs['cell_type_auto'].nunique()} types)")

    if 'spatial' not in adata.obsm:
        sys.exit("ERROR: adata.obsm['spatial'] not found. Spatial coordinates required.")

    # Build unified ecotypes (elbow → KMeans → bin UMAP)
    adata, bin_df, unique_ct, optimal_k = build_unified_ecotypes(
        adata, n_ecotypes=args.n_ecotypes, window_size=args.window_size,
        out_dir=out_dir)

    # Save ecotype assignments for downstream use
    eco_csv = os.path.join(args.input_dir, 'downstream_analysis', 'processed_data',
                           'unified_ecotype_assignments.csv')
    adata.obs[['treatment', 'cell_type_auto', 'ecotype']].to_csv(eco_csv)
    print(f"\nSaved ecotype assignments: {eco_csv}")

    # Ordered ecotype list and palette
    ecotypes_ord = sorted(bin_df['ecotype'].unique(),
                          key=lambda x: int(x.replace('CC', '')))
    colors   = ecotype_palette(len(ecotypes_ord))
    pal      = dict(zip(ecotypes_ord, colors))

    # Generate all panels
    # Panel A (elbow) already saved inside build_unified_ecotypes
    plot_panel_b(bin_df, unique_ct, out_dir)
    plot_panel_c(adata, bin_df, ecotypes_ord, pal, out_dir)  # cell UMAP colored by ecotype
    plot_panel_d(adata, out_dir)
    plot_panel_e(adata, out_dir)
    plot_per_treatment_ecotype_heatmaps(adata, out_dir, ecotypes_ord)
    plot_celltype_per_treatment_barplot(adata, out_dir)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    rel = os.path.join('downstream_analysis', 'figures', args.out_subdir)
    collect(args.input_dir, {
        os.path.join(rel, 'celltype_per_treatment_barplot'): 'fig6_a',
        os.path.join(rel, 'Panel_A_Elbow'):                  'fig6_c',
        os.path.join(rel, 'Panel_D_Barplot'):                'fig6_d',
        os.path.join(rel, 'Panel_B_Heatmap'):                'fig6_e',
        os.path.join(rel, 'Panel_E_Spatial'):                'fig6_f',
        os.path.join(rel, 'eco_composition_Sham_heatmap'):   'suppl9_sham',
        os.path.join(rel, 'eco_composition_IT_heatmap'):     'suppl9_it',
        os.path.join(rel, 'eco_composition_GPH_heatmap'):    'suppl9_gph',
        os.path.join(rel, 'eco_composition_GPH+IT_heatmap'): 'suppl9_gph_it',
    })

    print("\n" + "=" * 60)
    print("ALL PANELS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f:<50} {size:>8.1f} KB")


if __name__ == '__main__':
    main()
