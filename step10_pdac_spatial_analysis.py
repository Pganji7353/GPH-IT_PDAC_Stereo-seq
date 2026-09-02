#!/usr/bin/env python3
"""
Post-9: PDAC-specific spatial analyses (manuscript-grade additions).

Outputs (under <input_dir>/downstream_analysis/figures/post9_pdac_spatial/):
  fig01_distance_to_tumor_gradient.png, radial mean expression of immune/CAF/hypoxia
  fig02_immune_exclusion_classification.png, cold vs hot mPDAC nests per treatment
  fig03_perivascular_niche.png, composition of 50 µm shell around endothelium
  fig04_tls_signature_spatial.png, TLS score spatial maps + counts per treatment
  fig05_hypoxia_core_map.png, Buffa hypoxia score spatial + correlation w/ CAFs
  fig06_neighborhood_zscore_matrix.png, squidpy nhood enrichment per treatment
  fig07_mhc_i_loss_map.png, H2-K1/D1/B2m antigen presentation spatial

Mouse data, Arial 10pt, per-plot intelligent sizing.
"""
from __future__ import annotations
import argparse, warnings, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import cKDTree, ConvexHull
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from matplotlib.lines import Line2D
import scipy.sparse as sp

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica',
                        'Nimbus Sans', 'FreeSans', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

TREATMENTS = ['Sham', 'IT', 'GPH', 'GPH+IT']

GENES = {
    'CD8_T':       ['Cd8a', 'Cd8b1', 'Gzmb', 'Prf1', 'Ifng', 'Nkg7'],
    'CAF_active':  ['Acta2', 'Fap', 'Col1a1', 'Pdgfra', 'Pdpn'],
    'Hypoxia':     ['Hif1a', 'Vegfa', 'Slc2a1', 'Ldha', 'Ca9', 'Pgk1', 'Eno1', 'Adm'],
    'TLS':         ['Cxcl13', 'Ccl19', 'Ccl21a', 'Cxcr5', 'Ccr7', 'Ms4a1',
                    'Cd3e', 'Cd4', 'Lamp3', 'Sell', 'Lta', 'Ltb', 'Cd79a'],
    'MHC_I':       ['H2-K1', 'H2-D1', 'B2m', 'Tap1', 'Tap2', 'Nlrc5'],
    'Perivasc_immunosuppr': ['Cxcl12', 'Foxp3', 'Arg1', 'Ly6g', 'Ly6c1'],
    'Endothelial': ['Pecam1', 'Cdh5', 'Vwf', 'Kdr'],
}


def _present(adata, genes):
    return [g for g in genes if g in adata.var_names]


def _coords(adata):
    if 'spatial' in adata.obsm:
        return np.asarray(adata.obsm['spatial'])
    for k in ('X_spatial', 'spatial_um'):
        if k in adata.obsm: return np.asarray(adata.obsm[k])
    raise KeyError("No spatial coords found in adata.obsm")


def _score(adata, name, genes):
    g = _present(adata, genes)
    if not g: return None
    sc.tl.score_genes(adata, g, score_name=f'_s_{name}', use_raw=False)
    return f'_s_{name}'


# ============================================================
# 1. Distance-to-tumor gradient
# ============================================================
def fig_distance_gradient(adata, out_dir, ct_key, treatment_key):
    coords = _coords(adata)
    is_tumor = adata.obs[ct_key].astype(str).str.contains('PDAC|Malignant|Tumor', case=False, regex=True).values
    if is_tumor.sum() < 50:
        print("  fig01: too few tumor cells"); return
    tree = cKDTree(coords[is_tumor])
    dist, _ = tree.query(coords, k=1)
    adata.obs['_dist_tumor'] = dist
    bins = [0, 50, 100, 200, 400, np.inf]
    labels = ['0-50', '50-100', '100-200', '200-400', '>400']
    adata.obs['_dist_bin'] = pd.cut(dist, bins=bins, labels=labels)

    panels = {'CD8 T': GENES['CD8_T'], 'CAF': GENES['CAF_active'], 'Hypoxia': GENES['Hypoxia']}
    score_keys = {n: _score(adata, n, g) for n, g in panels.items()}
    score_keys = {k: v for k, v in score_keys.items() if v}
    if not score_keys: return

    fig, axes = plt.subplots(1, len(score_keys), figsize=(5.5 * len(score_keys), 5.0), sharex=True)
    if len(score_keys) == 1: axes = [axes]
    palette = {t: c for t, c in zip(TREATMENTS, ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])}
    for ax, (name, key) in zip(axes, score_keys.items()):
        for tr in TREATMENTS:
            sub = adata.obs[adata.obs[treatment_key] == tr] if treatment_key in adata.obs else adata.obs
            if sub.empty: continue
            mean_per_bin = sub.groupby('_dist_bin', observed=True)[key].mean().reindex(labels)
            ax.plot(labels, mean_per_bin.values, '-o', label=tr, color=palette.get(tr, 'gray'), lw=2.5,
                    markersize=8)
        ax.set_title(f'{name}', fontsize=24)
        ax.set_xlabel('Distance to tumor (µm)', fontsize=22)
        ax.set_ylabel('Mean module score', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
    # Placed outside the axes: the default 'best' location lands the legend on
    # top of the dashed zero reference line in the Hypoxia panel.
    leg = axes[-1].legend(title='Treatment', frameon=False, fontsize=20,
                          loc='center left', bbox_to_anchor=(1.02, 0.5))
    leg.get_title().set_fontsize(21)
    fig.suptitle('Radial expression gradients from PDAC tumor', fontsize=26)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig01_distance_to_tumor_gradient.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig01_distance_to_tumor_gradient.pdf', bbox_inches='tight'); plt.close(fig)
    for k in list(score_keys.values()) + ['_dist_tumor', '_dist_bin']:
        if k in adata.obs: del adata.obs[k]


# ============================================================
# 2. Immune exclusion classification
# ============================================================
def fig_immune_exclusion(adata, out_dir, ct_key, treatment_key, sample_key='sample'):
    coords = _coords(adata)
    is_tumor = adata.obs[ct_key].astype(str).str.contains('PDAC|Malignant', case=False, regex=True).values
    is_cd8   = adata.obs[ct_key].astype(str).str.contains('CD8|Cytotoxic', case=False, regex=True).values
    if is_tumor.sum() < 50 or is_cd8.sum() < 20:
        print("  fig02: insufficient tumor or CD8 cells"); return

    # Per-treatment classification of tumor nests via DBSCAN
    cls_records = []
    cd8_tree = cKDTree(coords[is_cd8])
    sample_col = sample_key if sample_key in adata.obs else treatment_key

    for tr in adata.obs[treatment_key].unique():
        in_tr = (adata.obs[treatment_key] == tr).values
        mask_t = is_tumor & in_tr
        if mask_t.sum() < 30: continue
        c_t = coords[mask_t]
        try:
            db = DBSCAN(eps=200, min_samples=8).fit(c_t)
        except Exception:
            continue
        for nest_id in set(db.labels_):
            if nest_id == -1: continue
            nest_pts = c_t[db.labels_ == nest_id]
            centroid = nest_pts.mean(axis=0)
            n_in    = (cd8_tree.query_ball_point(nest_pts, r=50).__len__()
                       if False else sum(len(x) for x in cd8_tree.query_ball_point(nest_pts, r=50)))
            n_shell = sum(len(x) for x in cd8_tree.query_ball_point(nest_pts, r=150)) - n_in
            cls = 'Inflamed' if n_in / max(len(nest_pts), 1) > 0.05 else \
                  ('Excluded' if n_shell > n_in * 2 + 1 else 'Cold')
            cls_records.append({'treatment': tr, 'class': cls})
    if not cls_records:
        print("  fig02: no DBSCAN nests"); return

    df = pd.DataFrame(cls_records)
    counts = df.groupby(['treatment', 'class']).size().unstack(fill_value=0)
    counts = counts.reindex([t for t in TREATMENTS if t in counts.index])
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    counts.plot(kind='bar', stacked=True, ax=axes[0], legend=False,
                color={'Inflamed': '#d62728', 'Excluded': '#ff7f0e', 'Cold': '#1f77b4'})
    axes[0].set_title('Tumor nest immune phenotypes (count)', fontsize=26)
    axes[0].set_ylabel('# nests', fontsize=23)
    axes[0].set_xlabel('')
    # Horizontal, centred treatment names, matching the other Fig 7 panels.
    axes[0].tick_params(axis='x', rotation=0, labelsize=22)
    plt.setp(axes[0].get_xticklabels(), ha='center')
    axes[0].tick_params(axis='y', labelsize=22)
    pct.plot(kind='bar', stacked=True, ax=axes[1], legend=False,
             color={'Inflamed': '#d62728', 'Excluded': '#ff7f0e', 'Cold': '#1f77b4'})
    axes[1].set_title('% of nests', fontsize=26)
    axes[1].set_ylabel('Fraction (%)', fontsize=23)
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=0, labelsize=22)
    plt.setp(axes[1].get_xticklabels(), ha='center')
    axes[1].tick_params(axis='y', labelsize=22)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title='Phenotype', loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=22, title_fontsize=23)
    fig.suptitle('Immune exclusion phenotypes (Galon-style)', fontsize=29, y=1.06)
    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    fig.savefig(out_dir / 'fig02_immune_exclusion_classification.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig02_immune_exclusion_classification.pdf', bbox_inches='tight'); plt.close(fig)


# ============================================================
# 3. Perivascular niche
# ============================================================
def fig_perivascular_niche(adata, out_dir, ct_key, treatment_key, shell_um=50):
    coords = _coords(adata)
    is_endo = adata.obs[ct_key].astype(str).str.contains('Endothel', case=False, regex=True).values
    if is_endo.sum() < 30:
        print("  fig03: too few endothelial cells"); return
    endo_tree = cKDTree(coords[is_endo])
    shell_idx = np.unique(np.concatenate(endo_tree.query_ball_point(coords, r=shell_um)))
    in_shell = np.zeros(len(coords), dtype=bool)
    if len(shell_idx) == 0:
        print("  fig03: empty shell"); return
    # query_ball_point above returned tree-side indices; we instead need spot-side. Recompute:
    nbrs = endo_tree.query_ball_point(coords, r=shell_um)
    in_shell = np.array([len(n) > 0 for n in nbrs])
    df = adata.obs.loc[in_shell, [ct_key, treatment_key]].copy()
    if df.empty:
        print("  fig03: no spots in shell"); return
    comp = df.groupby([treatment_key, ct_key]).size().unstack(fill_value=0)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    comp_pct = comp_pct.reindex([t for t in TREATMENTS if t in comp_pct.index])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    comp_pct.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_title(f'Perivascular niche composition\n({shell_um} µm shell around endothelium)',
                 fontsize=22)
    ax.set_ylabel('% of spots', fontsize=21)
    ax.set_xlabel('Treatment', fontsize=23)  # match CAF/TAM/T_NK panels alongside it
    # Horizontal treatment names, matching the CAF/TAM/T_NK panels alongside it.
    ax.tick_params(axis='x', rotation=0, labelsize=20)
    plt.setp(ax.get_xticklabels(), ha='center')
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=19, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig03_perivascular_niche.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'fig03_perivascular_niche.pdf', bbox_inches='tight'); plt.close(fig)


# ============================================================
# 4. TLS detection
# ============================================================
def fig_tls(adata, out_dir, treatment_key):
    """Lymphoid aggregate marker mapping (Cd79a, Lta), the only TLS-associated
    genes present in this 4000-gene panel. Expression is extremely sparse
    (<0.1% of cells), so marker-positive cells are highlighted as an overlay
    rather than shown as a continuous score."""
    if adata.raw is None: return
    tls_genes = [g for g in ['Cd79a', 'Lta'] if g in adata.raw.var_names]
    if not tls_genes: return
    coords = _coords(adata)
    palette = {t: c for t, c in zip(TREATMENTS, ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])}
    gene_label = ' / '.join(tls_genes)

    X = adata.raw[:, tls_genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    pos_mask = X > 0
    any_pos = pos_mask.any(axis=1)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1], hspace=0.5, wspace=0.12)
    spatial_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_pct = fig.add_subplot(gs[1, :2])
    ax_comp = fig.add_subplot(gs[1, 2:])

    pct_pos, n_pos, n_tot, comp_counts = {}, {}, {}, {}
    for ax, tr in zip(spatial_axes, TREATMENTS):
        if treatment_key not in adata.obs:
            ax.set_axis_off(); continue
        m = (adata.obs[treatment_key] == tr).values
        if m.sum() == 0:
            ax.set_axis_off(); ax.set_title(tr); continue
        c = coords[m]; pos = any_pos[m]
        ax.scatter(c[~pos, 0], c[~pos, 1], s=1.5, c='lightgray', rasterized=True, linewidths=0)
        ax.scatter(c[pos, 0], c[pos, 1], s=8, c='crimson', rasterized=True, linewidths=0)
        ax.set_title(f'{tr}\n({pos.sum()} positive cells)', fontsize=25)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

        pct_pos[tr] = 100 * pos.mean()
        n_pos[tr], n_tot[tr] = int(pos.sum()), int(m.sum())
        pm = pos_mask[m]
        if len(tls_genes) == 2:
            comp_counts[tr] = {
                f'{tls_genes[0]} only': int((pm[:, 0] & ~pm[:, 1]).sum()),
                f'{tls_genes[1]} only': int((~pm[:, 0] & pm[:, 1]).sum()),
                'Both': int((pm[:, 0] & pm[:, 1]).sum()),
            }
        else:
            comp_counts[tr] = {f'{tls_genes[0]}+': int(pm[:, 0].sum())}

    legend_elem = [Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=8,
                          label=f'{gene_label}-positive'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8,
                          label='Other cells')]
    fig.legend(handles=legend_elem, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=2, frameon=False, fontsize=20)

    trs = [t for t in TREATMENTS if t in pct_pos]

    # % positive cells per treatment with 95% CI, significance vs Sham
    cis = [1.96 * np.sqrt((pct_pos[t] / 100) * (1 - pct_pos[t] / 100) / n_tot[t]) * 100 for t in trs]
    ax_pct.bar(trs, [pct_pos[t] for t in trs], yerr=cis, capsize=4,
               color=[palette[t] for t in trs])
    ax_pct.set_ylabel(f'% {gene_label}-positive cells', fontsize=23)
    ax_pct.set_title('Lymphoid marker-positive cell fraction', fontsize=26)
    ax_pct.tick_params(axis='both', labelsize=14)

    if 'Sham' in pct_pos:
        ymax_bar = max(pct_pos[t] + c for t, c in zip(trs, cis))
        for i, t in enumerate(trs):
            if t == 'Sham': continue
            p1, n1 = pct_pos[t] / 100, n_tot[t]
            p2, n2 = pct_pos['Sham'] / 100, n_tot['Sham']
            p_pool = (n_pos[t] + n_pos['Sham']) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se if se > 0 else 0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax_pct.text(i, pct_pos[t] + cis[i], star, ha='center', va='bottom', fontsize=22)
        ax_pct.set_ylim(top=ymax_bar * 1.25)

    # Composition of marker-positive cells per treatment
    comp_df = pd.DataFrame(comp_counts).T.reindex(trs).fillna(0)
    comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]
    comp_df.plot(kind='bar', stacked=True, ax=ax_comp, legend=False,
                  color=sns.color_palette('Set2', comp_df.shape[1]))
    ax_comp.set_ylabel('# positive cells', fontsize=23); ax_comp.set_xlabel('')
    ax_comp.set_title('Composition of positive cells', fontsize=26)
    ax_comp.tick_params(axis='x', rotation=0, labelsize=14)
    ax_comp.tick_params(axis='y', labelsize=14)
    handles, labels = ax_comp.get_legend_handles_labels()
    ax_comp.legend(handles, labels, loc='upper right', fontsize=19, frameon=False)

    fig.suptitle(f'Lymphoid aggregate marker spatial mapping ({gene_label})',
                  fontsize=32, y=1.02)
    # Not a curated paper figure -- skip saving (computation above still used).
    # fig.savefig(out_dir / 'fig04_tls_signature_spatial.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 4b. TLS-like structure detection via B/T cell co-localization
# ============================================================
def fig_tls_structures(adata, out_dir, ct_key, treatment_key,
                        eps_um=200.0, spatial_unit_um=0.5, min_samples=8,
                        min_b=3, min_t=5, min_cluster_size=20,
                        t_search_um=150.0):
    """Two-stage TLS detection:
      Stage 1: DBSCAN on B cells only (eps=200µm) to find B-cell cores.
      Stage 2: for each B-cell core with >= min_b cells and >= min_cluster_size
                 total lymphocytes, check whether >= min_t T cells lie within
                 t_search_um of the core centroid.
    This mirrors TLS biology (B-cell follicle surrounded by T-cell zone) and
    filters out small random B/T co-localizations that inflate counts in controls.
    """
    B_TYPES = {'B cells', 'Plasma cells', 'Memory B cells', 'Germinal center B cells'}
    T_TYPES = {'CD4 T cells', 'CD8 T cells', 'Cytotoxic T cells',
               'Effector CD4+ T cells', 'Tregs', 'NK cells',
               'NKT cells', 'Exhausted T cells', 'Memory T cells'}
    LYMPHO_TYPES = B_TYPES | T_TYPES

    eps_b   = eps_um / spatial_unit_um          # B-cell core clustering radius
    eps_t   = t_search_um / spatial_unit_um     # T-cell search radius around core
    units2_per_mm2 = (1000.0 / spatial_unit_um) ** 2

    if ct_key not in adata.obs or treatment_key not in adata.obs:
        return None

    coords_all = _coords(adata)
    ct_all     = adata.obs[ct_key].astype(str).values
    palette    = {t: c for t, c in zip(TREATMENTS, ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])}

    fig = plt.figure(figsize=(24, 20))
    gs  = fig.add_gridspec(4, 4, height_ratios=[2.0, 0.40, 1.2, 1.2], hspace=0.55, wspace=0.65,
                           top=0.92, bottom=0.07, left=0.16, right=0.97)
    spatial_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_legend_row = fig.add_subplot(gs[1, :])   # dedicated row for spatial legend
    ax_density    = fig.add_subplot(gs[2, :2])
    ax_size       = fig.add_subplot(gs[2, 2:])
    ax_comp       = fig.add_subplot(gs[3, :2])
    ax_lympho     = fig.add_subplot(gs[3, 2:])

    summary = {}
    for ax, tr in zip(spatial_axes, TREATMENTS):
        m      = (adata.obs[treatment_key] == tr).values
        coords = coords_all[m]
        ct     = ct_all[m]

        # Background tissue
        ax.scatter(coords[:, 0], coords[:, 1], s=1.5, c='lightgray',
                   rasterized=True, linewidths=0)

        is_b     = np.isin(ct, list(B_TYPES))
        is_t     = np.isin(ct, list(T_TYPES))
        is_lympho = is_b | is_t
        bc, tc   = coords[is_b], coords[is_t]

        n_mixed = n_b_only = n_t_only = 0
        cluster_sizes = []

        # Stage 1: cluster B cells
        if bc.shape[0] >= min_samples:
            b_labels = DBSCAN(eps=eps_b, min_samples=min_samples).fit_predict(bc)
            for cl in set(b_labels):
                if cl == -1:
                    continue
                b_pts = bc[b_labels == cl]
                n_b   = len(b_pts)
                if n_b < min_b:
                    continue

                # Stage 2: count T cells within t_search_um of B-core centroid
                centroid = b_pts.mean(axis=0)
                if tc.shape[0] > 0:
                    dists = np.linalg.norm(tc - centroid, axis=1)
                    t_near = tc[dists <= eps_t]
                    n_t   = len(t_near)
                else:
                    t_near = np.empty((0, 2))
                    n_t   = 0

                # Total lymphocytes in the aggregate
                all_pts = np.vstack([b_pts, t_near]) if n_t > 0 else b_pts
                if len(all_pts) < min_cluster_size:
                    continue  # too small, skip random coincidences

                cluster_sizes.append(len(all_pts))

                is_tls = n_b >= min_b and n_t >= min_t
                if is_tls:
                    n_mixed += 1
                    color, lw, z = 'crimson', 2.0, 4
                elif n_b >= min_b:
                    n_b_only += 1
                    color, lw, z = 'royalblue', 1.2, 3
                else:
                    n_t_only += 1
                    color, lw, z = 'darkorange', 1.2, 3

                ax.scatter(all_pts[:, 0], all_pts[:, 1], s=5, c=color,
                           rasterized=True, linewidths=0, zorder=z, alpha=0.85)
                if all_pts.shape[0] >= 3:
                    try:
                        hull = ConvexHull(all_pts)
                        hp   = np.vstack([all_pts[hull.vertices],
                                          all_pts[hull.vertices[:1]]])
                        ax.plot(hp[:, 0], hp[:, 1], color=color, lw=lw, zorder=z)
                    except Exception:
                        pass

        try:
            tissue_area_mm2 = ConvexHull(coords).volume / units2_per_mm2
        except Exception:
            tissue_area_mm2 = np.nan

        avg_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        density  = n_mixed / tissue_area_mm2 if tissue_area_mm2 and not np.isnan(tissue_area_mm2) else 0.0

        summary[tr] = dict(n_mixed=n_mixed, n_b_only=n_b_only, n_t_only=n_t_only,
                           tissue_area_mm2=tissue_area_mm2,
                           density_per_mm2=density,
                           avg_cluster_size=avg_size,
                           lympho_pct=100.0 * is_lympho.sum() / max(len(ct), 1))

        # Clip view tightly to this treatment's tissue with a small margin
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        xpad = (xmax - xmin) * 0.04
        ypad = (ymax - ymin) * 0.04
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

        ax.set_title(tr, fontsize=58, pad=8)
        ax.set_aspect('auto'); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.2)

    # Title only (no subtitle, keeps top clean)
    fig.suptitle('Tertiary Lymphoid Structure (TLS) Detection',
                 fontsize=70, y=0.98)

    # Dedicated legend row: spatial legend left, composition legend right
    ax_legend_row.axis('off')
    legend_elem = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',    markersize=16, label='TLS-like (B-core + T zone)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',  markersize=16, label='B-cell aggregate only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=16, label='T-cell aggregate'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',  markersize=16, label='Other cells'),
    ]
    leg_spatial = ax_legend_row.legend(handles=legend_elem, loc='center left',
                                       bbox_to_anchor=(0.0, 0.5), ncol=2,
                                       fontsize=38, frameon=True, framealpha=0.95,
                                       fancybox=True, edgecolor='#bbbbbb')
    ax_legend_row.add_artist(leg_spatial)  # keep first legend visible

    trs = [t for t in TREATMENTS if t in summary]

    # Panel A: TLS density per mm²
    bars = ax_density.bar(trs, [summary[t]['density_per_mm2'] for t in trs],
                           color=[palette[t] for t in trs], edgecolor='black', linewidth=1.5)
    for bar, t in zip(bars, trs):
        v = summary[t]['density_per_mm2']
        ax_density.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=46)
    ax_density.set_ylabel('Aggregates / mm²', fontsize=44)
    ax_density.set_title('TLS density (per tissue area)', fontsize=49)
    ax_density.tick_params(axis='both', labelsize=28)
    ax_density.set_ylim(0, max(summary[t]['density_per_mm2'] for t in trs) * 1.3)

    # Panel B: Average TLS cluster size
    bars2 = ax_size.bar(trs, [summary[t]['avg_cluster_size'] for t in trs],
                         color=[palette[t] for t in trs], edgecolor='black', linewidth=1.5)
    for bar, t in zip(bars2, trs):
        v = summary[t]['avg_cluster_size']
        ax_size.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=46)
    ax_size.set_ylabel('Avg cells / aggregate', fontsize=44)
    ax_size.set_title('TLS aggregate size (cells)', fontsize=49)
    ax_size.tick_params(axis='both', labelsize=28)
    ax_size.set_ylim(0, max((summary[t]['avg_cluster_size'] for t in trs), default=1) * 1.3)

    # Panel C: Composition stacked bar
    comp_df = pd.DataFrame({t: {'B+T mixed': summary[t]['n_mixed'],
                                  'B only':    summary[t]['n_b_only'],
                                  'T only':    summary[t]['n_t_only']} for t in trs}).T
    comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]
    comp_df.plot(kind='bar', stacked=True, ax=ax_comp, legend=False,
                  color=sns.color_palette('Set2', comp_df.shape[1]))
    ax_comp.set_ylabel('# aggregates', fontsize=44, labelpad=8)
    ax_comp.set_xlabel('')
    ax_comp.set_title('Aggregate composition', fontsize=49)
    ax_comp.tick_params(axis='x', rotation=0, labelsize=28)
    ax_comp.tick_params(axis='y', labelsize=28)
    comp_handles, comp_labels = ax_comp.get_legend_handles_labels()
    ax_comp.get_legend() and ax_comp.get_legend().remove()
    # Add composition legend to the right side of the dedicated legend row
    ax_legend_row.legend(handles=comp_handles, labels=comp_labels,
                         loc='center right', bbox_to_anchor=(1.0, 0.5),
                         ncol=1, fontsize=38, frameon=True, framealpha=0.95,
                         fancybox=True, edgecolor='#bbbbbb', title='Composition',
                         title_fontsize=35)

    # Panel D: % lymphocytes per treatment
    bars4 = ax_lympho.bar(trs, [summary[t]['lympho_pct'] for t in trs],
                           color=[palette[t] for t in trs], edgecolor='black', linewidth=1.5)
    for bar, t in zip(bars4, trs):
        v = summary[t]['lympho_pct']
        ax_lympho.text(bar.get_x() + bar.get_width()/2, v + 0.1, f'{v:.1f}%',
                       ha='center', va='bottom', fontsize=46)
    ax_lympho.set_ylabel('Lymphocytes (%)', fontsize=44)
    ax_lympho.set_title('Lymphocyte infiltration per treatment', fontsize=49)
    ax_lympho.tick_params(axis='both', labelsize=28)
    ax_lympho.set_ylim(0, max(summary[t]['lympho_pct'] for t in trs) * 1.3)

    # Not a curated paper figure -- skip saving (computation above still used).
    # fig.savefig(out_dir / 'fig04b_tls_structures_spatial.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("TLS-like (B+T mixed) aggregate summary:")
    for t in trs:
        s = summary[t]
        print(f"  {t}: {s['n_mixed']} TLS, {s['n_b_only']} B-only, {s['n_t_only']} T-only | "
              f"area={s['tissue_area_mm2']:.1f}mm² density={s['density_per_mm2']:.2f}/mm² "
              f"avg_size={s['avg_cluster_size']:.0f}cells lympho={s['lympho_pct']:.1f}%")
    return summary


# ============================================================
# 5. Hypoxia spatial map
# ============================================================
def fig_hypoxia(adata, out_dir, treatment_key):
    key = _score(adata, 'Hyp', GENES['Hypoxia'])
    if not key: return
    coords = _coords(adata)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, tr in zip(axes, TREATMENTS):
        if treatment_key not in adata.obs: break
        m = (adata.obs[treatment_key] == tr).values
        if m.sum() == 0:
            ax.set_axis_off(); ax.set_title(tr); continue
        c = coords[m]; s = adata.obs.loc[m, key].values
        sc_ = ax.scatter(c[:, 0], c[:, 1], c=s, s=3, cmap='Reds',
                         vmin=np.percentile(s, 2), vmax=np.percentile(s, 98),
                         rasterized=True, linewidths=0)
        # Explicit size: these inherit rcParams otherwise, which leaves the
        # treatment names far smaller than the panel heading above them.
        ax.set_title(tr, fontsize=20)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sc_, ax=axes.ravel().tolist(), shrink=0.7,
                      label='Hypoxia score')
    cb.set_label('Hypoxia score', fontsize=17)
    cb.ax.tick_params(labelsize=15)
    fig.suptitle('Buffa hypoxia signature (Hif1a/Vegfa/Slc2a1/Ldha/Ca9/...)',
                 fontsize=17)
    fig.savefig(out_dir / 'fig05_hypoxia_core_map.png')
    # PDF twin for the vector publication composite.
    fig.savefig(out_dir / 'fig05_hypoxia_core_map.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    if key in adata.obs: del adata.obs[key]


# ============================================================
# 6. Neighborhood Z-score matrix (squidpy)
# ============================================================
def fig_nhood_zscore(adata, out_dir, ct_key, treatment_key):
    try:
        import squidpy as sq
    except Exception:
        print("  fig06: squidpy not available"); return
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, tr in zip(axes, TREATMENTS):
        if treatment_key not in adata.obs:
            ax.set_axis_off(); continue
        sub = adata[adata.obs[treatment_key] == tr].copy()
        if sub.n_obs < 200:
            ax.set_axis_off(); ax.set_title(tr); continue
        try:
            sq.gr.spatial_neighbors(sub, coord_type='generic', delaunay=True)
            sq.gr.nhood_enrichment(sub, cluster_key=ct_key, n_perms=200, seed=0)
            z = sub.uns[f'{ct_key}_nhood_enrichment']['zscore']
            cats = sub.obs[ct_key].cat.categories if hasattr(sub.obs[ct_key], 'cat') else sorted(sub.obs[ct_key].unique())
            im = ax.imshow(z, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
            ax.set_xticks(range(len(cats))); ax.set_yticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=90, fontsize=11)
            ax.set_yticklabels(cats, fontsize=11)
            ax.set_title(tr)
        except Exception as e:
            ax.set_axis_off(); ax.set_title(f'{tr}\n{type(e).__name__}'); continue
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='Neighborhood Z')
    fig.suptitle('Neighborhood enrichment Z-scores per treatment',
                 fontsize=17)
    # Not a curated paper figure -- skip saving (computation above still used).
    # fig.savefig(out_dir / 'fig06_neighborhood_zscore_matrix.png'); plt.close(fig)


# ============================================================
# 7. MHC-I antigen presentation map
# ============================================================
def fig_mhc(adata, out_dir, treatment_key):
    key = _score(adata, 'MHC', GENES['MHC_I'])
    if not key: return
    coords = _coords(adata)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, tr in zip(axes, TREATMENTS):
        if treatment_key not in adata.obs: break
        m = (adata.obs[treatment_key] == tr).values
        if m.sum() == 0:
            ax.set_axis_off(); ax.set_title(tr); continue
        c = coords[m]; s = adata.obs.loc[m, key].values
        vmax = float(np.nanpercentile(np.abs(s), 95)) if np.any(np.isfinite(s)) else 1.0
        if not np.isfinite(vmax) or vmax == 0: vmax = 1.0
        sc_ = ax.scatter(c[:, 0], c[:, 1], c=s, s=2, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, rasterized=True, linewidths=0)
        # Highlight low-MHC patches
        thr_low = np.percentile(s, 5)
        cold = c[s < thr_low]
        if len(cold) >= 10:
            try:
                db = DBSCAN(eps=60, min_samples=10).fit(cold)
                n_loss = len(set(db.labels_) - {-1})
                ax.set_title(f'{tr}  (loss patches≈{n_loss})')
            except Exception:
                ax.set_title(tr)
        else:
            ax.set_title(tr)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc_, ax=axes.ravel().tolist(), shrink=0.7, label='MHC-I score')
    fig.suptitle('MHC class-I antigen presentation (H2-K1/D1/B2m/Tap1-2/Nlrc5)',
                 fontsize=17)
    # Not a curated paper figure -- skip saving (computation above still used).
    # fig.savefig(out_dir / 'fig07_mhc_i_loss_map.png'); plt.close(fig)
    if key in adata.obs: del adata.obs[key]


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--h5ad', default=None)
    p.add_argument('--ct_key', default='cell_type')
    p.add_argument('--treatment_key', default='treatment')
    p.add_argument('--skip_nhood', action='store_true', help='skip squidpy nhood Z-score')
    args = p.parse_args()

    base = Path(args.input_dir)
    proc = base / 'downstream_analysis' / 'processed_data'
    out_dir = base / 'downstream_analysis' / 'figures' / 'post9_pdac_spatial'
    out_dir.mkdir(parents=True, exist_ok=True)

    h5ad = Path(args.h5ad) if args.h5ad else proc / 'merged_annotated.h5ad'
    print("="*70)
    print("POST-9: PDAC-specific spatial analyses")
    print(f"  output: {out_dir}")
    print("="*70)

    if not h5ad.exists():
        print(f"ERROR: {h5ad} not found"); return
    adata = sc.read_h5ad(h5ad)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    ct, tr = args.ct_key, args.treatment_key
    if ct not in adata.obs:
        for alt in ('cell_type_auto', 'cluster_annotation', 'celltype'):
            if alt in adata.obs: ct = alt; break
    if tr not in adata.obs:
        for alt in ('Treatment', 'group'):
            if alt in adata.obs: tr = alt; break

    print("\n[1/7] Distance-to-tumor gradients...")
    try: fig_distance_gradient(adata, out_dir, ct, tr)
    except Exception as e: print(f"  fig01 failed: {e}")

    print("\n[2/7] Immune exclusion classification...")
    try: fig_immune_exclusion(adata, out_dir, ct, tr)
    except Exception as e: print(f"  fig02 failed: {e}")

    print("\n[3/7] Perivascular niche...")
    try: fig_perivascular_niche(adata, out_dir, ct, tr)
    except Exception as e: print(f"  fig03 failed: {e}")

    print("\n[4/7] TLS signature spatial...")
    try: fig_tls(adata, out_dir, tr)
    except Exception as e: print(f"  fig04 failed: {e}")

    print("\n[5/7] Hypoxia spatial map...")
    try: fig_hypoxia(adata, out_dir, tr)
    except Exception as e: print(f"  fig05 failed: {e}")

    if not args.skip_nhood:
        print("\n[6/7] Neighborhood Z-score matrix...")
        try: fig_nhood_zscore(adata, out_dir, ct, tr)
        except Exception as e: print(f"  fig06 failed: {e}")

    print("\n[7/7] MHC-I loss map...")
    try: fig_mhc(adata, out_dir, tr)
    except Exception as e: print(f"  fig07 failed: {e}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    rel = os.path.join('downstream_analysis', 'figures', 'post9_pdac_spatial')
    collect(args.input_dir, {
        os.path.join(rel, 'fig01_distance_to_tumor_gradient'):        'fig7_h',
        os.path.join(rel, 'fig02_immune_exclusion_classification'):   'fig7_g',
        os.path.join(rel, 'fig03_perivascular_niche'):                'fig7_f',
        os.path.join(rel, 'fig05_hypoxia_core_map'):                  'suppl13_a',
    })

    print("\nPOST-9 COMPLETE"); print(f"Figures: {out_dir}")


if __name__ == '__main__':
    main()
