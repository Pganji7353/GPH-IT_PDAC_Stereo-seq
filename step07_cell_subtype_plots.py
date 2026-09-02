#!/usr/bin/env python3
"""
Publication-quality cell-type-specific plots for the Stereo-seq PDAC analysis.

Sections:
  1. PDAC: mPDAC + ePDAC combined (9 dotplots, 2 violin plots)
  2. CAFs, myCAFs, iCAF, apCAFs, qCAF
             (subtype marker dotplot, 3 pathway dotplots, 2 violin,
              UMAP combined + per-treatment, PAGA trajectories, 2 barplots)
  3. TAMs, M1 Macrophage, M2 Macrophage
             (marker dotplot, M1/M2 dotplot, Mrc1 ridge plot,
              UMAP per-treatment, 2 barplots)
  4. T/NK: CD4/CD8/Effector-CD4/Cytotoxic/NK/Tregs
             (marker dotplot, effector dotplot, UMAP per-treatment, barplot)
  5. Endothelial (skipped if not present)
             (cytokine, growth factor, ER-stress, angiogenic dotplots)

Run after: step02_build_annotated_h5ad.py

Usage:
    python step07_cell_subtype_plots.py --h5ad_path merged_annotated.h5ad --out_dir figures/post6_subtype_plots
    python step07_cell_subtype_plots.py --h5ad_path merged_annotated.h5ad --out_dir figures/post6_subtype_plots --sections pdac caf
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
from scipy.stats import gaussian_kde

import matplotlib.font_manager as _fm

# ── Publication-quality font & style ─────────────────────────────────────────
_pref = next(
    (f for f in ['Arial', 'Liberation Sans', 'FreeSans', 'DejaVu Sans']
     if f in {x.name for x in _fm.fontManager.ttflist}),
    'sans-serif'
)
plt.rcParams.update({
    'font.family':          _pref,
    'font.size':            10,
    'axes.titlesize':       10,
    'axes.labelsize':       10,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      10,
    'legend.title_fontsize': 10,
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
    'figure.dpi':           150,
})

# ── Treatment order & colors ──────────────────────────────────────────────────
TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
TREAT_COLORS = {
    'Sham':   '#9E9E9E',
    'IT':     '#4E79A7',
    'GPH':    '#59A14F',
    'GPH+IT': '#E15759',
}

MAX_CELLS_UMAP = 80_000   # subsample for UMAP speed; dotplots use all cells


# =============================================================================
#  GENE LISTS (all mouse gene symbols)
# =============================================================================

# ── PDAC ─────────────────────────────────────────────────────────────────────
PDAC_SURVIVAL = [
    'Tgfb1','Ccl2','Chek2','Chek1','Itgb4','Itgb1','Itga9','Itga6',
    'Becn1','Stat3','Nfkb1','Birc5','Bak1','Bax','Bcl2','Mcl1',
    'Igf1r','Erbb2','Map2k1','Braf','Mapk1','Mtor','Akt1','Akt2',
    'Pik3r1','Pik3cb','Pik3ca','Pten','Smad4','Cdkn2a','Kras',
]
PDAC_MESENCHYMAL = [
    'Zeb2','Vim','Vcan','Sparc','Pmp22','Pcolce','Mertk','Nme2',
    'Inhba','Htra1','Gypc','Fstl1','Fn1','Emp3','Dact1',
    'Col6a2','Col6a1','Cmtm3','Cald1','Axl','Adamts2',
]
PDAC_EMT_TF = ['Zeb2','Zeb1','Twist2','Twist1','Snai2','Snai1']
PDAC_RAS    = ['Ralb','Rala','Rheb1','Rheb','Rras2','Rras','Nras','Hras','Kras']
PDAC_INSULIN = [
    'Slc2a1','Pck2','Pck1','Foxo3','Foxo1','Sos2','Sos1','Grb2',
    'Mapk1','Map2k1','Raf1','Pdk1','Tsc2','Tsc1','Rictor','Mtor',
    'Akt1','Akt2','Pik3r1','Pik3cb','Pik3ca','Irs4','Irs2','Irs1',
    'Igf2r','Igf1r','Igf2','Igf1','Ins2','Ins1',
]
PDAC_CYTOKINES = [
    'Cxcl12','Cxcl10','Cxcl9','Cxcl5','Cxcl2','Cxcl1',
    'Ccl7','Ccl5','Ccl4','Ccl3','Ccl2','Tgfb1',
    'Il33','Il23a','Il17a','Il11','Il10','Il6','Il1b',
]
PDAC_GROWTH_FACTORS = [
    'Mdk','Ngf','Bmp7','Bmp4','Bmp2','Tgfa','Egf',
    'Pdgfd','Pdgfc','Pdgfb','Pdgfa','Hgf',
    'Igf2','Igf1','Fgf10','Fgf7','Fgf2','Fgf1',
    'Vegfc','Vegfb','Vegfa',
]
PDAC_INVASION = [
    'Adam17','Adam10','Ctsd','Ctsb','Serpine1',
    'Plaur','Plat','Plau','Mmp14','Mmp9','Mmp7','Mmp3','Mmp2',
]
PDAC_STEMNESS = [
    'Krt19','Tead2','Tead1','Yap1','Smo','Lef1','Ctnnb1','Dll1',
    'Notch2','Notch1','Aldh1a3','Epcam','Mki67','Prom1',
    'Cd24a','Cd44','Hnf1b','Nanog','Sox9','Sox2',
]
PDAC_VIOLIN_GENES = ['Hif1a', 'Stat3']

# ── CAFs ──────────────────────────────────────────────────────────────────────
CAF_SUBTYPES = ['myCAFs', 'iCAF', 'apCAFs', 'qCAF']
CAF_SUBTYPE_MARKERS = {
    'myCAFs': ['Acta2','Tagln','Myl9','Tpm2','Hhip','Sfrp2','Sfrp4','Pdgfa','Postn'],
    'iCAFs':  ['Il6','Cxcl1','Cxcl12','Has1','Ly6c1','Pdgfra','Clec3b','Cxcl2','Cxcl5'],
    'apCAFs': ['H2-Ab1','H2-Aa','Cd74','Slpi','Saa3','Ptprc','Ly6d','Ccl8'],
    'qCAFs':  ['Col1a1','Col1a2','Col3a1','Mfap5','Mmp2','Lox','Dcn','Fn1'],
}
CAF_AUTOPHAGY = [
    'Ulk1','Ulk2','Becn1','Atg5','Atg7','Atg12','Atg14',
    'Rb1cc1','Pik3c3','Ambra1','Gabarap','Gabarapl1','Gabarapl2',
    'Map1lc3a','Map1lc3b','Sqstm1','Calcoco2',
    'Bnip3','Bnip3l','Fis1','Casp3','Lamp1','Lamp2',
]
CAF_ER_STRESS = [
    'Xbp1','Ddit3','Atf4','Atf6','Hspa5','Hsp90b1',
    'Pdia3','Calr','Ern1','Eif2ak3','Hyou1',
    'Dnajb11','Cebpa','Mapk1','Casp3',
]
CAF_ASPARTATE = [
    'Got1','Got2','Mdh1','Mdh2','Slc25a11','Slc25a12',
    'Gls','Gls2','Adss','Adsl','Ass1','Asl','Ctps1','Nme1','Nme2',
]
CAF_VIOLIN_GENES = ['Hif1a', 'Stat3']

# ── TAMs / Macrophages ────────────────────────────────────────────────────────
TAM_SUBTYPES = ['M1 Macrophage', 'M2 Macrophage']
TAM_MARKERS = [
    # M2 / pro-tumoral
    'Mrc1','Cd163','Lyve1','Folr2','Timd4','Pdcd1lg2','Cd274',
    'Arg1','Arg2','Il10','Tgfb1',
    # M1 / inflammatory
    'Il1b','Tnf','Il6','Nos2','Cxcl10','Cxcl9','Il12a','Il12b',
    # Angiogenic
    'Vegfa','Mmp9','Nrp1','Ang',
    # IFN-stimulated
    'Ifit1','Ifit3','Isg15','Mx1','Cxcl10',
    # Tissue-resident / lipid-associated
    'Spp1','Gpnmb','Trem2','Lgals3','Apoe',
    # Phagocytosis / general
    'Csf1r','Adgre1','Fcgr1','Itgam','Cx3cr1',
    # Proliferating
    'Mki67','Top2a',
    # Co-stimulatory
    'Cd80','Cd86',
]
TAM_M1M2_MARKERS = ['Cd163','Mrc1','Arg1','Nos2','Il1b','Il10','Cd80','Cd86']
TAM_RIDGEPLOT_GENES = ['Mrc1']

# ── T / NK cells ─────────────────────────────────────────────────────────────
TCELL_SUBTYPES = [
    'CD4 T cells', 'CD8 T cells', 'Effector CD4+ T cells',
    'Cytotoxic T cells', 'NK cells', 'Tregs', 'B cells',
]
TCELL_MARKERS = [
    # CD8 T
    'Cd8a','Cd8b1','Gzmb','Gzma','Prf1','Ifng','Klrg1',
    # NK
    'Nkg7','Ncr1','Klrb1c','Klrd1','Xcl1',
    # Treg
    'Foxp3','Il2ra','Ctla4','Ikzf2','Tnfrsf18',
    # B cell markers (reference dataset has B cells)
    'Cd79a','Cd79b','Ms4a1','Ighm','Ighd',
    # T NK
    'Klrc1','Klrk1',
    # CD4
    'Cd4','Il7r','Ccr7',
    # General T
    'Cd3e','Cd3d','Cd3g','Trac','Trbc2','Cd2',
    # Exhaustion
    'Pdcd1','Lag3','Havcr2','Tigit',
    # Activation
    'Tnfrsf4','Tnfrsf9',
    # Cytotoxic
    'Gzmk','Gzmm',
]
TCELL_EFFECTOR = [
    'Ifng','Tnf','Il2','Gzmb','Gzma','Prf1','Fasl',
    'Klrg1','Tbx21','Hif1a','Fas',
    'Eomes','Prdm1','Id2',
]

# ── Endothelial (optional) ────────────────────────────────────────────────────
ENDO_CYTOKINES = [
    'Cxcl12','Cxcl10','Cxcl9','Cxcl5','Cxcl2','Cxcl1',
    'Ccl7','Ccl5','Ccl4','Ccl3','Ccl2','Tgfb1',
    'Il33','Il23a','Il17a','Il11','Il10','Il6','Il1b',
]
ENDO_GROWTH_FACTORS = [
    'Mdk','Ngf','Bmp7','Bmp4','Bmp2','Tgfa','Egf',
    'Pdgfd','Pdgfc','Pdgfb','Pdgfa','Hgf',
    'Igf2','Igf1','Fgf10','Fgf7','Fgf2','Fgf1',
    'Vegfc','Vegfb','Vegfa',
]
ENDO_ER_STRESS = [
    'Xbp1','Ddit3','Atf4','Atf6','Hspa5','Hsp90b1',
    'Ern1','Eif2ak3','Hyou1','Cebpa','Mapk1','Casp3',
]
ENDO_ANGIOGENIC = [
    'Angptl4','Dll4','Timp2','Timp1','Egfr','Tgfb1','Ccl2','Il6',
    'Epas1','Hif1a','Itgb5','Itgb3','Itgav','Itga5',
    'Serpine1','Plau','Plat','Mmp2','Mmp9','Mmp14',
    'Tek','Pdgfrb','Fgfr2','Fgfr1','Nrp2','Nrp1',
    'Flt4','Flt1','Kdr','Angpt2','Angpt1',
    'Pdgfd','Pdgfb','Fgf2','Fgf1','Vegfd','Vegfc','Vegfb','Vegfa',
]


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def get_expr(matrix):
    """Safely flatten sparse or dense expression matrix to 1-D numpy array."""
    import scipy.sparse as sp
    if sp.issparse(matrix):
        return np.asarray(matrix.todense()).flatten()
    return np.asarray(matrix).flatten()


def filter_genes(gene_list, adata, min_expr_fraction=0.005):
    """
    Return only genes that are:
      1. Present in the dataset (raw.var_names or var_names)
      2. Expressed in at least min_expr_fraction of cells in the subset

    This prevents generating dotplots/violins where all dots are empty
    because the genes exist in the genome index but aren't detected in
    this particular cell-type subset.
    """
    import scipy.sparse as sp

    if adata.raw is not None:
        var_names  = adata.raw.var_names
        get_col    = lambda g: adata.raw[:, g].X
    else:
        var_names  = adata.var_names
        get_col    = lambda g: adata[:, g].X

    available  = set(var_names)
    filtered   = []
    missing    = []
    low_expr   = []

    for g in dict.fromkeys(gene_list):         # dedup, preserve order
        if g not in available:
            missing.append(g)
            continue
        x = get_col(g)
        n_expr = int((x > 0).sum()) if not sp.issparse(x) else int((x > 0).nnz)
        frac   = n_expr / max(x.shape[0], 1)
        if frac < min_expr_fraction:
            low_expr.append(g)
        else:
            filtered.append(g)

    if missing:
        print(f"    [absent from dataset ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}]")
    if low_expr:
        print(f"    [expr <{min_expr_fraction:.0%} in subset, dropped ({len(low_expr)}): "
              f"{low_expr[:5]}{'...' if len(low_expr) > 5 else ''}]")
    if filtered:
        print(f"    [expressed genes kept: {len(filtered)}/{len(set(gene_list))}]")
    return filtered


# Filename stems that collect_results.py copies out as curated paper figures;
# every other figure this script builds is still computed but not saved, since
# only these end up in the paper.
_CURATED_STEMS = {
    'CAF_proportion_barplot', 'TAM_proportion_barplot', 'T_NK_proportion_barplot',
    'caf_subtype_markers',
}


def save_fig(path_no_ext, dpi=300):
    """Save current figure as both PNG and PDF, if it's a curated paper figure."""
    if os.path.basename(path_no_ext) not in _CURATED_STEMS:
        return
    for ext in ('png', 'pdf'):
        plt.savefig(f"{path_no_ext}.{ext}", dpi=dpi,
                    bbox_inches='tight', facecolor='white')


def make_dotplot(adata, genes, groupby, title, out_path_prefix,
                 swap_axes=True, use_raw=True, figsize=None):
    """
    Publication-quality dotplot.
    swap_axes=True → genes on y-axis (rows), groups on x-axis (columns).
    """
    genes = filter_genes(genes, adata, min_expr_fraction=0.005)
    if len(genes) < 2:
        print(f"    SKIP {title}: fewer than 2 expressed genes in this subset")
        return

    use_raw = use_raw and (adata.raw is not None)
    n_genes  = len(genes)
    n_groups = adata.obs[groupby].nunique()

    if figsize is None:
        if swap_axes:
            w = max(8, n_groups * 1.6 + 3.0)   # min 8 prevents legend squish
            h = max(5, n_genes  * 0.45 + 2.5)
        else:
            w = max(8, n_genes  * 0.50 + 3.5)
            h = max(5, n_groups * 0.60 + 2.5)
        figsize = (w, h)

    # scanpy's dotplot takes every label size from rcParams, which sit at 10pt
    # for this module -- far too small once the panel is scaled onto a figure
    # page. Raise them for this plot only, and crucially WITHOUT enlarging the
    # canvas: the composite normalises panels by rendered text size, so growing
    # both together would cancel out and change nothing. The gene rows are the
    # tightest axis at ~31pt of pitch, so 22pt type still clears its neighbours.
    dot_rc = {
        'font.size': 22, 'axes.titlesize': 26, 'axes.labelsize': 22,
        'xtick.labelsize': 22, 'ytick.labelsize': 22,
        'legend.fontsize': 20, 'legend.title_fontsize': 21,
    }
    try:
        with plt.rc_context(dot_rc):
            dp = sc.pl.dotplot(
                adata, genes, groupby=groupby,
                use_raw=use_raw,
                color_map='RdBu_r',
                swap_axes=swap_axes,
                title=title,
                figsize=figsize,
                show=False,
                return_fig=True,
                colorbar_title='Average\nExpression',
                size_title='Percent\nExpressed',
            )
            dp.style(
                cmap='RdBu_r',
                dot_edge_color='black',
                dot_edge_lw=0.3,
                grid=True,
            )
            # The size legend's tick labels sit at fixed fractions of the legend
            # column, so at the enlarged type "10 30 50 70" ran together into
            # "10305070". Widening the column is what separates them; shrinking
            # the type back would undo the legibility this whole block buys.
            try:
                dp.legend(width=3.2)
            except Exception:
                pass
            if os.path.basename(out_path_prefix) in _CURATED_STEMS:
                dp.savefig(f"{out_path_prefix}.png", dpi=300,
                           bbox_inches='tight', facecolor='white')
                dp.savefig(f"{out_path_prefix}.pdf", dpi=300,
                           bbox_inches='tight', facecolor='white')
        plt.close('all')
        print(f"    Saved: {os.path.basename(out_path_prefix)}")
    except Exception as e:
        print(f"    WARNING: dotplot '{title}' failed: {e}")
        plt.close('all')


def make_violin_plots(adata, genes, groupby, fname_prefix, out_dir,
                      order=None, palette=None, use_raw=True):
    """Per-gene violin plots split by treatment."""
    if order is None:
        order = TREATMENT_ORDER
    if palette is None:
        palette = TREAT_COLORS

    expressed_genes = filter_genes(genes, adata, min_expr_fraction=0.005)
    for gene in expressed_genes:
        use_r = use_raw and (adata.raw is not None) and (gene in adata.raw.var_names)
        if use_r:
            expr = get_expr(adata.raw[:, gene].X)
        else:
            expr = get_expr(adata[:, gene].X)

        order_present = [t for t in order if t in adata.obs[groupby].unique()]
        df = pd.DataFrame({'Expression': expr,
                           'Treatment':  adata.obs[groupby].values})
        df = df[df['Treatment'].isin(order_present)]

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = [palette[t] for t in order_present]
        sns.violinplot(
            data=df, x='Treatment', y='Expression',
            order=order_present,
            palette=dict(zip(order_present, colors)),
            inner='box', cut=0, linewidth=1.2, ax=ax,
        )
        ax.set_title(gene, fontsize=26)
        ax.set_xlabel('Treatment', fontsize=23)
        ax.set_ylabel('Expression Level', fontsize=23)
        # Horizontal and centred: the four treatment names are short enough to sit
        # flat, and the 15-degree tilt made them read as ragged and off-centre
        # under their bars.
        plt.setp(ax.get_xticklabels(), fontsize=18, rotation=0, ha='center')
        ax.tick_params(axis='y', labelsize=16)
        fig.tight_layout()
        sns.despine(ax=ax)
        plt.tight_layout()

        save_fig(os.path.join(out_dir, f"{fname_prefix}_{gene}_violin"))
        plt.close()
        print(f"    Saved: {fname_prefix}_{gene}_violin")


def make_ridgeplot(adata, gene, groupby, out_dir,
                   order=None, palette=None, use_raw=True):
    """Ridgeline density plot for a single gene across groups."""
    if order is None:
        order = TREATMENT_ORDER
    if palette is None:
        palette = TREAT_COLORS

    expressed = filter_genes([gene], adata, min_expr_fraction=0.005)
    if not expressed:
        print(f"    SKIP ridgeplot {gene}: not expressed in this subset")
        return
    use_r = use_raw and (adata.raw is not None) and (gene in adata.raw.var_names)
    if use_r:
        expr = get_expr(adata.raw[:, gene].X)
    else:
        expr = get_expr(adata[:, gene].X)

    order_present = [t for t in order if t in adata.obs[groupby].unique()]
    groups = adata.obs[groupby].values
    x_max  = float(np.quantile(expr[expr > 0], 0.99)) if (expr > 0).sum() > 0 else 3.0
    x_range = np.linspace(0, x_max, 400)

    n = len(order_present)
    fig, axes = plt.subplots(n, 1, figsize=(8, n * 1.9), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (treat, ax) in enumerate(zip(reversed(order_present), axes)):
        mask = groups == treat
        vals = expr[mask]
        color = palette.get(treat, '#888888')
        if vals.sum() > 0 and len(vals) >= 20:
            kde     = gaussian_kde(vals, bw_method=0.3)
            density = kde(x_range)
            ax.fill_between(x_range, density, alpha=0.78, color=color, linewidth=0)
            ax.plot(x_range, density, color='black', linewidth=0.8)
        ax.set_ylabel(treat, fontsize=17, rotation=0, ha='right', va='center',
                      labelpad=80)
        ax.set_yticks([])
        ax.axhline(0, color='black', linewidth=0.6)
        sns.despine(ax=ax, left=True, bottom=(i < n - 1))

    axes[-1].set_xlabel('Expression Level', fontsize=20)
    fig.suptitle(gene, fontsize=26, y=1.01)
    plt.tight_layout(h_pad=-0.8)

    save_fig(os.path.join(out_dir, f"ridgeplot_{gene}"))
    plt.close()
    print(f"    Saved: ridgeplot_{gene}")


def make_barplots(adata, ct_col, ct_order, out_dir, title_prefix,
                  treat_order=None, treat_colors=None):
    """Stacked barplots: absolute counts + proportions per treatment."""
    if treat_order is None:
        treat_order = TREATMENT_ORDER
    if treat_colors is None:
        treat_colors = TREAT_COLORS

    counts = (adata.obs
              .groupby(['treatment', ct_col])
              .size()
              .unstack(fill_value=0))
    ct_cols_present = [c for c in ct_order if c in counts.columns]
    treat_present   = [t for t in treat_order if t in counts.index]
    counts = counts.reindex(index=treat_present, columns=ct_cols_present, fill_value=0)
    props  = counts.div(counts.sum(axis=1), axis=0)

    n_ct   = len(ct_cols_present)
    cmap   = matplotlib.colormaps.get_cmap('tab20').resampled(max(n_ct, 1))
    ct_clr = {ct: cmap(i) for i, ct in enumerate(ct_cols_present)}

    for data, ylabel, fsuffix in [
        (counts, 'Cell Count',  'counts_barplot'),
        (props,  'Proportion',  'proportion_barplot'),
    ]:
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        bottom = np.zeros(len(data.index))
        for ct in data.columns:
            vals = data[ct].values.astype(float)
            ax.bar(data.index, vals, bottom=bottom,
                   color=ct_clr[ct], label=ct, edgecolor='none', width=0.65)
            bottom += vals

        ax.set_xlabel('Treatment', fontsize=23)
        ax.set_ylabel(ylabel, fontsize=23)
        ax.set_title(f'{title_prefix} — {ylabel} per Treatment',
                     fontsize=26)
        # Horizontal and centred: the four treatment names are short enough to sit
        # flat, and the 15-degree tilt made them read as ragged and off-centre
        # under their bars.
        plt.setp(ax.get_xticklabels(), fontsize=18, rotation=0, ha='center')
        ax.tick_params(axis='y', labelsize=16)
        fig.tight_layout()
        sns.despine(ax=ax)

        handles = [mpatches.Patch(color=ct_clr[ct], label=ct) for ct in data.columns]
        ax.legend(handles=handles, title='Cell Type',
                  bbox_to_anchor=(1.02, 1), loc='upper left',
                  frameon=False, fontsize=16, title_fontsize=17)
        plt.tight_layout(rect=[0, 0, 0.83, 1])

        fname = f"{title_prefix.replace(' ','_').replace('/','_')}_{fsuffix}"
        save_fig(os.path.join(out_dir, fname))
        plt.close()
        print(f"    Saved: {fname}")


def prep_subset_for_umap(adata_sub, n_top_genes=2000, n_pcs=30,
                          n_neighbors=15, random_state=42,
                          batch_key='treatment', cell_type_key='cell_type_auto',
                          max_per_subtype=15000, umap_min_dist=0.5):
    """
    Full preprocessing pipeline on a cell-type subset for UMAP.
    Uses raw (all genes) if available; otherwise uses adata.X.
    Stratified subsampling ensures rare subtypes are represented.
    Harmony batch correction removes treatment-batch effects.
    """
    # Start from raw (all genes, normalized) if available
    if adata_sub.raw is not None:
        a = adata_sub.raw.to_adata()
        a.obs = adata_sub.obs.copy()
    else:
        a = adata_sub.copy()

    # Stratified subsample: cap each subtype at max_per_subtype to preserve rare populations
    if a.n_obs > MAX_CELLS_UMAP and cell_type_key in a.obs.columns:
        import numpy as np
        subtypes = a.obs[cell_type_key].unique()
        keep_idx = []
        np.random.seed(random_state)
        for st in subtypes:
            idx = np.where(a.obs[cell_type_key] == st)[0]
            if len(idx) > max_per_subtype:
                idx = np.random.choice(idx, max_per_subtype, replace=False)
            keep_idx.append(idx)
        keep_idx = np.concatenate(keep_idx)
        # If still above limit, downsample uniformly
        if len(keep_idx) > MAX_CELLS_UMAP:
            keep_idx = np.random.choice(keep_idx, MAX_CELLS_UMAP, replace=False)
        a = a[keep_idx].copy()
        print(f"    Stratified subsample: {a.n_obs:,} cells ({len(subtypes)} subtypes)")
    elif a.n_obs > MAX_CELLS_UMAP:
        print(f"    Subsampling {a.n_obs:,} → {MAX_CELLS_UMAP:,} for UMAP...")
        sc.pp.subsample(a, n_obs=MAX_CELLS_UMAP, random_state=random_state)

    # Ensure log-normalized
    if a.X.max() > 30:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)

    # seurat_v3 is appropriate for UMI/count data (CellBin)
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_top_genes, a.n_vars - 1),
                                  flavor='seurat_v3', batch_key=batch_key
                                  if batch_key in a.obs.columns else None)
    a = a[:, a.var['highly_variable']].copy()
    sc.pp.scale(a, max_value=10)
    n_comps = min(n_pcs, a.n_obs - 1, a.n_vars - 1)
    sc.tl.pca(a, n_comps=n_comps)

    # Harmony batch correction by treatment to remove batch-driven scatter
    if batch_key in a.obs.columns and a.obs[batch_key].nunique() > 1:
        try:
            import scanpy.external as sce
            sce.pp.harmony_integrate(a, batch_key, basis='X_pca',
                                     adjusted_basis='X_pca_harmony',
                                     max_iter_harmony=20, random_state=random_state)
            use_rep = 'X_pca_harmony'
            print(f"    Harmony correction applied (batch_key='{batch_key}')")
        except Exception as harmony_err:
            print(f"    Harmony failed ({harmony_err}), using X_pca")
            use_rep = 'X_pca'
    else:
        use_rep = 'X_pca'

    sc.pp.neighbors(a, n_neighbors=n_neighbors, n_pcs=n_comps, use_rep=use_rep)
    sc.tl.umap(a, random_state=random_state, min_dist=umap_min_dist)
    return a


# Publication-quality colour palette (colourblind-friendly)
_SUBTYPE_COLORS = [
    '#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261',
    '#9B5DE5', '#F15BB5', '#00BBF9', '#06D6A0', '#FFB703',
    '#8B4513', '#556B2F', '#4B0082', '#DC143C', '#008B8B',
]

def _color_is_dark(hex_color):
    """Return True if colour is dark (use white text over it)."""
    try:
        r, g, b = matplotlib.colors.to_rgb(hex_color)
        return (0.299 * r + 0.587 * g + 0.114 * b) < 0.45
    except Exception:
        return False


def _add_kde_contour(ax, pts, color, bw=0.25, level_frac=0.12, lw=2.0):
    """Draw a single-level KDE contour around a cluster of 2-D points."""
    if pts.shape[0] < 80:
        return
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(pts.T, bw_method=bw)
        x0, x1 = pts[:, 0].min() - 0.6, pts[:, 0].max() + 0.6
        y0, y1 = pts[:, 1].min() - 0.6, pts[:, 1].max() + 0.6
        xx, yy = np.mgrid[x0:x1:80j, y0:y1:80j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[zz.max() * level_frac],
                   colors=[color], linewidths=lw, alpha=0.95, zorder=3)
    except Exception:
        pass


def make_subset_umaps(adata_sub, color_col, title, out_dir, fname_prefix,
                       palette=None, treat_col='treatment',
                       treat_order=None, treat_colors=None):
    """
    Two UMAP figures:
      (A) Combined: scatter + KDE contours + centroid labels
      (B) Per-treatment: 4 panels, grey background, coloured foreground
    """
    if treat_order is None:
        treat_order = TREATMENT_ORDER
    if 'X_umap' not in adata_sub.obsm:
        print(f"    No UMAP in subset, skipping UMAP plots for {title}")
        return

    coords = adata_sub.obsm['X_umap']
    labels = adata_sub.obs[color_col].astype(str).values
    unique = sorted(set(labels))

    if palette is None:
        palette = {lb: _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)]
                   for i, lb in enumerate(unique)}

    # ── (A) Combined UMAP with KDE contours & centroid labels ────────────────
    import matplotlib.patheffects as _pe
    fig, ax = plt.subplots(figsize=(12, 10))
    # Light grey background for all cells
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c='#EBEBEB',
               alpha=0.4, linewidths=0, rasterized=True, zorder=1)
    for lb in unique:
        m = labels == lb
        ax.scatter(coords[m, 0], coords[m, 1], s=8, c=[palette[lb]],
                   alpha=0.75, linewidths=0, rasterized=True, zorder=2)
        _add_kde_contour(ax, coords[m], palette[lb])
        # Centroid label: white text with colored stroke (no box clutter)
        cx, cy = float(np.median(coords[m, 0])), float(np.median(coords[m, 1]))
        ax.text(cx, cy, lb, fontsize=14,
                ha='center', va='center', color='white', zorder=5,
                path_effects=[_pe.withStroke(linewidth=3,
                              foreground=palette.get(lb, '#333333'))])

    ax.set_xlabel('UMAP 1', fontsize=20)
    ax.set_ylabel('UMAP 2', fontsize=20)
    ax.set_title(title, fontsize=26)
    handles = [mpatches.Patch(color=palette[lb], label=lb) for lb in unique]
    ax.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.02, 1),
              loc='upper left', frameon=True, framealpha=0.9,
              fontsize=14, title_fontsize=16, markerscale=3,
              edgecolor='#CCCCCC')
    sns.despine(ax=ax)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    save_fig(os.path.join(out_dir, f"{fname_prefix}_umap_combined"))
    plt.close()
    print(f"    Saved: {fname_prefix}_umap_combined")

    # ── (B) Per-treatment (horizontal 4 panels) ──────────────────────────────
    treats_present = [t for t in treat_order
                      if t in adata_sub.obs[treat_col].unique()]
    n = len(treats_present)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, treat in zip(axes, treats_present):
        t_mask = adata_sub.obs[treat_col].values == treat
        ax.scatter(coords[:, 0], coords[:, 1], s=2, c='#DDDDDD',
                   alpha=0.3, linewidths=0, rasterized=True)
        for lb in unique:
            m = t_mask & (labels == lb)
            if m.sum() == 0:
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=6, c=[palette[lb]],
                       alpha=0.80, linewidths=0, rasterized=True)
            _add_kde_contour(ax, coords[m], palette[lb], lw=1.5)
            # Per-treatment centroid labels
            if m.sum() >= 10:
                cx = float(np.median(coords[m, 0]))
                cy = float(np.median(coords[m, 1]))
                ax.text(cx, cy, lb, fontsize=12,
                        ha='center', va='center', color='white', zorder=5,
                        path_effects=[_pe.withStroke(linewidth=2.5,
                                      foreground=palette.get(lb, '#333333'))])
        ax.set_title(treat, fontsize=22)
        ax.set_xlabel('UMAP 1', fontsize=17)
        sns.despine(ax=ax)
    axes[0].set_ylabel('UMAP 2', fontsize=17)

    handles = [mpatches.Patch(color=palette[lb], label=lb) for lb in unique]
    fig.legend(handles=handles, title='Cluster',
               bbox_to_anchor=(1.01, 0.5), loc='center left',
               frameon=True, framealpha=0.9, fontsize=13,
               title_fontsize=14, markerscale=3, edgecolor='#CCCCCC')
    fig.suptitle(f'{title} — per Treatment', fontsize=26)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    save_fig(os.path.join(out_dir, f"{fname_prefix}_umap_per_treatment"))
    plt.close()
    print(f"    Saved: {fname_prefix}_umap_per_treatment")


def add_paga_trajectory(adata_sub, cluster_key, out_dir, fname_prefix, palette=None):
    """
    Two-panel PAGA figure:
      Left:  UMAP scatter + PAGA trajectory arrows overlaid at cluster centroids
             (curved FancyArrowPatch, width ∝ connectivity).
      Right: PAGA abstracted graph, nodes placed with NetworkX spring layout,
             directed arrows with arrowheads (FancyArrowPatch), node size ∝ cell count.
    """
    if 'X_umap' not in adata_sub.obsm:
        return
    try:
        import networkx as nx
        from matplotlib.patches import FancyArrowPatch

        if not hasattr(adata_sub.obs[cluster_key], 'cat'):
            adata_sub.obs[cluster_key] = adata_sub.obs[cluster_key].astype('category')
        sc.tl.paga(adata_sub, groups=cluster_key)

        coords  = adata_sub.obsm['X_umap']
        labels  = adata_sub.obs[cluster_key].astype(str).values
        cats    = list(adata_sub.obs[cluster_key].cat.categories)
        unique  = sorted(set(labels))
        n_total = len(labels)

        if palette is None:
            palette = {lb: _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)]
                       for i, lb in enumerate(unique)}

        conn = adata_sub.uns['paga']['connectivities'].toarray()
        centroids   = {lb: coords[labels == lb].mean(axis=0)
                       for lb in cats if (labels == lb).sum() > 0}
        cell_counts = {lb: int((labels == lb).sum()) for lb in cats}
        max_count   = max(cell_counts.values())

        threshold = 0.05

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

        # ── LEFT: UMAP scatter + PAGA trajectory arrows overlaid ─────────────
        ax1.scatter(coords[:, 0], coords[:, 1], s=1, c='#EBEBEB',
                    alpha=0.4, linewidths=0, rasterized=True, zorder=1)
        for lb in unique:
            m = labels == lb
            ax1.scatter(coords[m, 0], coords[m, 1], s=5, c=[palette[lb]],
                        alpha=0.65, linewidths=0, rasterized=True, zorder=2)
            _add_kde_contour(ax1, coords[m], palette[lb])

        # Draw PAGA trajectory arrows between centroids on UMAP
        for i, c1 in enumerate(cats):
            for j, c2 in enumerate(cats):
                if i >= j:
                    continue
                w = float(conn[i, j])
                if w < threshold or c1 not in centroids or c2 not in centroids:
                    continue
                lw     = 1.5 + w * 8
                alpha  = min(0.9, 0.3 + w * 1.5)
                # Bidirectional curved arrow (PAGA is undirected but we show both ends)
                arw = FancyArrowPatch(
                    posA=tuple(centroids[c1]), posB=tuple(centroids[c2]),
                    arrowstyle='-|>', mutation_scale=18,
                    linewidth=lw, color='#333333', alpha=alpha,
                    connectionstyle='arc3,rad=0.25',
                    zorder=3, shrinkA=12, shrinkB=12)
                ax1.add_patch(arw)
                # reverse arrow
                arw2 = FancyArrowPatch(
                    posA=tuple(centroids[c2]), posB=tuple(centroids[c1]),
                    arrowstyle='-|>', mutation_scale=18,
                    linewidth=lw, color='#333333', alpha=alpha,
                    connectionstyle='arc3,rad=0.25',
                    zorder=3, shrinkA=12, shrinkB=12)
                ax1.add_patch(arw2)
                # connectivity label at midpoint
                mx = (centroids[c1][0] + centroids[c2][0]) / 2
                my = (centroids[c1][1] + centroids[c2][1]) / 2
                ax1.text(mx, my, f'{w:.2f}', fontsize=11, ha='center', va='center',
                         color='#111111', zorder=7,
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                   alpha=0.7, edgecolor='none'))

        # Centroid labels on top
        for lb in cats:
            if lb not in centroids:
                continue
            cx, cy = centroids[lb]
            col = palette.get(lb, '#888888')
            txt_color = 'white' if _color_is_dark(col) else '#222222'
            ax1.annotate(lb, (cx, cy), fontsize=13,
                         ha='center', va='center', zorder=8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=col,
                                   alpha=0.90, edgecolor='white', linewidth=1.5),
                         color=txt_color)

        ax1.set_xlabel('UMAP 1', fontsize=20)
        ax1.set_ylabel('UMAP 2', fontsize=20)
        ax1.set_title('Cell Subtype UMAP + PAGA Trajectories', fontsize=23)
        handles = [mpatches.Patch(color=palette[lb], label=lb) for lb in unique]
        ax1.legend(handles=handles, title='Subtype', loc='lower right',
                   frameon=True, framealpha=0.92, fontsize=13,
                   title_fontsize=14, markerscale=3, edgecolor='#CCCCCC')
        sns.despine(ax=ax1)

        # ── RIGHT: PAGA network graph, directed arrows (spring layout) ───────
        G = nx.DiGraph()
        for lb in cats:
            if lb in cell_counts:
                G.add_node(lb, count=cell_counts[lb])
        for i, c1 in enumerate(cats):
            for j, c2 in enumerate(cats):
                if i == j:
                    continue
                w = float(conn[i, j])
                if w >= threshold:
                    G.add_edge(c1, c2, weight=w)

        np.random.seed(42)
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, weight='weight', k=2.8,
                                   iterations=300, seed=42)
        else:
            pos = {}

        # Draw directed arrows between nodes using FancyArrowPatch
        # Use alternating curvature to separate bidirectional pairs
        drawn = set()
        for (c1, c2, d) in G.edges(data=True):
            w = d['weight']
            key  = tuple(sorted([c1, c2]))
            rad  = 0.25 if key not in drawn else -0.25
            drawn.add(key)
            lw    = 1.5 + w * 9
            alpha = min(0.88, 0.3 + w * 1.6)
            arw = FancyArrowPatch(
                posA=pos[c1], posB=pos[c2],
                arrowstyle='-|>', mutation_scale=22,
                linewidth=lw, color='#333333', alpha=alpha,
                connectionstyle=f'arc3,rad={rad}',
                zorder=2, shrinkA=28, shrinkB=28)
            ax2.add_patch(arw)
            # Weight label at midpoint
            mx = (pos[c1][0] + pos[c2][0]) / 2
            my = (pos[c1][1] + pos[c2][1]) / 2
            ax2.text(mx, my, f'{w:.2f}', fontsize=12, ha='center', va='center',
                     color='#222222', zorder=4,
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               alpha=0.80, edgecolor='none'))

        # Nodes
        for lb in G.nodes():
            cx, cy = pos[lb]
            node_r = 1500 + 4000 * (cell_counts.get(lb, 0) / max_count)
            col    = palette.get(lb, '#888888')
            ax2.scatter([cx], [cy], s=node_r, c=[col],
                        edgecolors='white', linewidths=3,
                        zorder=3, alpha=0.95)
            pct = 100 * cell_counts.get(lb, 0) / n_total
            txt_color = 'white' if _color_is_dark(col) else '#222222'
            ax2.annotate(f'{lb}\n{pct:.0f}%', (cx, cy),
                         fontsize=13,
                         ha='center', va='center', zorder=5,
                         color=txt_color)

        # Expand axis limits so arrows aren't clipped
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            pad = 0.4
            ax2.set_xlim(min(xs) - pad, max(xs) + pad)
            ax2.set_ylim(min(ys) - pad, max(ys) + pad)

        ax2.set_title('PAGA — Cell State Transition Graph\n'
                      '(arrows = connectivity strength, node size ∝ cell count)',
                      fontsize=20)
        ax2.axis('off')

        # Node size legend
        for pct_leg, lbl_leg in [(25, '25% of cells'), (75, '75% of cells')]:
            s_val = 1500 + 4000 * (pct_leg / 100)
            ax2.scatter([], [], s=s_val, c='#AAAAAA', alpha=0.7,
                        edgecolors='white', linewidths=2, label=lbl_leg)
        ax2.legend(title='Node size', loc='lower right',
                   frameon=True, framealpha=0.9,
                   fontsize=13, title_fontsize=14, edgecolor='#CCCCCC')

        fig.suptitle(f'{fname_prefix.replace("_", " ").upper()}  —  PAGA Trajectories',
                     fontsize=29, y=1.01)
        plt.tight_layout()
        save_fig(os.path.join(out_dir, f"{fname_prefix}_umap_paga"))
        plt.close()
        print(f"    Saved: {fname_prefix}_umap_paga")
    except Exception as e:
        print(f"    WARNING: PAGA trajectory failed: {e}")
        import traceback; traceback.print_exc()
        plt.close('all')


def add_paga_trajectory_per_treatment(adata_sub, cluster_key, out_dir,
                                       fname_prefix, palette=None,
                                       treat_col='treatment',
                                       treat_order=None):
    """
    Generate one individual 2-panel figure per treatment, identical layout to
    add_paga_trajectory() (UMAP+arrows on left, state diagram on right).
    Saved as {fname_prefix}_umap_paga_{treatment}.png/pdf for each treatment.
    """
    if 'X_umap' not in adata_sub.obsm:
        print(f"    SKIP per-treatment PAGA: no X_umap in {fname_prefix}")
        return
    if treat_order is None:
        treat_order = TREATMENT_ORDER

    coords = adata_sub.obsm['X_umap']
    labels = adata_sub.obs[cluster_key].astype(str).values
    cats   = sorted(set(labels))
    treats = [t for t in treat_order
              if t in adata_sub.obs[treat_col].unique()]

    if palette is None:
        palette = {lb: _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)]
                   for i, lb in enumerate(cats)}

    for treat in treats:
        try:
            from matplotlib.patches import FancyArrowPatch
            import networkx as nx

            t_mask   = adata_sub.obs[treat_col].values == treat
            a_t      = adata_sub[t_mask].copy()
            n_cells  = int(t_mask.sum())
            t_coords = coords[t_mask]
            t_labels = labels[t_mask]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

            # ── LEFT: UMAP scatter (grey bg) + treatment cells + PAGA arrows ──
            # Grey background: all cells in subsample
            ax1.scatter(coords[:, 0], coords[:, 1], s=1, c='#EBEBEB',
                        alpha=0.4, linewidths=0, rasterized=True, zorder=1)

            for lb in cats:
                m = t_labels == lb
                if m.sum() == 0:
                    continue
                ax1.scatter(t_coords[m, 0], t_coords[m, 1],
                            s=5, c=[palette[lb]], alpha=0.65,
                            linewidths=0, rasterized=True, zorder=2)
                _add_kde_contour(ax1, t_coords[m], palette[lb])

            # Run PAGA on treatment subset
            conn       = None
            t_cats     = []
            t_centroids = {}
            cell_counts = {lb: int((t_labels == lb).sum()) for lb in cats}
            n_total    = len(t_labels)
            max_count  = max(cell_counts.values()) if cell_counts else 1
            threshold  = 0.05

            if n_cells >= 20:
                try:
                    if not hasattr(a_t.obs[cluster_key], 'cat'):
                        a_t.obs[cluster_key] = a_t.obs[cluster_key].astype('category')
                    n_nb = min(10, n_cells - 1)
                    n_pc = min(20, a_t.n_obs - 1, a_t.n_vars - 1)
                    sc.pp.neighbors(a_t, n_neighbors=n_nb, n_pcs=n_pc)
                    sc.tl.paga(a_t, groups=cluster_key)
                    t_cats = list(a_t.obs[cluster_key].cat.categories)
                    conn   = a_t.uns['paga']['connectivities'].toarray()
                    t_centroids = {lb: t_coords[t_labels == lb].mean(axis=0)
                                   for lb in t_cats if (t_labels == lb).sum() > 0}
                except Exception as pe:
                    print(f"      PAGA computation failed for {treat}: {pe}")

            # Draw PAGA arrows on UMAP (if PAGA succeeded)
            if conn is not None:
                for i, c1 in enumerate(t_cats):
                    for j, c2 in enumerate(t_cats):
                        if i >= j:
                            continue
                        w = float(conn[i, j])
                        if w < threshold or c1 not in t_centroids or c2 not in t_centroids:
                            continue
                        lw    = 1.5 + w * 8
                        alpha = min(0.9, 0.3 + w * 1.5)
                        for posA, posB in [(t_centroids[c1], t_centroids[c2]),
                                           (t_centroids[c2], t_centroids[c1])]:
                            arw = FancyArrowPatch(
                                posA=tuple(posA), posB=tuple(posB),
                                arrowstyle='-|>', mutation_scale=18,
                                linewidth=lw, color='#333333', alpha=alpha,
                                connectionstyle='arc3,rad=0.25',
                                zorder=3, shrinkA=12, shrinkB=12)
                            ax1.add_patch(arw)
                        mx = (t_centroids[c1][0] + t_centroids[c2][0]) / 2
                        my = (t_centroids[c1][1] + t_centroids[c2][1]) / 2
                        ax1.text(mx, my, f'{w:.2f}', fontsize=11,
                                 ha='center', va='center', color='#111111', zorder=7,
                                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                           alpha=0.7, edgecolor='none'))

            # Centroid labels
            for lb in cats:
                m = t_labels == lb
                if m.sum() < 5:
                    continue
                cx, cy = t_coords[m, 0].mean(), t_coords[m, 1].mean()
                col = palette.get(lb, '#888888')
                txt_color = 'white' if _color_is_dark(col) else '#222222'
                ax1.annotate(lb, (cx, cy), fontsize=13,
                             ha='center', va='center', zorder=8,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=col,
                                       alpha=0.90, edgecolor='white', linewidth=1.5),
                             color=txt_color)

            ax1.set_xlabel('UMAP 1', fontsize=20)
            ax1.set_ylabel('UMAP 2', fontsize=20)
            ax1.set_title(f'Cell Subtype UMAP + PAGA Trajectories\n{treat}  (n={n_cells:,})',
                          fontsize=23)
            handles = [mpatches.Patch(color=palette[lb], label=lb)
                       for lb in cats if (t_labels == lb).sum() > 0]
            ax1.legend(handles=handles, title='Subtype', loc='lower right',
                       frameon=True, framealpha=0.92, fontsize=13,
                       title_fontsize=14, markerscale=3, edgecolor='#CCCCCC')
            sns.despine(ax=ax1)

            # ── RIGHT: PAGA network graph (state diagram) ─────────────────────
            if conn is not None and len(t_cats) > 0:
                G = nx.DiGraph()
                for lb in t_cats:
                    if cell_counts.get(lb, 0) > 0:
                        G.add_node(lb, count=cell_counts[lb])
                for i, c1 in enumerate(t_cats):
                    for j, c2 in enumerate(t_cats):
                        if i == j:
                            continue
                        w = float(conn[i, j])
                        if w >= threshold:
                            G.add_edge(c1, c2, weight=w)

                np.random.seed(42)
                pos = nx.spring_layout(G, weight='weight', k=2.8,
                                       iterations=300, seed=42) if len(G.nodes) > 0 else {}

                drawn = set()
                for (c1, c2, d) in G.edges(data=True):
                    w    = d['weight']
                    key  = tuple(sorted([c1, c2]))
                    rad  = 0.25 if key not in drawn else -0.25
                    drawn.add(key)
                    lw    = 1.5 + w * 9
                    alpha = min(0.88, 0.3 + w * 1.6)
                    arw = FancyArrowPatch(
                        posA=pos[c1], posB=pos[c2],
                        arrowstyle='-|>', mutation_scale=22,
                        linewidth=lw, color='#333333', alpha=alpha,
                        connectionstyle=f'arc3,rad={rad}',
                        zorder=2, shrinkA=28, shrinkB=28)
                    ax2.add_patch(arw)
                    mx = (pos[c1][0] + pos[c2][0]) / 2
                    my = (pos[c1][1] + pos[c2][1]) / 2
                    ax2.text(mx, my, f'{w:.2f}', fontsize=12,
                             ha='center', va='center', color='#222222', zorder=4,
                             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                       alpha=0.80, edgecolor='none'))

                for lb in G.nodes():
                    cx, cy  = pos[lb]
                    node_r  = 1500 + 4000 * (cell_counts.get(lb, 0) / max_count)
                    col     = palette.get(lb, '#888888')
                    ax2.scatter([cx], [cy], s=node_r, c=[col],
                                edgecolors='white', linewidths=3,
                                zorder=3, alpha=0.95)
                    pct = 100 * cell_counts.get(lb, 0) / max(n_total, 1)
                    txt_color = 'white' if _color_is_dark(col) else '#222222'
                    ax2.annotate(f'{lb}\n{pct:.0f}%', (cx, cy),
                                 fontsize=13,
                                 ha='center', va='center', zorder=5,
                                 color=txt_color)

                if pos:
                    xs = [p[0] for p in pos.values()]
                    ys = [p[1] for p in pos.values()]
                    pad = 0.4
                    ax2.set_xlim(min(xs) - pad, max(xs) + pad)
                    ax2.set_ylim(min(ys) - pad, max(ys) + pad)

                for pct_leg, lbl_leg in [(25, '25% of cells'), (75, '75% of cells')]:
                    s_val = 1500 + 4000 * (pct_leg / 100)
                    ax2.scatter([], [], s=s_val, c='#AAAAAA', alpha=0.7,
                                edgecolors='white', linewidths=2, label=lbl_leg)
                ax2.legend(title='Node size', loc='lower right',
                           frameon=True, framealpha=0.9,
                           fontsize=13, title_fontsize=14, edgecolor='#CCCCCC')
            else:
                ax2.text(0.5, 0.5, 'PAGA not available\n(insufficient cells)',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=19)

            ax2.set_title('PAGA — Cell State Transition Graph\n'
                          '(arrows = connectivity strength, node size ∝ cell count)',
                          fontsize=20)
            ax2.axis('off')

            fig.suptitle(
                f'{fname_prefix.replace("_", " ").upper()}, PAGA Trajectories  |  {treat}',
                fontsize=29, y=1.01)
            plt.tight_layout()

            # Save as individual file per treatment (safe filename)
            safe_treat = treat.replace('+', '_plus_').replace(' ', '_')
            save_fig(os.path.join(out_dir, f"{fname_prefix}_umap_paga_{safe_treat}"))
            plt.close()
            print(f"    Saved: {fname_prefix}_umap_paga_{safe_treat}")

        except Exception as e:
            print(f"    WARNING: per-treatment PAGA failed for {treat}: {e}")
            import traceback; traceback.print_exc()
            plt.close('all')


# =============================================================================
#  SECTION 1: PDAC
# =============================================================================

def run_pdac(adata_full, out_dir):
    print("\n" + "=" * 65)
    print("SECTION 1: PDAC  (mPDAC + ePDAC combined)")
    print("=" * 65)

    mask = adata_full.obs['cell_type_auto'].isin(['mPDAC', 'ePDAC'])
    if mask.sum() < 50:
        print("  SKIP: fewer than 50 PDAC cells")
        return

    adata = adata_full[mask].copy()
    adata.obs['cell_type'] = 'PDAC'
    print(f"  Cells: {adata.n_obs:,}  (mPDAC + ePDAC)")

    treats = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    adata.obs['treatment'] = pd.Categorical(
        adata.obs['treatment'], categories=treats, ordered=True)

    dotplot_specs = [
        (PDAC_SURVIVAL,       'PDAC: Survival Pathway Geneset',    'pdac_survival'),
        (PDAC_MESENCHYMAL,    'PDAC: Mesenchymal Gene Signature',  'pdac_mesenchymal'),
        (PDAC_EMT_TF,         'PDAC: EMT Transcription Factors',   'pdac_emt_tf'),
        (PDAC_RAS,            'PDAC: RAS Family Genes',            'pdac_ras'),
        (PDAC_INSULIN,        'PDAC: Insulin Signalling Pathway',  'pdac_insulin'),
        (PDAC_CYTOKINES,      'PDAC: Cytokine Profile',            'pdac_cytokines'),
        (PDAC_GROWTH_FACTORS, 'PDAC: Growth Factors Profile',      'pdac_growth_factors'),
        (PDAC_INVASION,       'PDAC: Invasion Markers',            'pdac_invasion'),
        (PDAC_STEMNESS,       'PDAC: Stemness Profile',            'pdac_stemness'),
    ]
    for genes, title, fname in dotplot_specs:
        make_dotplot(adata, genes, 'treatment', title,
                     os.path.join(out_dir, fname), swap_axes=True)

    print("  Violin plots...")
    make_violin_plots(adata, PDAC_VIOLIN_GENES, 'treatment', 'pdac', out_dir,
                      order=treats)
    print(f"  ✓ PDAC done → {out_dir}")


# =============================================================================
#  SECTION 2: CAFs
# =============================================================================

def run_caf(adata_full, out_dir):
    print("\n" + "=" * 65)
    print("SECTION 2: CAFs")
    print("=" * 65)
    os.makedirs(out_dir, exist_ok=True)

    caf_present = [c for c in CAF_SUBTYPES
                   if c in adata_full.obs['cell_type_auto'].unique()]
    if not caf_present:
        print("  SKIP: no CAF cell types found")
        return

    mask  = adata_full.obs['cell_type_auto'].isin(caf_present)
    adata = adata_full[mask].copy()
    adata.obs['cell_type_auto'] = pd.Categorical(
        adata.obs['cell_type_auto'], categories=caf_present, ordered=True)
    treats = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    adata.obs['treatment'] = pd.Categorical(
        adata.obs['treatment'], categories=treats, ordered=True)
    print(f"  Cells: {adata.n_obs:,}  subtypes: {caf_present}")

    # ── CAF subtype marker dotplot (grouped by subtype) ───────────────────────
    subtype_genes = list(dict.fromkeys(
        g for ct in caf_present for g in CAF_SUBTYPE_MARKERS.get(ct, [])
    ))
    make_dotplot(adata, subtype_genes, 'cell_type_auto',
                 'CAF Subtypes, Marker Genes',
                 os.path.join(out_dir, 'caf_subtype_markers'),
                 swap_axes=True)

    # ── Pathway dotplots (by treatment) ───────────────────────────────────────
    pathway_specs = [
        (CAF_AUTOPHAGY,  'CAFs, Autophagy Genes',   'caf_autophagy'),
        (CAF_ER_STRESS,  'CAFs, ER-Stress Genes',   'caf_er_stress'),
        (CAF_ASPARTATE,  'CAFs, Aspartate Pathway', 'caf_aspartate'),
    ]
    for genes, title, fname in pathway_specs:
        make_dotplot(adata, genes, 'treatment', title,
                     os.path.join(out_dir, fname), swap_axes=True)

    # ── Violin plots ─────────────────────────────────────────────────────────
    make_violin_plots(adata, CAF_VIOLIN_GENES, 'treatment', 'caf', out_dir,
                      order=treats)

    # ── Barplots ─────────────────────────────────────────────────────────────
    make_barplots(adata, 'cell_type_auto', caf_present, out_dir, 'CAF',
                  treat_order=treats)

    # ── UMAP + PAGA ──────────────────────────────────────────────────────────
    print("  Computing CAF UMAP...")
    try:
        adata_u = prep_subset_for_umap(adata, umap_min_dist=0.1)
        # Re-attach treatment + subtype from original obs (after subsample)
        CAF_FIXED_COLORS = {
            'myCAFs': '#7B1FA2',   # deep purple
            'iCAF':   '#F9A825',   # amber/gold
            'apCAFs': '#E91E63',   # vivid pink
            'qCAF':   '#795548',   # brown
        }
        pal = {ct: CAF_FIXED_COLORS.get(ct, _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)])
               for i, ct in enumerate(caf_present)}
        make_subset_umaps(adata_u, 'cell_type_auto', 'CAF Subtypes',
                          out_dir, 'caf', palette=pal, treat_order=treats)
        add_paga_trajectory(adata_u, 'cell_type_auto', out_dir, 'caf',
                            palette=pal)
        add_paga_trajectory_per_treatment(adata_u, 'cell_type_auto', out_dir, 'caf',
                                          palette=pal, treat_order=treats)
    except Exception as e:
        print(f"  WARNING: CAF UMAP failed: {e}")

    print(f"  ✓ CAF done → {out_dir}")


# =============================================================================
#  SECTION 3: TAMs / Macrophages
# =============================================================================

def run_tam(adata_full, out_dir):
    print("\n" + "=" * 65)
    print("SECTION 3: TAMs / Macrophages")
    print("=" * 65)
    os.makedirs(out_dir, exist_ok=True)

    tam_present = [c for c in TAM_SUBTYPES
                   if c in adata_full.obs['cell_type_auto'].unique()]
    if not tam_present:
        print("  SKIP: no Macrophage cell types found")
        return

    mask  = adata_full.obs['cell_type_auto'].isin(tam_present)
    adata = adata_full[mask].copy()
    adata.obs['cell_type_auto'] = pd.Categorical(
        adata.obs['cell_type_auto'], categories=tam_present, ordered=True)
    treats = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    adata.obs['treatment'] = pd.Categorical(
        adata.obs['treatment'], categories=treats, ordered=True)
    print(f"  Cells: {adata.n_obs:,}  subtypes: {tam_present}")

    # ── TAM subtype marker dotplot ────────────────────────────────────────────
    make_dotplot(adata, TAM_MARKERS, 'cell_type_auto',
                 'TAM Subtypes, Marker Genes',
                 os.path.join(out_dir, 'tam_marker_dotplot'), swap_axes=True)

    # ── M1/M2 markers dotplot (by treatment) ─────────────────────────────────
    make_dotplot(adata, TAM_M1M2_MARKERS, 'treatment',
                 'TAMs, M1/M2 Markers per Treatment',
                 os.path.join(out_dir, 'tam_m1m2_treatment_dotplot'),
                 swap_axes=False)

    # ── Mrc1 ridge plot ───────────────────────────────────────────────────────
    for gene in TAM_RIDGEPLOT_GENES:
        make_ridgeplot(adata, gene, 'treatment', out_dir, order=treats)

    # ── Barplots ─────────────────────────────────────────────────────────────
    make_barplots(adata, 'cell_type_auto', tam_present, out_dir, 'TAM',
                  treat_order=treats)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("  Computing TAM UMAP...")
    try:
        adata_u = prep_subset_for_umap(adata, umap_min_dist=0.1)
        # High-contrast fixed palette for M1/M2 distinction
        TAM_FIXED_COLORS = {
            'M1 Macrophage': '#7CB342',   # lime green
            'M2 Macrophage': '#E53935',   # vivid red
        }
        pal = {ct: TAM_FIXED_COLORS.get(ct, _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)])
               for i, ct in enumerate(tam_present)}
        make_subset_umaps(adata_u, 'cell_type_auto', 'Macrophage Subtypes',
                          out_dir, 'tam', palette=pal, treat_order=treats)
    except Exception as e:
        print(f"  WARNING: TAM UMAP failed: {e}")

    print(f"  ✓ TAM done → {out_dir}")


# =============================================================================
#  SECTION 4: T / NK cells
# =============================================================================

def run_tcell(adata_full, out_dir):
    print("\n" + "=" * 65)
    print("SECTION 4: T / NK cells")
    print("=" * 65)
    os.makedirs(out_dir, exist_ok=True)

    t_present = [c for c in TCELL_SUBTYPES
                 if c in adata_full.obs['cell_type_auto'].unique()]
    if not t_present:
        print("  SKIP: no T/NK cell types found")
        return

    mask  = adata_full.obs['cell_type_auto'].isin(t_present)
    adata = adata_full[mask].copy()
    adata.obs['cell_type_auto'] = pd.Categorical(
        adata.obs['cell_type_auto'], categories=t_present, ordered=True)
    treats = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    adata.obs['treatment'] = pd.Categorical(
        adata.obs['treatment'], categories=treats, ordered=True)
    print(f"  Cells: {adata.n_obs:,}  subtypes: {t_present}")

    # ── T cell subtype marker dotplot ─────────────────────────────────────────
    make_dotplot(adata, TCELL_MARKERS, 'cell_type_auto',
                 'T / NK Cells, Marker Genes',
                 os.path.join(out_dir, 'tcell_marker_dotplot'), swap_axes=True)

    # ── Effector marker dotplot (by treatment) ────────────────────────────────
    make_dotplot(adata, TCELL_EFFECTOR, 'treatment',
                 'T Cells, Effector Markers per Treatment',
                 os.path.join(out_dir, 'tcell_effector_dotplot'), swap_axes=True)

    # ── Barplots ─────────────────────────────────────────────────────────────
    make_barplots(adata, 'cell_type_auto', t_present, out_dir, 'T/NK',
                  treat_order=treats)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("  Computing T/NK UMAP...")
    try:
        adata_u = prep_subset_for_umap(adata, umap_min_dist=0.1)
        # Hand-curated maximally distinct palette matching the main annotation palette
        TCELL_FIXED_COLORS = {
            'CD4 T cells':          '#1565C0',   # royal blue
            'CD8 T cells':          '#0097A7',   # teal
            'Effector CD4+ T cells': '#2E7D32',  # forest green
            'Cytotoxic T cells':    '#00BFA5',   # cyan/mint
            'Tregs':                '#E040FB',   # vivid orchid
            'NK cells':             '#FF6F00',   # amber
        }
        pal = {ct: TCELL_FIXED_COLORS.get(ct, _SUBTYPE_COLORS[i % len(_SUBTYPE_COLORS)])
               for i, ct in enumerate(t_present)}
        make_subset_umaps(adata_u, 'cell_type_auto', 'T / NK Cells',
                          out_dir, 'tcell', palette=pal, treat_order=treats)
    except Exception as e:
        print(f"  WARNING: T/NK UMAP failed: {e}")

    print(f"  ✓ T/NK done → {out_dir}")


# =============================================================================
#  SECTION 5: Endothelial (optional)
# =============================================================================

def run_endothelial(adata_full, out_dir):
    print("\n" + "=" * 65)
    print("SECTION 5: Endothelial cells")
    print("=" * 65)

    endo_types = [c for c in adata_full.obs['cell_type_auto'].unique()
                  if 'ndothel' in c.lower()]
    if not endo_types:
        print("  SKIP: no Endothelial cell types in this dataset")
        return

    mask  = adata_full.obs['cell_type_auto'].isin(endo_types)
    adata = adata_full[mask].copy()
    treats = [t for t in TREATMENT_ORDER if t in adata.obs['treatment'].unique()]
    adata.obs['treatment'] = pd.Categorical(
        adata.obs['treatment'], categories=treats, ordered=True)
    print(f"  Cells: {adata.n_obs:,}")

    dotplot_specs = [
        (ENDO_CYTOKINES,      'Endothelial, Cytokine Profile',          'endo_cytokines'),
        (ENDO_GROWTH_FACTORS, 'Endothelial, Growth Factor Genes',       'endo_growth_factors'),
        (ENDO_ER_STRESS,      'Endothelial, ER-Stress Genes',           'endo_er_stress'),
        (ENDO_ANGIOGENIC,     'Endothelial, Angiogenic Pathway Markers','endo_angiogenic'),
    ]
    for genes, title, fname in dotplot_specs:
        make_dotplot(adata, genes, 'treatment', title,
                     os.path.join(out_dir, fname), swap_axes=True)

    print(f"  ✓ Endothelial done → {out_dir}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Publication-quality cell-subtype plots')
    parser.add_argument(
        '--h5ad_path',
        default=os.path.join(os.path.expanduser('~'), 'stereo-seq', 'stereoseq-analysis', 'downstream_analysis', 'processed_data', 'merged_annotated.h5ad'),
        help='Path to merged_annotated.h5ad')
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.expanduser('~'), 'stereo-seq', 'stereoseq-analysis', 'downstream_analysis', 'figures', 'post6_subtype_plots'),
        help='Root output directory')
    parser.add_argument(
        '--sections', nargs='+',
        default=['pdac', 'caf', 'tam', 'tcell', 'endo'],
        choices=['pdac', 'caf', 'tam', 'tcell', 'endo'],
        help='Which sections to run (default: all)')
    args = parser.parse_args()

    if not os.path.exists(args.h5ad_path):
        print(f"ERROR: h5ad not found: {args.h5ad_path}")
        sys.exit(1)

    print("=" * 65)
    print("STEP 2 POST-6: CELL SUBTYPE PLOTS")
    print("=" * 65)
    print(f"h5ad:     {args.h5ad_path}")
    print(f"out_dir:  {args.out_dir}")
    print(f"sections: {args.sections}")

    print("\nLoading h5ad...")
    adata = sc.read_h5ad(args.h5ad_path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    print(f"  Cell types: {sorted(adata.obs['cell_type_auto'].unique())}")
    print(f"  Treatments: {sorted(adata.obs['treatment'].unique())}")
    print(f"  adata.raw available: {adata.raw is not None}")

    os.makedirs(args.out_dir, exist_ok=True)

    RUNNERS = {
        'pdac':  (run_pdac,         '01_pdac'),
        'caf':   (run_caf,          '02_caf'),
        'tam':   (run_tam,          '03_tam'),
        'tcell': (run_tcell,        '04_tcell'),
        'endo':  (run_endothelial,  '05_endothelial'),
    }
    for key in args.sections:
        fn, subdir = RUNNERS[key]
        fn(adata, os.path.join(args.out_dir, subdir))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    base_dir = args.out_dir.split('downstream_analysis')[0] or '.'
    collect(base_dir, {
        os.path.join('downstream_analysis', 'figures', 'post6_subtype_plots', '02_caf', 'CAF_proportion_barplot'): 'fig7_c',
        os.path.join('downstream_analysis', 'figures', 'post6_subtype_plots', '03_tam', 'TAM_proportion_barplot'): 'fig7_d',
        os.path.join('downstream_analysis', 'figures', 'post6_subtype_plots', '04_tcell', 'T_NK_proportion_barplot'): 'fig7_e',
        os.path.join('downstream_analysis', 'figures', 'post6_subtype_plots', '02_caf', 'caf_subtype_markers'): 'suppl12',
    })

    print("\n" + "=" * 65)
    print("ALL SECTIONS COMPLETE")
    print("=" * 65)
    print(f"\nFigures saved to: {args.out_dir}")
    print("\nSubdirectories:")
    for key in args.sections:
        _, subdir = RUNNERS[key]
        p = os.path.join(args.out_dir, subdir)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if f.endswith('.png')])
            print(f"  {subdir}: {n} PNG files")


if __name__ == '__main__':
    main()
