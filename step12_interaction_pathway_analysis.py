#!/usr/bin/env python3
"""
Cell-Cell Interaction & Pathway Enrichment Analysis

Run after step02_build_annotated_h5ad.py completes.

Input: merged_annotated.h5ad from Step 2
Output: CellChat interaction networks, pathway enrichment plots

Core analyses:
  1. Cell-Cell Interaction (CellPhoneDB-style analysis)
  2. Pathway Enrichment (GO/KEGG/GSEA)
  3. Treatment-specific interaction rewiring
  4. Spatial interaction neighborhoods

Usage:
  python step12_interaction_pathway_analysis.py \
    --input_dir /path/to/outputs \
    --output_dir /path/to/outputs \
    --species mouse
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
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu
from itertools import combinations
import json
from matplotlib.patches import Arc, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set publication-quality defaults
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

# =====================================================================
# CONFIGURATION
# =====================================================================
TREATMENT_ORDER = ['Sham', 'IT', 'GPH', 'GPH+IT']
TREATMENT_COLORS = {
    'Sham': '#2166AC', 'IT': '#EF8A62', 'GPH': '#67A9CF', 'GPH+IT': '#B2182B'
}

# Mouse ligand-receptor pairs (curated from CellPhoneDB/CellChatDB/NATMI)
# Format: (ligand, receptor, pathway/annotation)
LR_PAIRS_MOUSE = [
    # Growth factors & cytokines
    ('Tgfb1', 'Tgfbr1', 'TGFb_signaling'),
    ('Tgfb1', 'Tgfbr2', 'TGFb_signaling'),
    ('Pdgfa', 'Pdgfra', 'PDGF_signaling'),
    ('Pdgfb', 'Pdgfrb', 'PDGF_signaling'),
    ('Vegfa', 'Kdr', 'VEGF_signaling'),
    ('Vegfa', 'Flt1', 'VEGF_signaling'),
    ('Fgf2', 'Fgfr1', 'FGF_signaling'),
    ('Fgf7', 'Fgfr2', 'FGF_signaling'),
    ('Egf', 'Egfr', 'EGF_signaling'),
    ('Hbegf', 'Egfr', 'EGF_signaling'),
    
    # Immune checkpoints & co-stimulation
    ('Cd274', 'Pdcd1', 'PD1_PDL1'),
    ('Pdcd1lg2', 'Pdcd1', 'PD1_PDL2'),
    ('Cd80', 'Ctla4', 'CD80_CTLA4'),
    ('Cd86', 'Ctla4', 'CD86_CTLA4'),
    ('Cd80', 'Cd28', 'CD80_CD28'),
    ('Cd86', 'Cd28', 'CD86_CD28'),
    ('Tnfsf9', 'Tnfrsf9', '41BB_signaling'),
    ('Tnfsf4', 'Tnfrsf4', 'OX40_signaling'),
    ('Cd70', 'Cd27', 'CD70_CD27'),
    ('Icam1', 'Itgal', 'ICAM1_LFA1'),
    
    # Chemokines
    ('Cxcl12', 'Cxcr4', 'CXCL12_CXCR4'),
    ('Ccl2', 'Ccr2', 'CCL2_CCR2'),
    ('Ccl5', 'Ccr5', 'CCL5_CCR5'),
    ('Cxcl10', 'Cxcr3', 'CXCL10_CXCR3'),
    ('Cxcl9', 'Cxcr3', 'CXCL9_CXCR3'),
    ('Ccl19', 'Ccr7', 'CCL19_CCR7'),
    ('Ccl21a', 'Ccr7', 'CCL21_CCR7'),
    
    # Interleukins
    ('Il6', 'Il6ra', 'IL6_signaling'),
    ('Il6', 'Il6st', 'IL6_signaling'),
    ('Il10', 'Il10ra', 'IL10_signaling'),
    ('Il1b', 'Il1r1', 'IL1_signaling'),
    ('Il2', 'Il2ra', 'IL2_signaling'),
    ('Il15', 'Il15ra', 'IL15_signaling'),
    
    # TNF family
    ('Tnf', 'Tnfrsf1a', 'TNF_signaling'),
    ('Tnf', 'Tnfrsf1b', 'TNF_signaling'),
    ('Fasl', 'Fas', 'FAS_FASL'),
    ('Tnfsf10', 'Tnfrsf10b', 'TRAIL_DR5'),
    
    # Notch signaling
    ('Dll1', 'Notch1', 'Notch_signaling'),
    ('Dll4', 'Notch1', 'Notch_signaling'),
    ('Jag1', 'Notch1', 'Notch_signaling'),
    ('Jag2', 'Notch2', 'Notch_signaling'),
    
    # ECM & adhesion
    ('Fn1', 'Itga5', 'Fibronectin_signaling'),
    ('Col1a1', 'Itga2', 'Collagen_signaling'),
    ('Spp1', 'Itgav', 'SPP1_integrin'),
    ('Thbs1', 'Cd47', 'Thrombospondin'),
    ('Vcam1', 'Itga4', 'VCAM1_VLA4'),
    
    # Semaphorins
    ('Sema3a', 'Nrp1', 'Semaphorin_signaling'),
    ('Sema4d', 'Plxnb1', 'Semaphorin_signaling'),
    
    # Wnt signaling
    ('Wnt5a', 'Fzd5', 'Wnt_signaling'),
    ('Wnt3a', 'Fzd1', 'Wnt_signaling'),
]


# The only GSEA-results figure this script builds that ends up in the paper
# (suppl10_a); the KEGG variant is still computed but not saved.
_CURATED_STEMS = {'fig36_GSEA_Hallmark'}


def setup_dirs(output_dir):
    """Build the output directory paths.

    Only `data` is created here; `interaction` and `gsea` are created
    lazily by the functions that write their curated figures into them.
    `pathway` and `summary` are never used for a curated figure, so they're
    never created.
    """
    base = os.path.join(output_dir, 'downstream_analysis')
    dirs = {
        'interaction': os.path.join(base, 'figures', '11_cellchat_interaction'),
        'pathway': os.path.join(base, 'figures', '12_pathway_enrichment'),
        'gsea': os.path.join(base, 'figures', '13_gsea'),
        'summary': os.path.join(base, 'figures', '14_integration_summary'),
        'data': os.path.join(base, 'processed_data'),
    }
    os.makedirs(dirs['data'], exist_ok=True)
    return dirs


# =====================================================================
# STEP 1: LIGAND-RECEPTOR INTERACTION ANALYSIS (CellPhoneDB-STYLE)
# =====================================================================
def cellchat_interaction_analysis(adata, dirs, species='mouse'):
    """
    CellPhoneDB-style ligand-receptor interaction analysis.
    Tests for significant enrichment of L-R pairs between cell type pairs.
    """
    print("\n" + "="*60)
    print("CELL-CELL INTERACTION ANALYSIS (CellPhoneDB-style)")
    print("="*60)
    
    # Use raw normalized counts for expression
    if adata.raw is not None:
        expr_data = adata.raw.X
        genes = adata.raw.var_names
    else:
        expr_data = adata.X
        genes = adata.var_names
    
    # Filter L-R pairs to those with both genes present
    lr_pairs_available = []
    for lig, rec, pathway in LR_PAIRS_MOUSE:
        if lig in genes and rec in genes:
            lr_pairs_available.append((lig, rec, pathway))
    
    print(f"\n{len(lr_pairs_available)}/{len(LR_PAIRS_MOUSE)} L-R pairs available in data")
    
    if len(lr_pairs_available) < 10:
        print("WARNING: Too few L-R pairs available. Check gene names.")
        return None
    
    # Compute mean expression per cell type
    cell_types = sorted(adata.obs['cell_type_auto'].unique())
    
    expr_by_celltype = {}
    for ct in cell_types:
        mask = (adata.obs['cell_type_auto'] == ct).values
        ct_expr = np.array(expr_data[mask, :].mean(axis=0)).flatten()
        expr_by_celltype[ct] = pd.Series(ct_expr, index=genes)
    
    # Test all L-R interactions between cell type pairs
    results = []
    for sender_ct in cell_types:
        for receiver_ct in cell_types:
            sender_expr = expr_by_celltype[sender_ct]
            receiver_expr = expr_by_celltype[receiver_ct]
            
            for lig, rec, pathway in lr_pairs_available:
                lig_expr = sender_expr[lig]
                rec_expr = receiver_expr[rec]
                
                # Interaction score: geometric mean (common approach)
                score = np.sqrt(lig_expr * rec_expr)
                
                # Require minimum expression in both
                if lig_expr > 0.1 and rec_expr > 0.1:
                    results.append({
                        'sender': sender_ct,
                        'receiver': receiver_ct,
                        'ligand': lig,
                        'receptor': rec,
                        'pathway': pathway,
                        'lig_expr': lig_expr,
                        'rec_expr': rec_expr,
                        'score': score,
                    })
    
    interactions_df = pd.DataFrame(results)
    
    if len(interactions_df) == 0:
        print("No interactions passed filters")
        return None
    
    print(f"\n{len(interactions_df):,} interactions detected")
    
    # Save all interactions
    interactions_df.to_csv(os.path.join(dirs['data'], 'lr_interactions_all.csv'), index=False)
    
    print("\nGenerating interaction visualizations...")
    
    # --- Figure 27: Top interactions heatmap (all treatments combined) ---
    # Aggregate by sender-receiver pair
    interaction_agg = interactions_df.groupby(['sender', 'receiver'])['score'].sum().reset_index()
    pivot = interaction_agg.pivot(index='sender', columns='receiver', values='score').fillna(0)
    
    fig, ax = plt.subplots(figsize=(20, 17), facecolor='white')
    sns.heatmap(pivot, cmap='Reds', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Total Interaction Score'},
                square=True, annot=False, fmt='.1f')
    ax.collections[0].colorbar.set_label('Total Interaction Score', fontsize=24)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    ax.set_title('Cell-Cell Interaction Strength (All Treatments)',
                fontsize=34)
    ax.set_xlabel('Receiver Cell Type', fontsize=30)
    ax.set_ylabel('Sender Cell Type', fontsize=30)
    plt.xticks(rotation=45, ha='right', fontsize=24)
    plt.yticks(rotation=0, fontsize=24)
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['interaction'], 'fig27_interaction_heatmap_all.png'),
    # dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig27 - Interaction heatmap")
    
    # --- Figure 28: CHORD DIAGRAM (NEW!) ---
    plot_chord_diagram(interactions_df, dirs, 
                      'fig28_interaction_chord_all.png',
                      title='Cell-Cell Interaction Chord Diagram (All Treatments)')
    
    # --- Figure 29: Network diagram (top interactions) ---
    plot_interaction_network(interaction_agg, dirs, 
                            'fig29_interaction_network_all.png',
                            title='Cell-Cell Interaction Network (All Treatments)')
    print("  ✓ fig29 - Interaction network")
    
    # --- Figure 30: Ligand-Receptor Dotplot (NEW!) ---
    plot_interaction_dotplot(interactions_df, dirs,
                            'fig30_lr_dotplot_all.png',
                            title='Ligand-Receptor Interaction Dotplot (All Treatments)')
    
    return interactions_df


def cellchat_per_treatment(adata, dirs):
    """Run CellPhoneDB-style analysis per treatment to identify rewiring."""
    print("\n" + "="*60)
    print("TREATMENT-SPECIFIC INTERACTION ANALYSIS")
    print("="*60)
    
    # Use raw normalized counts
    if adata.raw is not None:
        expr_data = adata.raw.X
        genes = adata.raw.var_names
    else:
        expr_data = adata.X
        genes = adata.var_names
    
    # Filter L-R pairs
    lr_pairs_available = []
    for lig, rec, pathway in LR_PAIRS_MOUSE:
        if lig in genes and rec in genes:
            lr_pairs_available.append((lig, rec, pathway))
    
    treatment_interactions = {}
    
    for treatment in TREATMENT_ORDER:
        print(f"\n--- {treatment} ---")
        mask = adata.obs['treatment'] == treatment
        adata_t = adata[mask].copy()
        
        cell_types = sorted(adata_t.obs['cell_type_auto'].unique())
        
        # Mean expression per cell type in this treatment
        expr_by_celltype = {}
        for ct in cell_types:
            ct_mask = (adata_t.obs['cell_type_auto'] == ct).values
            mat = adata_t.raw.X if adata_t.raw else adata_t.X
            ct_expr = np.array(mat[ct_mask, :].mean(axis=0)).flatten()
            expr_by_celltype[ct] = pd.Series(ct_expr, index=genes)
        
        # Compute interactions
        results = []
        for sender_ct in cell_types:
            for receiver_ct in cell_types:
                sender_expr = expr_by_celltype[sender_ct]
                receiver_expr = expr_by_celltype[receiver_ct]
                
                for lig, rec, pathway in lr_pairs_available:
                    lig_expr = sender_expr[lig]
                    rec_expr = receiver_expr[rec]
                    score = np.sqrt(lig_expr * rec_expr)
                    
                    if lig_expr > 0.1 and rec_expr > 0.1:
                        results.append({
                            'treatment': treatment,
                            'sender': sender_ct,
                            'receiver': receiver_ct,
                            'ligand': lig,
                            'receptor': rec,
                            'pathway': pathway,
                            'lig_expr': lig_expr,
                            'rec_expr': rec_expr,
                            'score': score,
                        })
        
        if results:
            treatment_interactions[treatment] = pd.DataFrame(results)
            print(f"  {len(results):,} interactions detected")
    
    # Save per-treatment interactions
    all_treatment_df = pd.concat(treatment_interactions.values(), ignore_index=True)
    all_treatment_df.to_csv(os.path.join(dirs['data'], 'lr_interactions_per_treatment.csv'), 
                           index=False)
    
    print("\nGenerating per-treatment visualizations...")
    
    # --- Figure 31: Interaction networks per treatment (2x2 grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(34, 34), facecolor='white')
    axes_flat = axes.flatten()
    
    for j, treatment in enumerate(TREATMENT_ORDER):
        if treatment in treatment_interactions:
            df = treatment_interactions[treatment]
            interaction_agg = df.groupby(['sender', 'receiver'])['score'].sum().reset_index()
            
            ax = axes_flat[j]
            plot_interaction_network_ax(interaction_agg, ax,
                                       title=treatment,
                                       color=TREATMENT_COLORS[treatment],
                                       show_legend=(j == 0))
    
    plt.suptitle('Cell-Cell Interaction Networks per Treatment',
                fontsize=58, y=0.995)
    plt.tight_layout()
    # PDF twin for the vector composite (see plot_gsea_results).
    os.makedirs(dirs['interaction'], exist_ok=True)
    for _ext in ('png', 'pdf'):
        plt.savefig(os.path.join(dirs['interaction'],
                                 f'fig31_interaction_networks_per_treatment.{_ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig31 - Per-treatment networks")
    
    # --- Figure 32: Chord diagrams per treatment (separate files for clarity) ---
    for treatment in TREATMENT_ORDER:
        if treatment in treatment_interactions:
            df = treatment_interactions[treatment]
            plot_chord_diagram(df, dirs, 
                             f'fig32_chord_{treatment}.png',
                             # Treatment only: every chord diagram plots the
                             # same quantity, so the analysis name belongs to
                             # the figure, not to each facet.
                             title=treatment)
    print("  ✓ fig32 - Per-treatment chord diagrams")
    
    # --- Figure 33: Interaction rewiring heatmap (GPH+IT vs Sham) ---
    if 'Sham' in treatment_interactions and 'GPH+IT' in treatment_interactions:
        sham_df = treatment_interactions['Sham']
        combo_df = treatment_interactions['GPH+IT']
        
        # Aggregate by sender-receiver pair
        sham_agg = sham_df.groupby(['sender', 'receiver'])['score'].sum()
        combo_agg = combo_df.groupby(['sender', 'receiver'])['score'].sum()
        
        # Compute log2 fold change
        common_pairs = sham_agg.index.intersection(combo_agg.index)
        fc = np.log2((combo_agg.loc[common_pairs] + 0.1) / (sham_agg.loc[common_pairs] + 0.1))
        
        # Reshape to matrix
        fc_df = fc.reset_index()
        fc_pivot = fc_df.pivot(index='sender', columns='receiver', values='score').fillna(0)
        
        # Scale to the actual data range so contrast is visible (data is mostly small positive values)
        vmax_data = max(abs(fc.min()), abs(fc.max()), 0.1)

        fig, ax = plt.subplots(figsize=(22, 20), facecolor='white')
        sns.heatmap(fc_pivot, cmap='RdBu_r', center=0, vmin=-vmax_data, vmax=vmax_data,
                   ax=ax, linewidths=0.5, cbar_kws={'label': 'log₂(FC) GPH+IT vs Sham'},
                   square=True, annot=False, fmt='.2f')
        ax.collections[0].colorbar.set_label('log₂(FC) GPH+IT vs Sham', fontsize=30)
        ax.collections[0].colorbar.ax.tick_params(labelsize=26)
        ax.set_title('Interaction Rewiring: GPH+IT vs Sham',
                    fontsize=44)
        ax.set_xlabel('Receiver Cell Type', fontsize=36)
        # labelpad clears the longest tick label ("M2 Macrophage"); at the
        # default the axis title sat directly against it.
        ax.set_ylabel('Sender Cell Type', fontsize=36, labelpad=18)
        plt.xticks(rotation=45, ha='right', fontsize=30)
        plt.yticks(rotation=0, fontsize=30)

        # Highlight and annotate the top-3 most upregulated sender-receiver pairs
        top3 = fc.sort_values(ascending=False).head(3)
        row_order, col_order = list(fc_pivot.index), list(fc_pivot.columns)
        for (sender, receiver), val in top3.items():
            if sender in row_order and receiver in col_order:
                yi, xi = row_order.index(sender), col_order.index(receiver)
                ax.add_patch(mpatches.Rectangle((xi, yi), 1, 1, fill=False,
                                                 edgecolor='black', lw=3.5))
                ax.text(xi + 0.5, yi + 0.5, f'{val:.2f}', ha='center', va='center',
                        fontsize=26, color='black')
        fig.text(0.5, -0.02, 'Black boxes = top 3 most upregulated sender→receiver pairs',
                 ha='center', va='top', fontsize=26)

        plt.tight_layout()
        os.makedirs(dirs['interaction'], exist_ok=True)
        for _ext in ('png', 'pdf'):
            plt.savefig(os.path.join(dirs['interaction'],
                                     f'fig33_interaction_rewiring_heatmap.{_ext}'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✓ fig33 - Interaction rewiring heatmap")
    
    return treatment_interactions


def plot_interaction_network(interaction_df, dirs, filename, title='Interaction Network'):
    """Plot interaction network using circular layout."""
    fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')
    plot_interaction_network_ax(interaction_df, ax, title)
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()


def plot_interaction_network_ax(interaction_df, ax, title='', color='#B2182B', show_legend=False):
    """Helper to plot professional network on given axis."""
    # Get top interactions
    top_n = min(50, len(interaction_df))
    top_interactions = interaction_df.nlargest(top_n, 'score')
    
    # Get unique cell types
    cell_types = sorted(set(top_interactions['sender'].unique()) | 
                       set(top_interactions['receiver'].unique()))
    n_ct = len(cell_types)
    
    # Circular layout
    angles = np.linspace(0, 2*np.pi, n_ct, endpoint=False)
    positions = {ct: (np.cos(a), np.sin(a)) for ct, a in zip(cell_types, angles)}
    
    # Color palette for cell types
    colors_ct = sns.color_palette("Set2", n_ct)
    ct_colors = {ct: colors_ct[i] for i, ct in enumerate(cell_types)}
    
    # Node degree = total interaction score sent + received (drives node size)
    degree = {ct: 0.0 for ct in cell_types}
    for _, row in top_interactions.iterrows():
        degree[row['sender']] += row['score']
        degree[row['receiver']] += row['score']
    max_degree = max(degree.values()) or 1.0

    # Draw edges (interactions) with curved lines, colored by SENDER for visual separation
    s_max = float(top_interactions['score'].max()) or 1.0
    for _, row in top_interactions.iterrows():
        sender = row['sender']
        receiver = row['receiver']
        score = row['score']

        if sender in positions and receiver in positions:
            x1, y1 = positions[sender]
            x2, y2 = positions[receiver]

            width = np.clip(score / s_max * 4, 0.3, 4)
            alpha = np.clip(score / s_max, 0.25, 0.85)
            edge_color = ct_colors.get(sender, color)  # color by sender cell type

            if sender != receiver:
                cx, cy = (x1 + x2) / 2 * 0.7, (y1 + y2) / 2 * 0.7
                t = np.linspace(0, 1, 50)
                bx = (1-t)**2 * x1 + 2*(1-t)*t*cx + t**2*x2
                by = (1-t)**2 * y1 + 2*(1-t)*t*cy + t**2*y2
                ax.plot(bx, by, color=edge_color, alpha=alpha, linewidth=width,
                        zorder=1, solid_capstyle='round')
            else:
                circle = Circle((x1*1.2, y1*1.2), 0.15, fill=False,
                              edgecolor=edge_color, linewidth=width, alpha=alpha, zorder=1)
                ax.add_patch(circle)
    
    # Shorten long cell-type names so labels fit without colliding
    label_abbrev = {
        'Effector CD4+ T cells': 'Eff. CD4+ T',
        'Endothelial cells': 'Endothelial',
        'Cytotoxic T cells': 'Cytotoxic T',
        'CD4 T cells': 'CD4 T',
        'CD8 T cells': 'CD8 T',
        'M1 Macrophage': 'M1 Mac',
        'M2 Macrophage': 'M2 Mac',
    }

    # Draw nodes, sized by interaction degree (hub cell types appear larger)
    for i, (ct, (x, y)) in enumerate(positions.items()):
        node_size = 300 + (degree[ct] / max_degree) * 2200
        ax.scatter(x, y, s=node_size, c=[ct_colors[ct]], edgecolors='black',
                  linewidths=2.5, zorder=3, alpha=0.9)
        # Node label with background; alternate radial offset to reduce
        # collisions between labels of adjacent nodes around the circle.
        #
        # Labels are horizontal while nodes are spaced by ANGLE, so the gap
        # between neighbours collapses near the top and bottom of the circle
        # where they sit side by side -- that is where "Endothelial" and
        # "Eff. CD4+ T" ran together. Widening the alternation separates such a
        # pair radially, the one axis that still has room there.
        label_r = 1.18 if i % 2 == 0 else 1.62
        label_text = label_abbrev.get(ct, ct)
        # Near 3 and 9 o'clock the horizontal text grows straight back toward
        # its own node, so a centred anchor lays the glyphs over the marker --
        # that is what made "Tregs", "B cells" and "qCAF" touch their nodes.
        # Anchor those to the inner edge so the text starts clear of the node
        # and runs outward; the top and bottom keep the centred alternation,
        # which is what separates neighbours there.
        if abs(x) > 0.35:
            r_eff = max(label_r, 1.34)
            ha = 'left' if x > 0 else 'right'
        else:
            r_eff = label_r
            ha = 'center'
        ax.text(x*r_eff, y*r_eff, label_text, fontsize=28, ha=ha, va='center', bbox=dict(boxstyle='round,pad=0.3',
               facecolor='white', edgecolor='none', alpha=0.8), zorder=4)

    # Was +/-2.4, which left the network occupying well under half its own
    # axes. Side labels now start at 1.34 and run outward rather than being
    # centred on 1.62, so the longest of them ("Cytotoxic T") reaches further
    # than before; 2.30 keeps it inside the axes.
    ax.set_xlim(-2.30, 2.30)
    ax.set_ylim(-2.30, 2.30)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=46, pad=40)

    if show_legend:
        for sz, lbl in [(300, 'Low'), (300 + 2200 * 0.5, 'Medium'), (300 + 2200, 'High')]:
            ax.scatter([], [], s=sz, facecolor='lightgray', edgecolor='black',
                       linewidths=1.5, label=lbl)
        leg = ax.legend(title='Node size = interaction degree\n(edge width = interaction score)',
                  loc='lower left', bbox_to_anchor=(-0.05, -0.08), fontsize=26,
                  title_fontsize=26, frameon=False, labelspacing=1.4, handletextpad=1.2)
        leg._legend_box.align = 'left'


def plot_chord_diagram(interaction_df, dirs, filename, title='Cell-Cell Interaction Chord Diagram'):
    """
    Plot professional chord diagram showing ligand-receptor interactions.
    Publication-quality circular plot with cell types arranged in a circle
    and interactions shown as colored chords.
    """
    print(f"  Generating chord diagram: {filename}")
    
    # Get top interactions for clarity
    top_n = min(80, len(interaction_df))
    top_interactions = interaction_df.nlargest(top_n, 'score')
    
    # Get unique cell types
    cell_types = sorted(set(top_interactions['sender'].unique()) | 
                       set(top_interactions['receiver'].unique()))
    n_ct = len(cell_types)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(22, 22), facecolor='white')
    
    # Circular layout
    angles = np.linspace(0, 2*np.pi, n_ct, endpoint=False)
    positions = {ct: angles[i] for i, ct in enumerate(cell_types)}
    
    # Color palette
    colors_ct = sns.color_palette("Set3", n_ct)
    ct_colors = {ct: colors_ct[i] for i, ct in enumerate(cell_types)}
    
    # Compute gap size between cell types
    gap = 0.05
    arc_size = (2*np.pi - gap*n_ct) / n_ct
    
    # Draw cell type arcs (outer ring)
    radius_outer = 1.0
    radius_inner = 0.85
    
    for i, ct in enumerate(cell_types):
        theta1 = angles[i] + gap/2
        theta2 = angles[i] + arc_size - gap/2
        
        # Draw arc
        wedge = Wedge((0, 0), radius_outer, np.degrees(theta1), np.degrees(theta2),
                     width=radius_outer-radius_inner, facecolor=ct_colors[ct],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(wedge)
        
        # Add label
        label_angle = (theta1 + theta2) / 2
        label_x = np.cos(label_angle) * 1.45
        label_y = np.sin(label_angle) * 1.45

        # Rotate text to be tangent to circle
        rotation = np.degrees(label_angle)
        if np.cos(label_angle) < 0:
            rotation += 180

        ax.text(label_x, label_y, ct, fontsize=26,
               ha='center', va='center', rotation=rotation)

    # Draw interaction chords (Bezier curves)
    for _, row in top_interactions.iterrows():
        sender = row['sender']
        receiver = row['receiver']
        score = row['score']
        
        if sender == receiver:
            continue  # Skip self-interactions for chord diagram
        
        # Get angles
        theta1 = positions[sender] + arc_size/2
        theta2 = positions[receiver] + arc_size/2
        
        # Start and end points on inner circle
        x1 = np.cos(theta1) * radius_inner
        y1 = np.sin(theta1) * radius_inner
        x2 = np.cos(theta2) * radius_inner
        y2 = np.sin(theta2) * radius_inner
        
        # Bezier curve through origin for chord effect
        t = np.linspace(0, 1, 100)
        # Control points pull toward center for chord shape
        cx1, cy1 = x1 * 0.3, y1 * 0.3
        cx2, cy2 = x2 * 0.3, y2 * 0.3
        
        # Cubic Bezier curve
        bx = (1-t)**3*x1 + 3*(1-t)**2*t*cx1 + 3*(1-t)*t**2*cx2 + t**3*x2
        by = (1-t)**3*y1 + 3*(1-t)**2*t*cy1 + 3*(1-t)*t**2*cy2 + t**3*y2
        
        # Width and transparency based on interaction strength
        width = np.clip(score / top_interactions['score'].max() * 8, 0.5, 8)
        alpha = np.clip(score / top_interactions['score'].max(), 0.15, 0.6)
        
        # Use sender color
        ax.plot(bx, by, color=ct_colors[sender], alpha=alpha, 
               linewidth=width, solid_capstyle='round', zorder=1)
    
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=24, pad=50)
    
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()


def plot_interaction_dotplot(interaction_df, dirs, filename, title='Ligand-Receptor Interactions'):
    """
    Plot professional dotplot showing specific L-R pairs between cell types.
    Size = interaction score, Color = significance/strength.
    """
    print(f"  Generating L-R dotplot: {filename}")
    
    # First select top-30 L-R pairs by max score (each pair gets at least one row)
    interaction_df = interaction_df.copy()
    interaction_df['interaction'] = interaction_df['ligand'] + ' - ' + interaction_df['receptor']
    interaction_df['cell_pair']   = interaction_df['sender'] + ' → ' + interaction_df['receiver']

    lr_top = interaction_df.groupby('interaction')['score'].max().nlargest(30).index.tolist()
    plot_data = interaction_df[interaction_df['interaction'].isin(lr_top)].copy()
    if plot_data.empty:
        print("    ⚠ No data to plot"); return

    # Then keep top-30 cell pairs by max score across surviving L-Rs
    cp_top = plot_data.groupby('cell_pair')['score'].max().nlargest(30).index.tolist()
    plot_data = plot_data[plot_data['cell_pair'].isin(cp_top)]
    
    fig, ax = plt.subplots(figsize=(22, 17), facecolor='white')
    
    # Create dotplot
    pivot = plot_data.pivot_table(index='interaction', columns='cell_pair', 
                                   values='score', fill_value=0)
    
    # Plot dots
    for i, interaction in enumerate(pivot.index):
        for j, cell_pair in enumerate(pivot.columns):
            value = pivot.loc[interaction, cell_pair]
            if value > 0:
                size = np.clip(value / pivot.max().max() * 500, 10, 500)
                color_val = np.clip(value / pivot.max().max(), 0.2, 1.0)
                ax.scatter(j, i, s=size, c=[plt.cm.Reds(color_val)], 
                          edgecolors='black', linewidths=0.5, zorder=2, alpha=0.8)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=22)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=22)
    ax.set_xlabel('Cell-Cell Interaction', fontsize=30)
    ax.set_ylabel('Ligand-Receptor Pair', fontsize=30)
    ax.set_title(title, fontsize=34, pad=20)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=pivot.max().max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Interaction Score', fontsize=26)

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()
    
    print(f"    ✓ L-R dotplot saved")


# =====================================================================
# STEP 2: PATHWAY ENRICHMENT (GO/KEGG)
# =====================================================================
def pathway_enrichment_analysis(adata, dirs, species='mouse'):
    """
    Pathway enrichment using GSEAPY (GO/KEGG) on DEGs from treatment comparisons.
    """
    print("\n" + "="*60)
    print("PATHWAY ENRICHMENT ANALYSIS (GO/KEGG)")
    print("="*60)
    
    try:
        import gseapy as gp
    except ImportError:
        print("ERROR: gseapy not installed. Install with: pip install gseapy")
        print("Skipping pathway enrichment analysis.")
        return None
    
    # Load DEGs from previous analysis or compute new ones
    deg_file = os.path.join(dirs['data'], 'DEGs_GPH_IT_vs_Sham.csv')
    if os.path.exists(deg_file):
        print(f"\nLoading DEGs from: {deg_file}")
        degs = pd.read_csv(deg_file)
    else:
        print("\nComputing DEGs: GPH+IT vs Sham")
        mask = adata.obs['treatment'].isin(['Sham', 'GPH+IT'])
        adata_sub = adata[mask].copy()
        sc.tl.rank_genes_groups(adata_sub, groupby='treatment', groups=['GPH+IT'],
                               reference='Sham', method='wilcoxon')
        degs = sc.get.rank_genes_groups_df(adata_sub, group='GPH+IT')
        degs.to_csv(deg_file, index=False)
    
    # Filter significant DEGs
    sig_degs = degs[(degs['pvals_adj'] < 0.05) & (abs(degs['logfoldchanges']) > 0.5)]
    up_genes = sig_degs[sig_degs['logfoldchanges'] > 0]['names'].tolist()
    down_genes = sig_degs[sig_degs['logfoldchanges'] < 0]['names'].tolist()
    
    print(f"\n{len(up_genes)} upregulated genes")
    print(f"{len(down_genes)} downregulated genes")
    
    if len(up_genes) < 5 and len(down_genes) < 5:
        print("Too few DEGs for enrichment analysis")
        return None
    
    # Organism database
    organism = 'mouse' if species == 'mouse' else 'human'
    
    enrichment_results = {}
    
    # --- GO Biological Process enrichment (UP genes) ---
    if len(up_genes) >= 5:
        print("\nRunning GO enrichment (UP genes)...")
        try:
            enr_up_go = gp.enrichr(gene_list=up_genes,
                                  gene_sets=['GO_Biological_Process_2021'],
                                  organism=organism,
                                  outdir=None,
                                  cutoff=0.05)
            enrichment_results['UP_GO_BP'] = enr_up_go.results
            print(f"  {len(enr_up_go.results)} GO terms enriched (UP)")
        except Exception as e:
            print(f"  GO enrichment failed: {e}")
    
    # --- GO Biological Process enrichment (DOWN genes) ---
    if len(down_genes) >= 5:
        print("\nRunning GO enrichment (DOWN genes)...")
        try:
            enr_down_go = gp.enrichr(gene_list=down_genes,
                                    gene_sets=['GO_Biological_Process_2021'],
                                    organism=organism,
                                    outdir=None,
                                    cutoff=0.05)
            enrichment_results['DOWN_GO_BP'] = enr_down_go.results
            print(f"  {len(enr_down_go.results)} GO terms enriched (DOWN)")
        except Exception as e:
            print(f"  GO enrichment failed: {e}")
    
    # --- KEGG pathway enrichment (UP genes) ---
    if len(up_genes) >= 5:
        print("\nRunning KEGG enrichment (UP genes)...")
        try:
            enr_up_kegg = gp.enrichr(gene_list=up_genes,
                                    gene_sets=['KEGG_2019_Mouse' if organism == 'mouse' else 'KEGG_2021_Human'],
                                    organism=organism,
                                    outdir=None,
                                    cutoff=0.05)
            enrichment_results['UP_KEGG'] = enr_up_kegg.results
            print(f"  {len(enr_up_kegg.results)} KEGG pathways enriched (UP)")
        except Exception as e:
            print(f"  KEGG enrichment failed: {e}")
    
    # --- KEGG pathway enrichment (DOWN genes) ---
    if len(down_genes) >= 5:
        print("\nRunning KEGG enrichment (DOWN genes)...")
        try:
            enr_down_kegg = gp.enrichr(gene_list=down_genes,
                                      gene_sets=['KEGG_2019_Mouse' if organism == 'mouse' else 'KEGG_2021_Human'],
                                      organism=organism,
                                      outdir=None,
                                      cutoff=0.05)
            enrichment_results['DOWN_KEGG'] = enr_down_kegg.results
            print(f"  {len(enr_down_kegg.results)} KEGG pathways enriched (DOWN)")
        except Exception as e:
            print(f"  KEGG enrichment failed: {e}")
    
    # Save enrichment results
    for key, result_df in enrichment_results.items():
        if result_df is not None and len(result_df) > 0:
            result_df.to_csv(os.path.join(dirs['data'], f'enrichment_{key}_GPH_IT_vs_Sham.csv'), 
                           index=False)
    
    # --- Figure 31: GO enrichment bar plot (UP genes) ---
    if 'UP_GO_BP' in enrichment_results and len(enrichment_results['UP_GO_BP']) > 0:
        plot_enrichment_barplot(enrichment_results['UP_GO_BP'], dirs,
                               'fig31_GO_enrichment_UP_genes.png',
                               title='GO Biological Process (Upregulated in GPH+IT)',
                               top_n=20)
    
    # --- Figure 32: GO enrichment bar plot (DOWN genes) ---
    if 'DOWN_GO_BP' in enrichment_results and len(enrichment_results['DOWN_GO_BP']) > 0:
        plot_enrichment_barplot(enrichment_results['DOWN_GO_BP'], dirs,
                               'fig32_GO_enrichment_DOWN_genes.png',
                               title='GO Biological Process (Downregulated in GPH+IT)',
                               top_n=20)
    
    # --- Figure 33: KEGG pathway enrichment (UP genes) ---
    if 'UP_KEGG' in enrichment_results and len(enrichment_results['UP_KEGG']) > 0:
        plot_enrichment_barplot(enrichment_results['UP_KEGG'], dirs,
                               'fig33_KEGG_enrichment_UP_genes.png',
                               title='KEGG Pathways (Upregulated in GPH+IT)',
                               top_n=15)
    
    # --- Figure 34: KEGG pathway enrichment (DOWN genes) ---
    if 'DOWN_KEGG' in enrichment_results and len(enrichment_results['DOWN_KEGG']) > 0:
        plot_enrichment_barplot(enrichment_results['DOWN_KEGG'], dirs,
                               'fig34_KEGG_enrichment_DOWN_genes.png',
                               title='KEGG Pathways (Downregulated in GPH+IT)',
                               top_n=15)
    
    # --- Figure 35: GO enrichment dot plot (combined UP/DOWN) ---
    plot_enrichment_combined_dotplot(enrichment_results, dirs,
                                    'fig35_GO_enrichment_combined_dotplot.png')
    
    return enrichment_results


def plot_enrichment_barplot(enr_df, dirs, filename, title='', top_n=20):
    """Plot enrichment results as horizontal bar plot."""
    if len(enr_df) == 0:
        return
    
    # Sort by adjusted p-value, take top N
    enr_df = enr_df.sort_values('Adjusted P-value').head(top_n).copy()
    enr_df['neg_log10_padj'] = -np.log10(enr_df['Adjusted P-value'].clip(lower=1e-20))
    
    fig, ax = plt.subplots(figsize=(17, max(8, len(enr_df) * 0.42)))
    
    bars = ax.barh(range(len(enr_df)), enr_df['neg_log10_padj'], 
                   color='#B2182B', edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(enr_df)))
    ax.set_yticklabels(enr_df['Term'], fontsize=12)
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.axvline(-np.log10(0.05), color='grey', linestyle='--', alpha=0.5, linewidth=1)
    ax.invert_yaxis()

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()


def plot_enrichment_combined_dotplot(enrichment_results, dirs, filename):
    """Plot combined UP/DOWN enrichment as dot plot."""
    if 'UP_GO_BP' not in enrichment_results and 'DOWN_GO_BP' not in enrichment_results:
        return
    
    # Combine top 10 from each
    combined = []
    if 'UP_GO_BP' in enrichment_results and len(enrichment_results['UP_GO_BP']) > 0:
        up_top = enrichment_results['UP_GO_BP'].sort_values('Adjusted P-value').head(10).copy()
        up_top['Direction'] = 'UP'
        combined.append(up_top)
    
    if 'DOWN_GO_BP' in enrichment_results and len(enrichment_results['DOWN_GO_BP']) > 0:
        down_top = enrichment_results['DOWN_GO_BP'].sort_values('Adjusted P-value').head(10).copy()
        down_top['Direction'] = 'DOWN'
        combined.append(down_top)
    
    if not combined:
        return
    
    df = pd.concat(combined, ignore_index=True)
    df['neg_log10_padj'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-20))
    
    # Shorten term names if too long
    df['Term_short'] = df['Term'].str.slice(0, 60)
    
    fig, ax = plt.subplots(figsize=(20, max(11, len(df) * 0.56)))
    
    colors = {'UP': '#B2182B', 'DOWN': '#2166AC'}
    for direction in ['UP', 'DOWN']:
        subset = df[df['Direction'] == direction]
        ax.scatter(subset['neg_log10_padj'], range(len(subset)), 
                  s=subset['neg_log10_padj']*20, c=colors[direction], 
                  alpha=0.7, label=direction, edgecolors='black', linewidths=0.5)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Term_short'], fontsize=11)
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=15)
    ax.set_title('GO Enrichment: UP vs DOWN Genes (GPH+IT vs Sham)', 
                fontsize=17)
    ax.legend(fontsize=13)
    ax.invert_yaxis()
    ax.axvline(-np.log10(0.05), color='grey', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()


# =====================================================================
# STEP 3: GENE SET ENRICHMENT ANALYSIS (GSEA)
# =====================================================================
def gsea_analysis(adata, dirs, species='mouse'):
    """
    Run GSEA (Gene Set Enrichment Analysis) on ranked gene list.
    """
    print("\n" + "="*60)
    print("GENE SET ENRICHMENT ANALYSIS (GSEA)")
    print("="*60)
    
    try:
        import gseapy as gp
    except ImportError:
        print("ERROR: gseapy not installed. Skipping GSEA.")
        return None
    
    # Load DEGs
    deg_file = os.path.join(dirs['data'], 'DEGs_GPH_IT_vs_Sham.csv')
    if not os.path.exists(deg_file):
        print(f"DEG file not found: {deg_file}")
        return None
    
    degs = pd.read_csv(deg_file)
    
    # Create ranked gene list (rank by log2FC * -log10(pval))
    degs['rank_score'] = degs['logfoldchanges'] * (-np.log10(degs['pvals'].clip(lower=1e-300)))
    ranked_genes = degs.sort_values('rank_score', ascending=False)[['names', 'rank_score']]
    ranked_genes = ranked_genes.dropna()
    
    print(f"\n{len(ranked_genes)} genes in ranked list")
    print(f"Top upregulated: {ranked_genes.head(5)['names'].tolist()}")
    print(f"Top downregulated: {ranked_genes.tail(5)['names'].tolist()}")
    
    # Save ranked gene list
    ranked_genes.to_csv(os.path.join(dirs['data'], 'ranked_genes_GPH_IT_vs_Sham.csv'), 
                       index=False)
    
    organism = 'mouse' if species == 'mouse' else 'human'

    import tempfile as _tempfile, urllib.request as _ureq
    def _resolve_lib(name):
        """Download Enrichr GMT directly via REST (bypasses broken gp.get_library)."""
        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={name}"
        path = _tempfile.NamedTemporaryFile(suffix=f'_{name}.gmt', delete=False).name
        try:
            _ureq.urlretrieve(url, path)
            # Uppercase all gene symbols so we can match against UPPERCASED ranked_genes
            with open(path) as fh:
                lines = fh.readlines()
            if len(lines) < 2:
                print(f"  Enrichr returned empty {name}")
                return None
            with open(path, 'w') as fh:
                for ln in lines:
                    parts = ln.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        fh.write(ln); continue
                    fh.write('\t'.join(parts[:2] + [g.upper() for g in parts[2:]]) + '\n')
            print(f"  Fetched {len(lines)} terms from Enrichr → {path}")
            return path
        except Exception as _e:
            print(f"  Enrichr fetch {name} failed: {_e}")
            return None

    # Mouse → upper-case for MSigDB matching. ranked_genes is a Series indexed by gene.
    if organism == 'mouse' and hasattr(ranked_genes, 'index'):
        ranked_genes = ranked_genes.copy()
        ranked_genes.index = [str(g).upper() for g in ranked_genes.index]

    gsea_results = {}
    
    # --- GSEA: Hallmark gene sets ---
    print("\nRunning GSEA: Hallmark gene sets...")
    try:
        gsea_hallmark = gp.prerank(
            rnk=ranked_genes,
            gene_sets=_resolve_lib('MSigDB_Hallmark_2020') or 'MSigDB_Hallmark_2020',
            outdir=None,
            permutation_num=1000,
            min_size=3,
            max_size=2000,
            seed=42,
        )
        gsea_results['Hallmark'] = gsea_hallmark.res2d
        print(f"  {len(gsea_hallmark.res2d)} Hallmark pathways tested")
    except Exception as e:
        print(f"  GSEA Hallmark failed: {e}")
    
    # --- GSEA: KEGG pathways ---
    print("\nRunning GSEA: KEGG pathways...")
    try:
        gsea_kegg = gp.prerank(
            rnk=ranked_genes,
            gene_sets=_resolve_lib('KEGG_2019_Mouse' if organism == 'mouse' else 'KEGG_2021_Human') or 'KEGG_2019_Mouse',
            outdir=None,
            permutation_num=1000,
            min_size=3,
            max_size=2000,
            seed=42,
        )
        gsea_results['KEGG'] = gsea_kegg.res2d
        print(f"  {len(gsea_kegg.res2d)} KEGG pathways tested")
    except Exception as e:
        print(f"  GSEA KEGG failed: {e}")
    
    # --- GSEA: GO Biological Process ---
    print("\nRunning GSEA: GO Biological Process...")
    try:
        gsea_go = gp.prerank(
            rnk=ranked_genes,
            gene_sets=_resolve_lib('GO_Biological_Process_2021') or 'GO_Biological_Process_2021',
            outdir=None,
            permutation_num=1000,
            min_size=3,
            max_size=2000,
            seed=42,
        )
        gsea_results['GO_BP'] = gsea_go.res2d
        print(f"  {len(gsea_go.res2d)} GO BP terms tested")
    except Exception as e:
        print(f"  GSEA GO BP failed: {e}")
    
    # Save GSEA results
    for key, result_df in gsea_results.items():
        if result_df is not None and len(result_df) > 0:
            result_df.to_csv(os.path.join(dirs['data'], f'gsea_{key}_GPH_IT_vs_Sham.csv'), 
                           index=False)
    
    # --- Figure 36: GSEA Hallmark pathways ---
    if 'Hallmark' in gsea_results and len(gsea_results['Hallmark']) > 0:
        plot_gsea_results(gsea_results['Hallmark'], dirs,
                         'fig36_GSEA_Hallmark.png',
                         title='GSEA: Hallmark Pathways (GPH+IT vs Sham)')
    
    # --- Figure 37: GSEA KEGG pathways ---
    if 'KEGG' in gsea_results and len(gsea_results['KEGG']) > 0:
        plot_gsea_results(gsea_results['KEGG'], dirs,
                         'fig37_GSEA_KEGG.png',
                         title='GSEA: KEGG Pathways (GPH+IT vs Sham)')
    
    # --- Figure 38: GSEA enrichment plot (top pathways) ---
    if 'Hallmark' in gsea_results:
        plot_gsea_enrichment_plot(gsea_results['Hallmark'], ranked_genes, dirs,
                                 'fig38_GSEA_enrichment_plot_top.png')
    
    return gsea_results


def plot_gsea_results(gsea_df, dirs, filename, title='', top_n=20, fdr_cutoff=0.25):
    """Plot GSEA results as horizontal bar plot."""
    # Coerce numeric columns (GSEA output sometimes returns strings)
    for col in ['NES', 'NOM p-val', 'FDR q-val']:
        if col in gsea_df.columns:
            gsea_df[col] = pd.to_numeric(gsea_df[col], errors='coerce')

    # Filter by FDR
    sig_gsea = gsea_df[gsea_df['FDR q-val'] < fdr_cutoff].copy()

    if len(sig_gsea) == 0:
        print(f"  No significant pathways (FDR < {fdr_cutoff})")
        return

    # Sort by NES, take top N in each direction
    pos = sig_gsea[sig_gsea['NES'] > 0].nlargest(top_n//2, 'NES')
    neg = sig_gsea[sig_gsea['NES'] < 0].nsmallest(top_n//2, 'NES')
    plot_df = pd.concat([pos, neg]).sort_values('NES')
    
    if len(plot_df) == 0:
        return
    
    # Shorten pathway names
    plot_df['Term_short'] = plot_df['Term'].str.replace('HALLMARK_', '').str.replace('_', ' ').str.slice(0, 50)
    
    fig, ax = plt.subplots(figsize=(17, max(8, len(plot_df) * 0.42)))
    
    colors = ['#2166AC' if x < 0 else '#B2182B' for x in plot_df['NES']]
    bars = ax.barh(range(len(plot_df)), plot_df['NES'], 
                   color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(plot_df)))
    # Type sized against the canvas, not in absolute points: on a 17in figure
    # the old 10-16pt sat at roughly half the text-to-canvas ratio of the other
    # panels, so it shrank to illegibility once placed on a figure page. Row
    # pitch here is ~57pt, so 26pt labels still clear their neighbours.
    ax.set_yticklabels(plot_df['Term_short'], fontsize=26)
    ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=30)
    ax.set_title(title, fontsize=32)
    ax.tick_params(axis='x', labelsize=26)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.invert_yaxis()

    # Add FDR text
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        fdr = row['FDR q-val']
        x_pos = row['NES'] + (0.1 if row['NES'] > 0 else -0.1)
        ax.text(x_pos, i, f"FDR={fdr:.3f}", fontsize=22, va='center',
               ha='left' if row['NES'] > 0 else 'right')

    plt.tight_layout()
    # PDF twin as well: the publication composite is assembled in vector space,
    # so a PNG-only panel would have to be rasterised into it.
    base = os.path.splitext(filename)[0]
    # Only the Hallmark call (base == 'fig36_GSEA_Hallmark') is the curated
    # paper figure (suppl10_a); the KEGG call still runs the analysis above
    # but its figure isn't saved.
    if base in _CURATED_STEMS:
        os.makedirs(dirs['gsea'], exist_ok=True)
        for ext in ('png', 'pdf'):
            plt.savefig(os.path.join(dirs['gsea'], f'{base}.{ext}'),
                        dpi=300, bbox_inches='tight')
    plt.close()


def plot_gsea_enrichment_plot(gsea_df, ranked_genes, dirs, filename):
    """Plot classic GSEA enrichment plot for top 4 pathways."""
    # Get top 2 positive and top 2 negative NES pathways
    sig_gsea = gsea_df[gsea_df['FDR q-val'] < 0.25]
    if len(sig_gsea) == 0:
        return
    
    pos = sig_gsea[sig_gsea['NES'] > 0].nlargest(2, 'NES')
    neg = sig_gsea[sig_gsea['NES'] < 0].nsmallest(2, 'NES')
    plot_pathways = pd.concat([pos, neg])
    
    if len(plot_pathways) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 17))
    axes_flat = axes.flatten()
    
    for i, (idx, row) in enumerate(plot_pathways.iterrows()):
        if i >= 4:
            break
        
        ax = axes_flat[i]
        pathway_name = row['Term'].replace('HALLMARK_', '').replace('_', ' ')
        
        # Simplified enrichment plot (running sum)
        # In real implementation, you'd extract gene set and compute running ES
        ax.plot([0, len(ranked_genes)], [0, row['NES']], 'b-', linewidth=2)
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.set_title(f"{pathway_name}\nNES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}",
                    fontsize=13)
        ax.set_xlabel('Rank in Gene List', fontsize=12)
        ax.set_ylabel('Enrichment Score', fontsize=12)
        ax.grid(alpha=0.3)
    
    # Hide unused axes
    for i in range(len(plot_pathways), 4):
        axes_flat[i].axis('off')
    
    plt.suptitle('GSEA Enrichment Plots: Top Pathways', fontsize=20)
    plt.tight_layout()
    # Not a curated paper figure -- skip saving (computation above still used).
    plt.close()


# =====================================================================
# STEP 4: INTEGRATION SUMMARY FIGURE
# =====================================================================
def integration_summary_figure(adata, interactions_df, enrichment_results, dirs):
    """Create comprehensive summary figure integrating all analyses."""
    print("\n" + "="*60)
    print("CREATING INTEGRATION SUMMARY FIGURE")
    print("="*60)
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Cell type composition per treatment
    ax1 = fig.add_subplot(gs[0, 0])
    prop = pd.crosstab(adata.obs['treatment'], adata.obs['cell_type_auto'], normalize='index')
    prop = prop.reindex(TREATMENT_ORDER)
    prop.T.plot(kind='bar', stacked=False, ax=ax1, color=[TREATMENT_COLORS[t] for t in TREATMENT_ORDER])
    ax1.set_title('A. Cell Type Composition', fontsize=15)
    ax1.set_xlabel('Cell Type', fontsize=13)
    ax1.set_ylabel('Proportion', fontsize=13)
    ax1.legend(title='Treatment', fontsize=10)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel B: Top interactions
    ax2 = fig.add_subplot(gs[0, 1:])
    if interactions_df is not None and len(interactions_df) > 0:
        top_int = interactions_df.nlargest(15, 'score')
        top_int['pair'] = top_int['sender'] + ' → ' + top_int['receiver']
        top_int = top_int.sort_values('score')
        ax2.barh(range(len(top_int)), top_int['score'], color='#B2182B', edgecolor='black', linewidth=0.5)
        ax2.set_yticks(range(len(top_int)))
        ax2.set_yticklabels(top_int['pair'], fontsize=11)
        ax2.set_xlabel('Interaction Score', fontsize=13)
        ax2.set_title('B. Top Cell-Cell Interactions', fontsize=15)
    
    # Panel C: GO enrichment (UP)
    ax3 = fig.add_subplot(gs[1, :])
    if enrichment_results and 'UP_GO_BP' in enrichment_results:
        enr_up = enrichment_results['UP_GO_BP'].sort_values('Adjusted P-value').head(10)
        enr_up['neg_log10_padj'] = -np.log10(enr_up['Adjusted P-value'].clip(lower=1e-20))
        enr_up['Term_short'] = enr_up['Term'].str.slice(0, 60)
        ax3.barh(range(len(enr_up)), enr_up['neg_log10_padj'], color='#B2182B', edgecolor='black', linewidth=0.5)
        ax3.set_yticks(range(len(enr_up)))
        ax3.set_yticklabels(enr_up['Term_short'], fontsize=11)
        ax3.set_xlabel('-log10(Adjusted P-value)', fontsize=13)
        ax3.set_title('C. GO Enrichment (Upregulated Genes)', fontsize=15)
        ax3.invert_yaxis()
    
    # Panel D: Treatment comparison summary
    ax4 = fig.add_subplot(gs[2, :])
    # Summary text
    summary_text = "SUMMARY:\n\n"
    summary_text += f"• Total cells analyzed: {adata.n_obs:,}\n"
    summary_text += f"• Cell types identified: {adata.obs['cell_type_auto'].nunique()}\n"
    if interactions_df is not None:
        summary_text += f"• L-R interactions detected: {len(interactions_df):,}\n"
    if enrichment_results:
        n_enr = sum(len(v) for v in enrichment_results.values() if v is not None)
        summary_text += f"• Pathways enriched: {n_enr}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=14, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.axis('off')
    ax4.set_title('D. Analysis Summary', fontsize=15, loc='left')
    
    plt.suptitle('Stereo-seq Analysis: Cell-Cell Interaction & Pathway Enrichment',
                fontsize=22)
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(os.path.join(dirs['summary'], 'fig39_integration_summary.png'),
    # dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Integration summary figure saved.")


# =====================================================================
# LIANA: COMPREHENSIVE L-R ANALYSIS WITH VALIDATED DATABASES
# =====================================================================
def run_liana_analysis(adata, dirs):
    """
    Run LIANA (LIgand-receptor ANAlysis) for comprehensive cell-cell communication.
    LIANA aggregates multiple L-R databases (CellPhoneDB, CellChat, NicheNet, etc.)
    for more robust interaction inference than any single database.
    """
    print("\n" + "="*60)
    print("LIANA: MULTI-DATABASE L-R ANALYSIS")
    print("="*60)

    try:
        import liana
        from liana.mt import rank_aggregate
        print(f"  LIANA version: {liana.__version__}")
    except ImportError:
        print("  WARNING: LIANA not installed. Install with: pip install liana")
        print("  Skipping LIANA analysis.")
        return

    if 'cell_type_auto' not in adata.obs.columns:
        print("  ERROR: 'cell_type_auto' not found. Run Step 2 annotation first.")
        return

    # Need log-normalized counts for LIANA
    if adata.raw is not None:
        adata_liana = adata.raw.to_adata().copy()
        adata_liana.obs = adata.obs.copy()
    else:
        adata_liana = adata.copy()

    # Enforce normalized + log1p input for robust LIANA scoring
    print("  Preprocessing LIANA input: normalize_total + log1p")
    sc.pp.normalize_total(adata_liana, target_sum=1e4)
    sc.pp.log1p(adata_liana)

    # LIANA requires log-normalized data
    print(f"  Running LIANA rank_aggregate (combines CellPhoneDB, CellChat, Connectome...)")
    print(f"  Cell types: {adata_liana.obs['cell_type_auto'].nunique()}")
    print(f"  Cells: {adata_liana.n_obs:,}")

    try:
        # Run aggregate ranking across multiple methods
        rank_aggregate(
            adata_liana,
            groupby='cell_type_auto',
            resource_name='mouseconsensus',  # Mouse-specific validated resource
            use_raw=False,
            verbose=True,
            n_perms=100,  # Permutation test for significance
            seed=42
        )

        # Extract results
        if 'liana_res' in adata_liana.uns:
            liana_res = adata_liana.uns['liana_res']
            print(f"\n  LIANA found {len(liana_res):,} L-R interactions")

            # Save all results
            liana_res.to_csv(os.path.join(dirs['data'], 'liana_interactions_all.csv'), index=False)
            print(f"  Saved: liana_interactions_all.csv")

            # Filter to significant interactions
            if 'aggregate_rank' in liana_res.columns:
                sig_liana = liana_res[liana_res['aggregate_rank'] < 0.05]
            elif 'magnitude_rank' in liana_res.columns:
                sig_liana = liana_res[liana_res['magnitude_rank'] < 0.5]
            else:
                sig_liana = liana_res.head(500)

            print(f"  Significant interactions: {len(sig_liana):,}")
            sig_liana.to_csv(os.path.join(dirs['data'], 'liana_interactions_significant.csv'), index=False)

            # --- Figure: LIANA dotplot (top interactions) ---
            try:
                from liana.plotting import dotplot as liana_dotplot
                top_liana = sig_liana.head(30) if len(sig_liana) >= 30 else sig_liana

                # Detect available score columns for this liana version
                _colour_col = next(
                    (c for c in ['magnitude_rank', 'aggregate_rank', 'lr_mean', 'specificity_rank']
                     if c in liana_res.columns), liana_res.columns[-1]
                )
                _size_col = next(
                    (c for c in ['specificity_rank', 'aggregate_rank', 'magnitude_rank']
                     if c in liana_res.columns), liana_res.columns[-1]
                )
                _orderby = 'aggregate_rank' if 'aggregate_rank' in liana_res.columns else _colour_col
                fig = liana_dotplot(
                    liana_res=liana_res,
                    source_labels=list(adata_liana.obs['cell_type_auto'].unique()[:6]),
                    target_labels=list(adata_liana.obs['cell_type_auto'].unique()[:6]),
                    top_n=30,
                    orderby=_orderby,
                    orderby_ascending=True,
                    colour=_colour_col,
                    size=_size_col,
                    figure_size=(20, 17)
                )
                plt.suptitle('LIANA: Top 30 L-R Interactions (Aggregate Rank)',
                           fontsize=44)
                # Not a curated paper figure -- skip saving (computation above still used).
                # plt.savefig(os.path.join(dirs['interaction'], 'fig_liana_dotplot_top30.png'),
                # dpi=300, bbox_inches='tight')
                plt.close()
                print("  Saved: fig_liana_dotplot_top30.png")
            except Exception as e:
                print(f"  Note: LIANA dotplot failed ({e}), creating custom plot...")
                # Custom fallback heatmap
                if len(sig_liana) > 0 and 'source' in sig_liana.columns and 'target' in sig_liana.columns:
                    pivot_col = 'aggregate_rank' if 'aggregate_rank' in sig_liana.columns else sig_liana.columns[-1]
                    top30 = sig_liana.head(50)
                    lig_col = next((c for c in ['ligand.complex', 'ligand'] if c in top30.columns), None)
                    rec_col = next((c for c in ['receptor.complex', 'receptor'] if c in top30.columns), None)
                    lig_str = top30[lig_col].astype(str) if lig_col else pd.Series([''] * len(top30), index=top30.index)
                    rec_str = top30[rec_col].astype(str) if rec_col else pd.Series([''] * len(top30), index=top30.index)
                    interaction_labels = top30['source'].astype(str) + ' -> ' + top30['target'].astype(str) + '\n(' + lig_str + '->' + rec_str + ')'

                    fig, ax = plt.subplots(figsize=(20, max(14, len(top30) * 0.35)))
                    scatter = ax.scatter(
                        range(len(top30)),
                        interaction_labels.values,
                        c=top30[pivot_col].values,
                        s=200, cmap='YlOrRd_r', edgecolors='black', linewidths=0.5
                    )
                    plt.colorbar(scatter, ax=ax, label=pivot_col)
                    ax.set_xlabel('Interaction Rank', fontsize=30)
                    ax.set_title('LIANA: Top L-R Interactions', fontsize=36)
                    plt.tight_layout()
                    # Not a curated paper figure -- skip saving (computation above still used).
                    # plt.savefig(os.path.join(dirs['interaction'], 'fig_liana_top_interactions.png'),
                    # dpi=300, bbox_inches='tight')
                    plt.close()

            # --- Figure: Interaction frequency heatmap (sender x receiver) ---
            if 'source' in sig_liana.columns and 'target' in sig_liana.columns:
                freq_matrix = pd.crosstab(sig_liana['source'], sig_liana['target'])

                fig, ax = plt.subplots(figsize=(max(17, len(freq_matrix.columns)*0.7),
                                               max(14, len(freq_matrix.index)*0.5)))
                sns.heatmap(freq_matrix, cmap='YlOrRd', ax=ax,
                           linewidths=0.5, linecolor='white',
                           cbar_kws={'label': '# Significant L-R Pairs'},
                           annot=True if freq_matrix.shape[0] <= 15 else False,
                           fmt='d')
                ax.set_title('LIANA: Interaction Frequency\n(Sender -> Receiver Cell Types)',
                           fontsize=36)
                ax.set_xlabel('Receiver Cell Type', fontsize=30)
                ax.set_ylabel('Sender Cell Type', fontsize=30)
                plt.xticks(rotation=45, ha='right', fontsize=22)
                plt.yticks(rotation=0, fontsize=22)
                plt.tight_layout()
                # Not a curated paper figure -- skip saving (computation above still used).
                # plt.savefig(os.path.join(dirs['interaction'], 'fig_liana_frequency_heatmap.png'),
                # dpi=300, bbox_inches='tight')
                plt.close()
                print("  Saved: fig_liana_frequency_heatmap.png")

            print("\n  LIANA analysis complete!")
            print(f"  Key outputs:")
            print(f"    liana_interactions_all.csv       - All {len(liana_res):,} interactions")
            print(f"    liana_interactions_significant.csv - {len(sig_liana):,} significant")
        else:
            print("  WARNING: LIANA results not found in adata.uns['liana_res']")

    except Exception as e:
        print(f"  ERROR: LIANA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Continuing without LIANA...")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Stereo-seq Step 3: Cell-Cell Interaction & Pathway Enrichment'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing processed_data from Step 2')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--species', type=str, default='mouse',
                       choices=['mouse', 'human'],
                       help='Species (default: mouse)')
    args = parser.parse_args()
    
    print("="*60)
    print("STEREO-SEQ STEP 3: CELL-CELL INTERACTION & PATHWAY ENRICHMENT")
    print("="*60)
    print(f"Species: {args.species}")
    
    dirs = setup_dirs(args.output_dir)
    
    # Load annotated data from Step 2
    merged_path = os.path.join(args.input_dir, 'downstream_analysis', 
                               'processed_data', 'merged_annotated.h5ad')
    if not os.path.exists(merged_path):
        print(f"\nERROR: Annotated data not found: {merged_path}")
        print("Please run step02_build_annotated_h5ad.py first.")
        sys.exit(1)
    
    print(f"\nLoading annotated data: {merged_path}")
    adata = sc.read_h5ad(merged_path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    print(f"  Cell types: {adata.obs['cell_type_auto'].nunique()}")
    print(f"  Treatments: {adata.obs['treatment'].unique().tolist()}")
    
    # Run analyses
    print("\n" + "="*60)
    print("STARTING ANALYSES")
    print("="*60)
    
    # 1. Cell-cell interactions
    interactions_all = cellchat_interaction_analysis(adata, dirs, species=args.species)
    interactions_per_treatment = cellchat_per_treatment(adata, dirs)
    
    # 2. Pathway enrichment
    enrichment_results = pathway_enrichment_analysis(adata, dirs, species=args.species)
    
    # 3. GSEA
    gsea_results = gsea_analysis(adata, dirs, species=args.species)
    
    # 4. Integration summary
    integration_summary_figure(adata, interactions_all, enrichment_results, dirs)

    # ===================================================================
    # LIANA: Multi-database L-R analysis
    # ===================================================================
    print("\nRunning LIANA multi-database L-R analysis...")
    run_liana_analysis(adata, dirs)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    collect(args.output_dir, {
        os.path.join('downstream_analysis', 'figures', '13_gsea', 'fig36_GSEA_Hallmark'): 'suppl10_a',
        os.path.join('downstream_analysis', 'figures', '11_cellchat_interaction', 'fig31_interaction_networks_per_treatment'): 'suppl11_a',
        os.path.join('downstream_analysis', 'figures', '11_cellchat_interaction', 'fig33_interaction_rewiring_heatmap'): 'suppl11_b',
    })

    print("\n" + "="*60)
    print("STEP 3 COMPLETE")
    print("="*60)
    print(f"\nAll outputs in: {args.output_dir}/downstream_analysis")
    print("\nGenerated figures:")
    print("  11_cellchat_interaction/, interaction heatmaps, networks, rewiring")
    print("  12_pathway_enrichment/, GO/KEGG enrichment bar plots")
    print("  13_gsea/, GSEA results and enrichment plots")
    print("  14_integration_summary/, comprehensive summary figure")
    print("\nGenerated data:")
    print("  lr_interactions_all.csv")
    print("  lr_interactions_per_treatment.csv")
    print("  enrichment_*.csv")
    print("  gsea_*.csv")


if __name__ == '__main__':
    main()
