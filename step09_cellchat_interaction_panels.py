#!/usr/bin/env python3
"""
Post-8: CellChat-style circos diagrams and cell-type-specific pathway panels.

Outputs (all under <input_dir>/downstream_analysis/figures/post8_cellchat_panels/):
  cellchat_circos_<treatment>.png, chord per treatment (top-N L-R)
  cellchat_circos_combined.png, 2x2 grid of all 4 treatments
  panel_adm_malignant.png, ADM + mPDAC/ePDAC marker dotplot + spatial
  panel_caf_metabolism.png, CAF AA metabolism + autophagy + ER-stress module scores
  panel_macrophage_tcell.png, M1/M2 ratio + effector T-cell markers
  panel_endothelial.png, Endothelial autophagy/ER-stress markers

Mouse data only. Arial 10pt base; per-plot sizing tuned for legibility.
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgba
import scanpy as sc

warnings.filterwarnings('ignore')

# -------- Publication style: Arial 10pt baseline --------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica',
                        'Nimbus Sans', 'FreeSans', 'DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    'figure.titlesize': 22,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

TREATMENTS = ['Sham', 'IT', 'GPH', 'GPH+IT']

# ============================================================
# Dual-size save helpers
# ============================================================

def _save_small(fig, path, target_inches=2.0, min_font=6):
    """Rescale figure to target_inches × target_inches with font scaling."""
    orig_size = fig.get_size_inches()
    scale = target_inches / max(orig_size)
    texts = list(fig.findobj(mpl.text.Text))
    orig_fs = [t.get_fontsize() for t in texts]
    for t, sz in zip(texts, orig_fs):
        t.set_fontsize(max(min_font, sz * scale))
    # Hide suptitle in small version to avoid collision with panel titles
    sup = fig._suptitle if hasattr(fig, '_suptitle') else None
    sup_vis = sup.get_visible() if sup is not None else None
    if sup is not None:
        sup.set_visible(False)
    # Chord-ring labels stay visible in the small version, unlike the suptitle above.
    hidden_ax_texts = []
    hidden_leaders = []
    fig.set_size_inches(target_inches, target_inches)
    # Not a curated paper figure (small_2x2 preview) -- skip saving.
    # fig.savefig(path, dpi=300, bbox_inches='tight')
    # Restore
    fig.set_size_inches(orig_size)
    for t, sz in zip(texts, orig_fs):
        t.set_fontsize(sz)
    if sup is not None and sup_vis is not None:
        sup.set_visible(sup_vis)
    for txt in hidden_ax_texts:
        txt.set_visible(True)
    for ln in hidden_leaders:
        ln.set_visible(True)


def save_dual(fig, out_dir, fname, close=True):
    """Save normal-size and 2×2-inch versions. Small files go to out_dir/small_2x2/."""
    # None of this helper's callers produce a curated paper figure -- skip saving
    # (the computation that builds each figure still runs).
    if close:
        plt.close(fig)

# Mouse marker panels
ADM_MARKERS       = ['Krt19', 'Sox9', 'Onecut2', 'Pdx1', 'Ptf1a', 'Cpa1', 'Krt8', 'Muc1']
MPDAC_EPDAC       = ['Krt19', 'Krt8', 'Vim', 'Cdh2', 'Snai1', 'Zeb1', 'Cdh1', 'Epcam']
CAF_AA_METAB      = ['Asns', 'Got1', 'Got2', 'Psat1', 'Phgdh', 'Shmt1', 'Shmt2', 'Gpt2', 'Gls', 'Slc1a5']
AUTOPHAGY         = ['Map1lc3a', 'Map1lc3b', 'Sqstm1', 'Atg5', 'Atg7', 'Atg12', 'Becn1', 'Ulk1']
ER_STRESS         = ['Hspa5', 'Atf4', 'Atf6', 'Ddit3', 'Eif2ak3', 'Ern1', 'Xbp1']
M1_MARKERS        = ['Nos2', 'Tnf', 'Il1b', 'Il6', 'Cxcl9', 'Cxcl10', 'Cd86']
M2_MARKERS        = ['Arg1', 'Mrc1', 'Cd163', 'Il10', 'Tgfb1', 'Chil3', 'Retnla']
T_EFFECTORS       = ['Gzmb', 'Prf1', 'Ifng', 'Tnf', 'Gzmk', 'Nkg7', 'Klrg1']
ENDOTHELIAL       = ['Pecam1', 'Cdh5', 'Vwf', 'Kdr', 'Tek', 'Cldn5']

# ============================================================
# Chord / circos diagram (CellChat style)
# ============================================================

def _bezier_chord(p0, p1, ctrl=(0, 0)):
    verts = [p0, ctrl, p1]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    return MplPath(verts, codes)

def draw_chord(ax, mat, labels, colors, title='', top_n=None, min_strength=None, label_fontsize=13, show_labels=True, title_fontsize=29):
    """
    mat: square interaction matrix (sender x receiver), already aggregated
    labels: list[str] of cell types
    colors: list of colors per cell type (sender color)
    """
    n = len(labels)
    if n == 0:
        ax.set_axis_off()
        ax.set_title(title)
        return
    M = mat.copy().astype(float)
    if min_strength is not None:
        M[M < min_strength] = 0
    if top_n is not None:
        flat = M.flatten()
        if (flat > 0).sum() > top_n:
            thr = np.sort(flat)[-top_n]
            M[M < thr] = 0

    # Arc layout
    totals = M.sum(axis=1) + M.sum(axis=0)  # diagonal is 0 (self-interactions excluded)
    if totals.sum() == 0:
        ax.text(0.5, 0.5, 'No interactions', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(title)
        return
    gap = np.deg2rad(2)
    total_gap = gap * n
    avail = 2 * np.pi - total_gap
    fracs = totals / totals.sum()
    angles = np.zeros((n, 2))
    cur = np.pi / 2
    for i in range(n):
        span = fracs[i] * avail
        angles[i] = [cur, cur - span]
        cur -= span + gap

    R = 1.0
    Rin = 0.92
    # Outer arcs
    for i in range(n):
        a0, a1 = angles[i]
        thetas = np.linspace(a0, a1, 60)
        x_o = R * np.cos(thetas); y_o = R * np.sin(thetas)
        x_i = Rin * np.cos(thetas[::-1]); y_i = Rin * np.sin(thetas[::-1])
        verts = list(zip(np.r_[x_o, x_i], np.r_[y_o, y_i]))
        verts.append(verts[0])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colors[i], edgecolor='white', lw=0.5))
        pass  # labels drawn below after all arcs

    # Ribbons: bezier curves through center, correct arc cursor directions
    arc_span = angles[:, 0] - angles[:, 1]   # positive span per cell
    cursor_send = angles[:, 0].copy()          # fills from arc start → end (decreasing)
    cursor_recv = angles[:, 1].copy()          # fills from arc end → start (increasing)
    for i in range(n):
        if M[i].sum() == 0:
            continue
        order = np.argsort(-M[i])
        for j in order:
            v = M[i, j]
            if v <= 0:
                continue
            # Sender segment (moves toward arc end, decreasing angle)
            s0 = cursor_send[i]
            s1 = s0 - arc_span[i] * (v / totals[i])
            cursor_send[i] = s1
            # Receiver segment (moves toward arc start, increasing angle)
            r0 = cursor_recv[j]
            r1 = r0 + arc_span[j] * (v / totals[j])
            cursor_recv[j] = r1

            t_s = np.linspace(s0, s1, 20)
            t_r = np.linspace(r0, r1, 20)
            xs = Rin * np.cos(t_s); ys = Rin * np.sin(t_s)
            xr = Rin * np.cos(t_r); yr = Rin * np.sin(t_r)

            # Build path: sender arc → bezier through center → receiver arc → bezier back
            verts, codes = [], []
            verts.append([xs[0], ys[0]]); codes.append(MplPath.MOVETO)
            for x, y in zip(xs[1:], ys[1:]):
                verts.append([x, y]); codes.append(MplPath.LINETO)
            # bezier through center to receiver start
            verts.append([0.0, 0.0]); codes.append(MplPath.CURVE3)
            verts.append([xr[0], yr[0]]); codes.append(MplPath.CURVE3)
            # receiver arc
            for x, y in zip(xr[1:], yr[1:]):
                verts.append([x, y]); codes.append(MplPath.LINETO)
            # bezier through center back to sender start
            verts.append([0.0, 0.0]); codes.append(MplPath.CURVE3)
            verts.append([xs[0], ys[0]]); codes.append(MplPath.CURVE3)
            verts.append([xs[0], ys[0]]); codes.append(MplPath.CLOSEPOLY)

            color = to_rgba(colors[i], alpha=0.45)
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                                   edgecolor='none', lw=0))

    # Draw labels: polar force-spacing so labels stay close to their arc
    if show_labels:
        _abbrev = {
            'M1 Macrophage': 'M1 Mac.', 'M2 Macrophage': 'M2 Mac.',
            'Endothelial cells': 'Endothelial', 'Cytotoxic T cells': 'Cytotoxic T',
            'Effector CD4+ T cells': 'Eff. CD4+ T', 'CD4 T cells': 'CD4 T',
            'CD8 T cells': 'CD8 T', 'NK cells': 'NK',
        }
        lr_arc   = 1.06   # leader line start (just outside arc)
        # Scale label radius and angular spacing with font size so large labels don't overlap.
        _fs_scale = max(1.0, label_fontsize / 9.0)
        lr_label = 1.40 + 0.25 * (_fs_scale - 1.0)   # push labels outward when larger
        min_dang = 0.32 * _fs_scale                   # widen angular gap when larger

        # Natural mid-angle per cell, sorted clockwise from top
        entries = []
        for i in range(n):
            amid = (angles[i, 0] + angles[i, 1]) / 2
            entries.append([amid, i, _abbrev.get(labels[i], labels[i])])
        # Sort descending (clockwise from π/2)
        entries.sort(key=lambda e: -e[0])

        # Distribute labels uniformly around the circle in their natural clockwise order.
        # This guarantees no overlap regardless of how arc segment sizes are distributed.
        # Uniform spacing = 2π / n, starting at the original topmost label's angle.
        step = 2 * np.pi / len(entries)
        uniform_step = max(step, min_dang)
        # If uniform_step * n exceeds 2π, fall back to 2π/n (just-fits, no overlap angularly)
        if uniform_step * len(entries) > 2 * np.pi:
            uniform_step = 2 * np.pi / len(entries)
        start = entries[0][0]   # anchor on natural topmost label
        for k, e in enumerate(entries):
            e[0] = start - k * uniform_step

        # Uniform *angular* spacing does not guarantee uniform *vertical* spacing.
        # Labels are horizontal text stacked down each flank, so what matters is
        # the gap in y. Near 12/6 o'clock that gap collapses to
        # r*dtheta*|cos(theta)| -> 0 ("B cells" vs "CD4 T"); a purely radial
        # stagger does not help on the flanks, where it instead pushes long
        # labels horizontally into their neighbours ("mPDAC" vs "myCAFs").
        # So enforce a minimum vertical gap explicitly, per side.
        _lim = max(2.4, lr_label + 0.6)

        # Text height in data units, from the axes' real geometry.
        _fig = ax.figure
        _ax_h_pt = _fig.get_size_inches()[1] * ax.get_position().height * 72.0
        _min_gap = (label_fontsize * 1.45) * (2 * _lim) / max(_ax_h_pt, 1.0)

        placed = []
        for side in (+1, -1):                      # right flank, then left flank
            grp = [e for e in entries
                   if (np.cos(e[0]) >= 0) == (side > 0)]
            if not grp:
                continue
            grp.sort(key=lambda e: -np.sin(e[0]))  # top -> bottom
            ys = [lr_label * np.sin(e[0]) for e in grp]
            # Sweep downward, pushing any too-close label further down.
            for k in range(1, len(ys)):
                if ys[k] > ys[k - 1] - _min_gap:
                    ys[k] = ys[k - 1] - _min_gap
            # Re-centre the column so it stays balanced about y=0.
            shift = (ys[0] + ys[-1]) / 2.0
            ys = [y - shift for y in ys]
            for e, y in zip(grp, ys):
                # Keep x on the label circle where possible; clamp so a pushed
                # label never drifts inside the chord diagram.
                x = np.sqrt(max(lr_label ** 2 - y ** 2, (0.55 * lr_label) ** 2))
                placed.append((e, side * x, y))
                _lim = max(_lim, abs(y) + 0.35)

        # Draw leader lines + labels
        for (final_am, i, display), lx, ly in placed:
            orig_am = (angles[i, 0] + angles[i, 1]) / 2
            ax.plot([lr_arc * np.cos(orig_am), lx],
                    [lr_arc * np.sin(orig_am), ly],
                    color='#666666', lw=1.4, alpha=0.85, zorder=0,
                    solid_capstyle='round')
            ax.text(lx, ly, display,
                    ha='left' if lx >= 0 else 'right', va='center',
                    fontsize=label_fontsize)
    else:
        _lim = 1.15

    ax.set_xlim(-_lim, _lim)
    ax.set_ylim(-_lim, _lim)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=max(12, title_fontsize * 1.2))


def build_interaction_matrix(df, score_col='score', cells=None):
    if cells is None:
        cells = sorted(set(df['sender']).union(df['receiver']))
    idx = {c: i for i, c in enumerate(cells)}
    M = np.zeros((len(cells), len(cells)))
    # Use mean score per sender-receiver pair (not sum) to avoid pathway-count bias
    grp = df.groupby(['sender', 'receiver'])[score_col].mean()
    for (s, r), w in grp.items():
        if s in idx and r in idx and s != r:
            M[idx[s], idx[r]] += float(w)
    return cells, M


def make_circos_figures(lr_csv: Path, out_dir: Path):
    if not lr_csv.exists():
        print(f"  L-R csv missing: {lr_csv}; skipping circos")
        return
    df = pd.read_csv(lr_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells = sorted(set(df['sender']).union(df['receiver']))
    cmap = plt.cm.get_cmap('tab20', max(len(all_cells), 3))
    color_map = {c: cmap(i) for i, c in enumerate(all_cells)}

    # Per-treatment + combined
    fig_all, axes = plt.subplots(2, 2, figsize=(24, 24))
    for ax, tr in zip(axes.flat, TREATMENTS):
        sub = df[df['treatment'] == tr]
        if sub.empty:
            ax.set_axis_off(); ax.set_title(tr); continue
        # Always use the full cell-type set so colors and positions are consistent across conditions.
        cells, M = build_interaction_matrix(sub, 'score', cells=all_cells)
        colors = [color_map[c] for c in cells]
        # Treatment name sits above a wide circos with long perimeter labels, so
        # it needs to be clearly larger than those labels to read as the panel's
        # heading rather than as one more annotation.
        draw_chord(ax, M, cells, colors, title=tr, top_n=40, label_fontsize=48, show_labels=True, title_fontsize=92)

        # Single-treatment standalone
        figs, axs = plt.subplots(figsize=(12, 12))
        # Treatment name only. The four subplots differ solely by treatment, so
        # repeating the method on each one costs width without adding anything;
        # the method is named in the panel context and the figure legend.
        # These standalone files are what the Fig 7 composite actually consumes,
        # so the treatment heading is sized here, not on the 2x2 grid above.
        # It has to out-weigh the 39 pt perimeter labels to read as a heading.
        draw_chord(axs, M, cells, colors, title=tr, top_n=40, label_fontsize=39,
                   title_fontsize=76)
        save_dual(figs, out_dir, f'cellchat_circos_{tr.replace("+", "_")}.png')

    # Labels are now drawn on each ring; a suptitle here collided with the
    # per-panel titles and edge labels, so the grid relies on those alone.
    fig_all.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, hspace=0.20, wspace=0.10)
    # Combined gets 4×4 small (2×2 per panel) for legibility
    fig_all.savefig(out_dir / 'cellchat_circos_combined.png', dpi=300, bbox_inches='tight')
    fig_all.savefig(out_dir / 'cellchat_circos_combined.pdf', dpi=300, bbox_inches='tight')
    small_dir = out_dir / 'small_2x2'
    try:
        _save_small(fig_all, small_dir / 'cellchat_circos_combined.png', target_inches=4.0, min_font=5)
    except Exception as e:
        print(f"  [warn] small combined failed: {e}")
    plt.close(fig_all)
    print(f"  Saved circos figures to {out_dir}")


# ============================================================
# Marker panels (dotplot + optional spatial)
# ============================================================

def _present(adata, genes):
    return [g for g in genes if g in adata.var_names]


def panel_dotplot(adata, genes, groupby, title, fname, out_dir, figsize=(12, 6),
                  treatment_split=False, treatment_key='treatment'):
    genes = _present(adata, genes)
    if not genes:
        print(f"  {fname}: no genes found"); return
    if treatment_split and treatment_key in adata.obs:
        ad = adata.copy()
        ad.obs['_grp'] = ad.obs[groupby].astype(str) + ' | ' + ad.obs[treatment_key].astype(str)
        gb = '_grp'
    else:
        ad, gb = adata, groupby
    dp = sc.pl.dotplot(ad, var_names=genes, groupby=gb, show=False, return_fig=True,
                       standard_scale='var', cmap='RdBu_r', dot_max=0.8,
                       figsize=figsize)
    dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5)
    axes_dict = dp.get_axes()
    main_ax = axes_dict.get('mainplot_ax') or list(axes_dict.values())[0]
    main_ax.set_title(title, fontsize=26, pad=10)
    # Not a curated paper figure -- skip saving (computation above still used).
    # dp.savefig(str(out_dir / fname), dpi=300, bbox_inches='tight')
    plt.close('all')


def panel_module_score(adata, gene_set_dict, groupby, title, fname, out_dir, figsize=(10, 8)):
    """Compute sc.tl.score_genes for each gene set and matrixplot mean per group."""
    rows = []
    cols = []
    for name, genes in gene_set_dict.items():
        g = _present(adata, genes)
        if not g:
            continue
        sc.tl.score_genes(adata, gene_list=g, score_name=f'_score_{name}', use_raw=False)
        cols.append(name)
    if not cols:
        print(f"  {fname}: no scorable gene sets"); return
    df = adata.obs.groupby(groupby)[[f'_score_{c}' for c in cols]].mean()
    df.columns = cols
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df.values, cmap='RdBu_r', aspect='auto',
                   vmin=-np.nanmax(np.abs(df.values)), vmax=np.nanmax(np.abs(df.values)))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    ax.set_title(title, fontsize=26)
    fig.colorbar(im, ax=ax, label='Mean module score', shrink=0.7)
    save_dual(fig, out_dir, fname)
    # Cleanup
    for c in cols:
        if f'_score_{c}' in adata.obs:
            del adata.obs[f'_score_{c}']


def panel_m1_m2_ratio(adata, out_dir, fname='panel_macrophage_tcell.png',
                      ct_key='cell_type', treatment_key='treatment'):
    if ct_key not in adata.obs or treatment_key not in adata.obs:
        print(f"  {fname}: missing obs keys"); return
    macs_mask = adata.obs[ct_key].astype(str).str.contains('Macroph|M1|M2|TAM', case=False, regex=True)
    ad_mac = adata[macs_mask].copy()
    if ad_mac.n_obs < 50:
        print(f"  {fname}: too few macrophages")
        return
    m1 = _present(ad_mac, M1_MARKERS); m2 = _present(ad_mac, M2_MARKERS)
    score_cols = []
    if m1: sc.tl.score_genes(ad_mac, m1, score_name='M1_score', use_raw=False); score_cols.append('M1_score')
    if m2: sc.tl.score_genes(ad_mac, m2, score_name='M2_score', use_raw=False); score_cols.append('M2_score')
    if not score_cols:
        print(f"  {fname}: no M1/M2 markers found in data"); return
    df = ad_mac.obs.groupby(treatment_key)[score_cols].mean()
    if 'M1_score' in df.columns and 'M2_score' in df.columns:
        df['M1−M2'] = df['M1_score'] - df['M2_score']
    df = df.reindex([t for t in TREATMENTS if t in df.index])

    # Only show panels that have data
    has_delta = 'M1−M2' in df.columns
    n_panels = 2 if has_delta else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    color_map = {'M1_score': '#d62728', 'M2_score': '#1f77b4'}
    df[score_cols].plot.bar(ax=axes[0], color=[color_map.get(c, '#888888') for c in score_cols],
                             width=0.6, edgecolor='white', linewidth=0.5)
    axes[0].set_title('M1 vs M2 module score', fontsize=26)
    axes[0].set_ylabel('Mean score', fontsize=23); axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=25, labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].legend(fontsize=19, frameon=False)
    axes[0].spines[['top', 'right']].set_visible(False)

    if has_delta:
        colors_d = ['#d62728' if v >= 0 else '#1f77b4' for v in df['M1−M2']]
        df['M1−M2'].plot.bar(ax=axes[1], color=colors_d, width=0.6, edgecolor='white', linewidth=0.5)
        axes[1].set_title('M1 − M2 polarization', fontsize=26)
        axes[1].set_ylabel('Δ score', fontsize=23); axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=25, labelsize=14)
        axes[1].tick_params(axis='y', labelsize=14)
        axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[1].spines[['top', 'right']].set_visible(False)

    fig.suptitle('Macrophage polarization', fontsize=32)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_dual(fig, out_dir, fname)

    # T-cell effectors as a separate clean dotplot
    t_mask = adata.obs[ct_key].astype(str).str.contains('T cell|CD8|CD4|NK', case=False, regex=True)
    ad_t = adata[t_mask].copy()
    teff = _present(ad_t, T_EFFECTORS)
    if ad_t.n_obs > 50 and teff:
        panel_dotplot(ad_t, teff, treatment_key,
                      title='T/NK effector markers per treatment',
                      fname='panel_tcell_effectors.png',
                      out_dir=out_dir, figsize=(14, 6))


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True,
                   help='Work dir containing downstream_analysis/')
    p.add_argument('--lr_csv', default=None,
                   help='Per-treatment L-R CSV (default: <input_dir>/downstream_analysis/processed_data/lr_interactions_per_treatment.csv)')
    p.add_argument('--h5ad', default=None,
                   help='merged_annotated.h5ad (default: <input_dir>/downstream_analysis/processed_data/merged_annotated.h5ad)')
    p.add_argument('--ct_key', default='cell_type')
    p.add_argument('--treatment_key', default='treatment')
    args = p.parse_args()

    base = Path(args.input_dir)
    proc = base / 'downstream_analysis' / 'processed_data'
    out_dir = base / 'downstream_analysis' / 'figures' / 'post8_cellchat_panels'
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_csv = Path(args.lr_csv) if args.lr_csv else proc / 'lr_interactions_per_treatment.csv'
    h5ad   = Path(args.h5ad)   if args.h5ad   else proc / 'merged_annotated.h5ad'

    print("="*70)
    print("POST-8: CellChat circos + pathway/cell-subtype panels")
    print(f"  output: {out_dir}")
    print("="*70)

    # 1) CellChat circos diagrams
    print("\n[1/5] CellChat-style circos diagrams...")
    make_circos_figures(lr_csv, out_dir)

    if not h5ad.exists():
        print(f"\nERROR: {h5ad} not found, skipping marker panels")
        return
    print(f"\nLoading {h5ad} ...")
    adata = sc.read_h5ad(h5ad)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    ct, tr = args.ct_key, args.treatment_key
    if ct not in adata.obs:
        for alt in ('cell_type_auto', 'cluster_annotation', 'celltype'):
            if alt in adata.obs:
                ct = alt; break
    if tr not in adata.obs:
        for alt in ('Treatment', 'group', 'sample'):
            if alt in adata.obs:
                tr = alt; break

    # 2) ADM / malignant transition panel
    print("\n[2/5] ADM / mPDAC ↔ ePDAC malignant transition panel...")
    mal_mask = adata.obs[ct].astype(str).str.contains('PDAC|ADM|Malignant|Tumor', case=False, regex=True)
    if mal_mask.sum() > 100:
        ad_mal = adata[mal_mask].copy()
        panel_dotplot(ad_mal, ADM_MARKERS, ct,
                      title='ADM markers (per malignant cluster)',
                      fname='panel_adm_markers.png',
                      out_dir=out_dir, figsize=(12, 6))
        if tr in ad_mal.obs:
            panel_dotplot(ad_mal, MPDAC_EPDAC, tr,
                          title='mPDAC ↔ ePDAC (EMT) markers per treatment',
                          fname='panel_adm_malignant.png',
                          out_dir=out_dir, figsize=(12, 6))
    else:
        print("  Too few malignant cells; skipping ADM panel")

    # 3) CAF amino-acid metabolism + autophagy + ER-stress
    print("\n[3/5] CAF amino-acid metabolism + autophagy + ER-stress...")
    caf_mask = adata.obs[ct].astype(str).str.contains('CAF|Fibroblast', case=False, regex=True)
    if caf_mask.sum() > 100:
        ad_caf = adata[caf_mask].copy()
        gene_sets = {'AA metabolism': CAF_AA_METAB,
                     'Autophagy': AUTOPHAGY,
                     'ER stress': ER_STRESS}
        panel_module_score(ad_caf, gene_sets, ct,
                           title='CAF subtype metabolic / stress programs',
                           fname='panel_caf_metabolism_per_subtype.png',
                           out_dir=out_dir, figsize=(10, 8))
        if tr in ad_caf.obs:
            panel_module_score(ad_caf, gene_sets, tr,
                               title='CAF metabolic / stress per treatment',
                               fname='panel_caf_metabolism_per_treatment.png',
                               out_dir=out_dir, figsize=(10, 8))
        # Combined dotplot
        all_g = _present(ad_caf, CAF_AA_METAB + AUTOPHAGY + ER_STRESS)
        if all_g:
            panel_dotplot(ad_caf, all_g, ct,
                          title='CAF metabolic & stress gene expression',
                          fname='panel_caf_metabolism_dotplot.png',
                          out_dir=out_dir, figsize=(12, 6))
    else:
        print("  Too few CAFs; skipping")

    # 4) Macrophage M1/M2 + T-cell effectors
    print("\n[4/5] M1/M2 polarization + T-cell effector panel...")
    panel_m1_m2_ratio(adata, out_dir, ct_key=ct, treatment_key=tr)

    # 5) Endothelial autophagy / ER-stress
    print("\n[5/5] Endothelial autophagy / ER-stress panel...")
    endo_mask = adata.obs[ct].astype(str).str.contains('Endothel', case=False, regex=True)
    if endo_mask.sum() > 50:
        ad_endo = adata[endo_mask].copy()
        gene_sets = {'Endothelial markers': ENDOTHELIAL,
                     'Autophagy': AUTOPHAGY,
                     'ER stress': ER_STRESS}
        if tr in ad_endo.obs:
            panel_module_score(ad_endo, gene_sets, tr,
                               title='Endothelial autophagy / ER-stress per treatment',
                               fname='panel_endothelial.png',
                               out_dir=out_dir, figsize=(10, 8))
        all_g = _present(ad_endo, ENDOTHELIAL + AUTOPHAGY + ER_STRESS)
        if all_g and tr in ad_endo.obs:
            panel_dotplot(ad_endo, all_g, tr,
                          title='Endothelial: marker / autophagy / ER-stress',
                          fname='panel_endothelial_dotplot.png',
                          out_dir=out_dir, figsize=(12, 6))
    else:
        print("  Too few endothelial cells; skipping")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    rel = os.path.join('downstream_analysis', 'figures', 'post8_cellchat_panels')
    collect(args.input_dir, {
        os.path.join(rel, 'cellchat_circos_combined'): 'fig7_a',
    })

    print("\nPOST-8 COMPLETE")
    print(f"Figures: {out_dir}")


if __name__ == '__main__':
    main()
