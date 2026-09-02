#!/usr/bin/env python3
"""
Sankey plots: Cell Type → Ecotype → Treatment (overview) and
              Comparator ← Ecotype → GPH+IT (pairwise).

Input  : unified_ecotype_assignments.csv
         (columns: treatment, cell_type_auto, ecotype)
Output : sankey_gph_it_focus.{png,pdf}
         sankey_gph_it_vs_{Sham,IT,GPH}.png

Usage  :
    python step11_sankey_plots.py \
        --ecotype_csv downstream_analysis/processed_data/unified_ecotype_assignments.csv \
        --out_dir     downstream_analysis/figures/panels_ecotype
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
from matplotlib.path import Path

# Publication typography. Arial is not installed on the cluster; Liberation
# Sans is metric-compatible with Arial and Nimbus Sans is a Helvetica clone.
# fonttype 42 embeds real TrueType text in the PDF instead of outlining it,
# which is what keeps the vector output selectable and editable.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica',
                        'Nimbus Sans', 'FreeSans', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Focal first so it stacks on top
TREATMENT_ORDER = ['GPH+IT', 'Sham', 'IT', 'GPH']
FOCAL = 'GPH+IT'

TREATMENT_COLORS = {
    'GPH+IT': '#D7263D',
    'Sham':   '#A8A8A8',
    'IT':     '#1B998B',
    'GPH':    '#2E86AB',
}


# ─────────────────────────────────────────────────────────────────────────────
# Bezier ribbon helper
# ─────────────────────────────────────────────────────────────────────────────
def _bezier_band(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color, alpha):
    cx0, cx1 = x0 + (x1 - x0) * 0.4, x0 + (x1 - x0) * 0.6
    verts = [
        (x0, y0_top),
        (cx0, y0_top), (cx1, y1_top), (x1, y1_top),
        (x1, y1_bot),
        (cx1, y1_bot), (cx0, y0_bot), (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(
        Path(verts, codes), facecolor=color, alpha=alpha,
        edgecolor='none', linewidth=0, zorder=1))


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_H   = 1.0
NODE_PAD  = 0.018


def _wrap_ct(name, limit=99):
    """Wrap a long cell-type label onto two lines at a word boundary."""
    if len(name) <= limit:
        return name
    words = name.split()
    if len(words) < 2:
        return name
    best, bestdiff = None, None
    for i in range(1, len(words)):
        a, b = ' '.join(words[:i]), ' '.join(words[i:])
        diff = abs(len(a) - len(b))
        if bestdiff is None or diff < bestdiff:
            best, bestdiff = (a, b), diff
    return '\n'.join(best)
NODE_W    = 0.018
MIN_NODE_H = 0.035


def _node_heights(totals: pd.Series, n_nodes: int) -> pd.Series:
    avail = TOTAL_H - max(n_nodes - 1, 0) * NODE_PAD
    raw = (totals / totals.sum()) * avail
    if (raw < MIN_NODE_H).any():
        deficit = (MIN_NODE_H - raw[raw < MIN_NODE_H]).sum()
        big = raw[raw >= MIN_NODE_H]
        if big.sum() > deficit:
            scaled = big - (big / big.sum()) * deficit
            return pd.concat([scaled,
                              pd.Series(MIN_NODE_H, index=raw[raw < MIN_NODE_H].index)]
                             ).reindex(raw.index)
    return raw


def _node_positions(node_list, heights: pd.Series) -> dict:
    pos = {}
    y = TOTAL_H
    for n in node_list:
        h = float(heights[n])
        pos[n] = (y, y - h)
        y -= h + NODE_PAD
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Size configs: large (PPT) and small (2×2" publication thumbnail)
# ─────────────────────────────────────────────────────────────────────────────
SANKEY_CONFIGS = [
    dict(
        # Drawn for a FULL-WIDTH row in the Fig 6 composite, so the canvas is
        # wider than tall.
        #
        # w=20 was too narrow to actually fill that row: the composite scales a
        # panel so its TEXT hits a target size, then upscales the row to the
        # body width under a cap. At 20in the panel needed ~4x to span the row
        # but the cap allowed 2.1, so G rendered at 53% width and left the right
        # half of its own row blank. Widening at FIXED fonts is what fixes this
        # -- it raises the panel's width per unit of text, so the same
        # normalisation yields a wider panel. (Scaling the fonts up with the
        # width, as the old note suggested, would exactly cancel that out.)
        #
        # Aspect is capped by the 16 cell-type labels down the left edge. Their
        # row pitch is (figure_height / y-range) * 72 / 16 points, and it must
        # exceed one line of ct_fs (~1.25 x the point size) or the names
        # collide. That depends on HEIGHT only, so widening is free of it: with
        # h=14in over a y-range of 1.52 the pitch stays ~41pt and ct_fs=30
        # (~38pt line) still clears.
        overview_figsize=(38, 14), pairwise_figsize=(14, 12),
        ct_fs=30, eco_fs=32, treat_fs=50, focal_fs=55,
        comp_fs=58, eco_pw_fs=46, focal_pw_fs=58,
        header_fs=56, header_pw_fs=27,
        title_fs=55, title_pw_fs=52,
        legend_fs=38, legend_title_fs=40,
        node_w=0.018,
        dpi=300, suffix='',
    ),
    dict(
        overview_figsize=(2.0, 2.0), pairwise_figsize=(2.0, 2.0),
        ct_fs=2.5, eco_fs=2.0, treat_fs=2.5, focal_fs=3.0,
        comp_fs=3.0, eco_pw_fs=2.0, focal_pw_fs=3.0,
        header_fs=3.0, header_pw_fs=2.5,
        title_fs=3.5, title_pw_fs=3.0,
        legend_fs=2.0, legend_title_fs=2.5,
        node_w=0.036,
        dpi=300, suffix='_small',
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Overview: Cell Type → Ecotype → Treatment
# ─────────────────────────────────────────────────────────────────────────────
def plot_sankey_matplotlib(eco_df: pd.DataFrame, out_dir: str, cfg: dict):
    ct_eco   = eco_df.groupby(['cell_type_auto', 'ecotype']).size().unstack(fill_value=0)
    eco_treat = eco_df.groupby(['ecotype', 'treatment']).size().unstack(fill_value=0)

    cell_types = sorted(ct_eco.index.tolist())
    ecotypes   = sorted(ct_eco.columns.tolist(),
                        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    treats     = [t for t in TREATMENT_ORDER if t in eco_treat.columns]

    ct_totals    = ct_eco.sum(axis=1)
    eco_totals   = ct_eco.sum(axis=0)
    treat_totals = eco_treat.reindex(columns=treats).sum(axis=0)

    left_h  = _node_heights(ct_totals,    len(cell_types))
    mid_h   = _node_heights(eco_totals,   len(ecotypes))
    right_h = _node_heights(treat_totals, len(treats))

    left_pos  = _node_positions(cell_types, left_h)
    mid_pos   = _node_positions(ecotypes,   mid_h)
    right_pos = _node_positions(treats,     right_h)

    LX, MX, RX = 0.0, 0.5, 1.0

    # The x-padding either side of the diagram exists only to hold the node
    # labels, which are text at a fixed point size and so occupy a FIXED width
    # in inches. The old padding (0.78 / 0.52 data units) was set when the
    # canvas was 20in wide; at 38in those same units buy far more room than the
    # labels need, leaving 36% of the panel blank -- measured as 619pt of dead
    # space on the left and 353pt on the right. Tightening the range hands that
    # width back to the diagram itself.
    #
    # XS rescales every horizontal size expressed in data units so the nodes and
    # label offsets keep their previous size ON THE PAGE; without it they would
    # inflate by 1/XS as the range shrank.
    X_LO, X_HI = -0.20, 1.17
    XS = (X_HI - X_LO) / 2.30          # new range vs the range these were set at
    ECO_NODE_W = 0.135 * XS   # wide enough that 'CC10' sits inside the node
    NODE_W = cfg['node_w'] * XS
    LBL_OFF = 0.015 * XS      # gap from a node edge to its label

    cmap_ct  = matplotlib.colormaps.get_cmap('tab20').resampled(max(len(cell_types), 1))
    ct_colors  = {ct:  mcolors.to_hex(cmap_ct(i))  for i, ct  in enumerate(cell_types)}
    cmap_eco = matplotlib.colormaps.get_cmap('tab10').resampled(max(len(ecotypes), 10))
    eco_colors = {eco: mcolors.to_hex(cmap_eco(i)) for i, eco in enumerate(ecotypes)}

    fig, ax = plt.subplots(figsize=cfg['overview_figsize'])
    # Node labels are drawn outside the node columns; widen the data limits so
    # they stay inside the axes. Otherwise bbox_inches='tight' inflates the
    # saved canvas far past figsize and the text ends up tiny once rescaled.
    ax.set_xlim(X_LO, X_HI)
    # Extra room below the diagram: the legend (titled "Ecotype (ribbon
    # colour)") sits under the axes, and at the enlarged font it rose into the
    # lowest cell-type labels (myCAFs / qCAF). Deepening the y-range pushes the
    # diagram up and leaves the legend a clear band of its own.
    ax.set_ylim(-0.42, 1.10)
    ax.set_axis_off()

    # ── Left ribbons: cell type → ecotype ───────────────────────────────────
    left_used     = {ct:  0.0 for ct  in cell_types}
    mid_used_left = {eco: 0.0 for eco in ecotypes}

    for ct in cell_types:
        for eco in ecotypes:
            v = float(ct_eco.loc[ct, eco]) if eco in ct_eco.columns else 0.0
            if v <= 0:
                continue
            lt, lb = left_pos[ct]
            seg_l = (v / float(ct_totals[ct])) * (lt - lb)
            l_top = lt - left_used[ct];  l_bot = l_top - seg_l
            left_used[ct] += seg_l

            mt, mb = mid_pos[eco]
            seg_m = (v / float(eco_totals[eco])) * (mt - mb)
            m_top = mt - mid_used_left[eco];  m_bot = m_top - seg_m
            mid_used_left[eco] += seg_m

            _bezier_band(ax, LX + NODE_W/2, l_top, l_bot,
                             MX - ECO_NODE_W/2, m_top, m_bot,
                             eco_colors[eco], 0.45)

    # ── Right ribbons: ecotype → treatment ──────────────────────────────────
    mid_used_right = {eco: 0.0 for eco in ecotypes}
    right_used     = {t:   0.0 for t   in treats}

    band_order = [t for t in treats if t != FOCAL] + ([FOCAL] if FOCAL in treats else [])
    for t in band_order:
        for eco in ecotypes:
            if t not in eco_treat.columns or eco not in eco_treat.index:
                continue
            v = float(eco_treat.loc[eco, t])
            if v <= 0:
                continue
            mt, mb = mid_pos[eco]
            seg_m = (v / float(eco_totals[eco])) * (mt - mb)
            m_top = mt - mid_used_right[eco];  m_bot = m_top - seg_m
            mid_used_right[eco] += seg_m

            rt, rb = right_pos[t]
            seg_r = (v / float(treat_totals[t])) * (rt - rb)
            r_top = rt - right_used[t];  r_bot = r_top - seg_r
            right_used[t] += seg_r

            _bezier_band(ax, MX + ECO_NODE_W/2, m_top, m_bot,
                             RX - NODE_W/2, r_top, r_bot,
                             eco_colors[eco], 0.45)

    # ── Draw left nodes (cell types) ────────────────────────────────────────
    for ct in cell_types:
        top, bot = left_pos[ct]
        ax.add_patch(mpatches.Rectangle((LX - NODE_W/2, bot), NODE_W, top - bot,
                                        facecolor=ct_colors[ct], edgecolor='black', linewidth=0.6,
                                        zorder=3))

    # Label y-positions, de-overlapped.
    #
    # Node HEIGHT is proportional to cell count, so node centres are NOT evenly
    # spaced -- the rare types (NK cells, Tregs, apCAFs) get thin adjacent nodes
    # whose centres fall far closer together than the average row pitch. Setting
    # each label at its own node centre therefore collided them regardless of
    # font size. Anchor at the centre, then push apart to a minimum gap derived
    # from the rendered line height, and re-centre the stack so it stays aligned
    # with the column it labels.
    _y_lo, _y_hi = ax.get_ylim()
    _pt_per_data = cfg['overview_figsize'][1] * 72.0 / (_y_hi - _y_lo)
    _min_gap = cfg['ct_fs'] * 1.30 / _pt_per_data

    _ordered = sorted(cell_types, key=lambda c: (left_pos[c][0] + left_pos[c][1]) / 2)
    _ys = [(left_pos[c][0] + left_pos[c][1]) / 2 for c in _ordered]
    for _i in range(1, len(_ys)):                      # push up from the bottom
        if _ys[_i] - _ys[_i - 1] < _min_gap:
            _ys[_i] = _ys[_i - 1] + _min_gap
    _span = _ys[-1] - _ys[0]
    _anchor = sum((left_pos[c][0] + left_pos[c][1]) / 2 for c in _ordered) / len(_ordered)
    _shift = _anchor - (_ys[0] + _span / 2)
    _ys = [y + _shift for y in _ys]
    # Keep the stack inside the column: re-centring alone pushed the topmost
    # name up into the "Cell Type" header sitting at TOTAL_H + 0.025.
    _ceil = TOTAL_H - _min_gap * 1.15
    _floor = -_min_gap * 0.10
    if _ys[-1] > _ceil:
        _ys = [y - (_ys[-1] - _ceil) for y in _ys]
    if _ys[0] < _floor:
        _ys = [y + (_floor - _ys[0]) for y in _ys]

    for ct, ly in zip(_ordered, _ys):
        top, bot = left_pos[ct]
        cy = (top + bot) / 2
        # A short connector only where the label had to move off its node.
        if abs(ly - cy) > _min_gap * 0.28:
            ax.plot([LX - NODE_W/2 - LBL_OFF * 0.72, LX - NODE_W/2],
                    [ly, cy], color='#888888', lw=0.6, alpha=0.7,
                    zorder=4, clip_on=False)
        ax.text(LX - NODE_W/2 - LBL_OFF, ly, _wrap_ct(ct),
                ha='right', va='center', fontsize=cfg['ct_fs'],
                linespacing=0.95, zorder=5)

    # ── Draw middle nodes (ecotypes) ─────────────────────────────────────────
    for eco in ecotypes:
        top, bot = mid_pos[eco]
        ax.add_patch(mpatches.Rectangle((MX - ECO_NODE_W/2, bot), ECO_NODE_W, top - bot,
                                        facecolor=eco_colors[eco], edgecolor='black', linewidth=0.8,
                                        zorder=3))
        # Black text. Thin ecotype bands are shorter
        # than their own label, so it spills past the node onto ribbons behind
        # it; a light stroke keeps it legible over both the node's fill colour
        # and whatever ribbon colour it lands on. zorder keeps it above every
        # ribbon.
        ax.text(MX, (top + bot)/2, eco,
                ha='center', va='center', fontsize=cfg['eco_fs'], color='black',
                zorder=5, clip_on=False,
                path_effects=[mpatheffects.withStroke(linewidth=3.0, foreground='white')])

    # ── Draw right nodes (treatments) ────────────────────────────────────────
    for t in treats:
        top, bot = right_pos[t]
        is_focal = (t == FOCAL)
        nw = NODE_W * (1.6 if is_focal else 1.0)
        ax.add_patch(mpatches.Rectangle((RX - nw/2, bot), nw, top - bot,
                                        facecolor=TREATMENT_COLORS.get(t, '#888888'),
                                        edgecolor='black',
                                        linewidth=1.4 if is_focal else 0.6,
                                        zorder=3))
        # Liberation Sans (the Arial substitute available here) has no U+2605
        # star, so it rendered as a missing-glyph box. A bullet exists in
        # every sans face and reads the same way next to the red label.
        label = f'\u2022 {t}' if is_focal else t
        ax.text(RX + nw/2 + LBL_OFF, (top + bot)/2, label,
                ha='left', va='center',
                fontsize=cfg['focal_fs'] if is_focal else cfg['treat_fs'],
                color=TREATMENT_COLORS[FOCAL] if is_focal else 'black',
                zorder=5)

    # Column headers. "Cell Type" is centred on its node column, so its left
    # half reaches back over the label gutter; at +0.025 it came down onto the
    # topmost name ("B cells") with about a point of clearance. Lift the whole
    # header row so all three stay clear.
    for x, lbl in [(LX, 'Cell Type'), (MX, 'Ecotype'), (RX, 'Treatment')]:
        ax.text(x, TOTAL_H + 0.075, lbl,
                ha='center', va='bottom', fontsize=cfg['header_fs'], color='#333333')

    # No title text: this panel is identified by its composite letter alone.
    # The focal treatment ({FOCAL}) is still marked with a star on the
    # Treatment column, so that information isn't lost.

    legend_handles = [mpatches.Patch(facecolor=eco_colors[eco], alpha=0.7, label=eco)
                      for eco in ecotypes]
    # Anchored by its TOP edge just under the axes, so the legend grows
    # downward into empty space instead of upward into the diagram.
    ax.legend(handles=legend_handles, title='Ecotype (ribbon colour)',
              loc='upper center', bbox_to_anchor=(0.5, -0.015),
              ncol=min(len(ecotypes), 5), frameon=False,
              fontsize=cfg['legend_fs'], title_fontsize=cfg['legend_title_fs'])

    plt.tight_layout()
    suffix = cfg['suffix']
    path = os.path.join(out_dir, f'sankey_gph_it_focus{suffix}.png')
    # Only the large (no-suffix) version is the curated paper figure (fig6_g);
    # the small variant is still built above but not saved.
    if not suffix:
        plt.savefig(path, dpi=cfg['dpi'], bbox_inches='tight', facecolor='white')
        # Vector twin for the publication PDF composite.
        plt.savefig(os.path.join(out_dir, f'sankey_gph_it_focus{suffix}.pdf'),
                    bbox_inches='tight', facecolor='white')
        print(f"  Saved: {path} (+ .pdf)")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise: Comparator (left) → Ecotype (middle) → GPH+IT (right)
# ─────────────────────────────────────────────────────────────────────────────
def plot_sankey_pairwise(eco_df: pd.DataFrame, comparator: str, out_dir: str, cfg: dict):
    treat_eco = eco_df.groupby(['treatment', 'ecotype']).size().unstack(fill_value=0)

    if comparator not in treat_eco.index or FOCAL not in treat_eco.index:
        print(f"  Skipping pairwise {comparator} vs {FOCAL}, data missing")
        return

    ecotypes = sorted(treat_eco.columns.tolist(),
                      key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    comp_vals  = treat_eco.loc[comparator].reindex(ecotypes, fill_value=0).astype(float)
    focal_vals = treat_eco.loc[FOCAL].reindex(ecotypes, fill_value=0).astype(float)

    LX, MX, RX = 0.0, 0.5, 1.0
    ECO_NODE_W = 0.135  # wide enough that 'CC10' sits inside the node

    # Middle node heights: proportional to average of both treatments
    avg = (comp_vals + focal_vals) / 2.0
    mid_h = _node_heights(avg, len(ecotypes))
    mid_pos = _node_positions(ecotypes, mid_h)

    cmap_eco = matplotlib.colormaps.get_cmap('tab10').resampled(max(len(ecotypes), 10))
    eco_colors = {eco: mcolors.to_hex(cmap_eco(i)) for i, eco in enumerate(ecotypes)}

    comp_color  = TREATMENT_COLORS.get(comparator, '#888888')
    focal_color = TREATMENT_COLORS[FOCAL]

    fig, ax = plt.subplots(figsize=cfg['pairwise_figsize'])
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.24, 1.10)
    ax.set_axis_off()

    comp_total  = comp_vals.sum()  if comp_vals.sum()  > 0 else 1.0
    focal_total = focal_vals.sum() if focal_vals.sum() > 0 else 1.0

    # Running cumulative offsets within the single left/right nodes
    comp_offset  = 0.0
    focal_offset = 0.0

    for eco in ecotypes:
        mt, mb = mid_pos[eco]

        # ── Left ribbon: comparator → ecotype ───────────────────────────────
        v_comp = float(comp_vals[eco])
        if v_comp > 0:
            seg_l = (v_comp / comp_total) * TOTAL_H
            l_top = TOTAL_H - comp_offset
            l_bot = l_top - seg_l
            comp_offset += seg_l
            _bezier_band(ax,
                         LX + cfg['node_w']/2, l_top, l_bot,
                         MX - ECO_NODE_W/2, mt, mb,
                         eco_colors[eco], 0.45)

        # ── Right ribbon: ecotype → GPH+IT ──────────────────────────────────
        v_focal = float(focal_vals[eco])
        if v_focal > 0:
            seg_r = (v_focal / focal_total) * TOTAL_H
            r_top = TOTAL_H - focal_offset
            r_bot = r_top - seg_r
            focal_offset += seg_r
            _bezier_band(ax,
                         MX + ECO_NODE_W/2, mt, mb,
                         RX - cfg['node_w']/2, r_top, r_bot,
                         eco_colors[eco], 0.45)

    # ── Draw left node (comparator) ──────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle((LX - cfg['node_w']/2, 0.0), cfg['node_w'], TOTAL_H,
                                    facecolor=comp_color, edgecolor='black', linewidth=0.8))
    ax.text(LX - cfg['node_w']/2 - 0.015, 0.5, comparator,
            ha='right', va='center', fontsize=cfg['comp_fs'], color=comp_color)

    # ── Draw middle nodes (ecotypes), labels inside the box ────────────────
    for i, eco in enumerate(ecotypes):
        top, bot = mid_pos[eco]
        mid_y = (top + bot) / 2
        ax.add_patch(mpatches.Rectangle((MX - ECO_NODE_W/2, bot), ECO_NODE_W, top - bot,
                                        facecolor=eco_colors[eco], edgecolor='black', linewidth=0.8))
        ax.text(MX, mid_y, eco, ha='center', va='center',
                fontsize=cfg['eco_pw_fs'], color='white', clip_on=True)

    # ── Draw right node (GPH+IT focal) ───────────────────────────────────────
    nw = cfg['node_w'] * 1.6
    ax.add_patch(mpatches.Rectangle((RX - nw/2, 0.0), nw, TOTAL_H,
                                    facecolor=focal_color, edgecolor='black', linewidth=1.4))
    ax.text(RX + nw/2 + 0.015, 0.5, f'\u2022 {FOCAL}',
            ha='left', va='center', fontsize=cfg['focal_pw_fs'], color=focal_color)

    # Column headers
    for x, lbl in [(LX, comparator), (MX, 'Ecotype'), (RX, FOCAL)]:
        ax.text(x, TOTAL_H + 0.04, lbl,
                ha='center', va='bottom', fontsize=cfg['header_pw_fs'], color='#333333')

    ax.set_title(f'Ecotype composition  ·  {comparator}  ←  ecotypes  →  {FOCAL}',
                 fontsize=cfg['title_pw_fs'], pad=20)

    legend_handles = [mpatches.Patch(facecolor=eco_colors[eco], alpha=0.7, label=eco)
                      for eco in ecotypes]
    ax.legend(handles=legend_handles, title='Ecotype (ribbon colour)',
              loc='lower center', bbox_to_anchor=(0.5, -0.07),
              ncol=min(len(ecotypes), 5), frameon=False,
              fontsize=cfg['legend_fs'], title_fontsize=cfg['legend_title_fs'])

    plt.tight_layout()
    suffix = cfg['suffix']
    safe = comparator.replace('+', '_plus_').replace(' ', '_')
    path = os.path.join(out_dir, f'sankey_gph_it_vs_{safe}{suffix}.png')
    # Not a curated paper figure -- skip saving (computation above still used).
    # plt.savefig(path, dpi=cfg['dpi'], bbox_inches='tight', facecolor='white')
    # print(f"  Saved: {path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ecotype_csv',
                   default=os.path.join(os.path.expanduser('~'),
                                        'stereo-seq', 'stereoseq-analysis',
                                        'downstream_analysis', 'processed_data',
                                        'unified_ecotype_assignments.csv'))
    p.add_argument('--out_dir',
                   default=os.path.join(os.path.expanduser('~'),
                                        'stereo-seq', 'stereoseq-analysis',
                                        'downstream_analysis', 'figures',
                                        'panels_ecotype'))
    args = p.parse_args()

    if not os.path.exists(args.ecotype_csv):
        raise FileNotFoundError(args.ecotype_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    eco_df = pd.read_csv(args.ecotype_csv, index_col=0)
    print(f"Loaded {len(eco_df):,} cells with columns: {list(eco_df.columns)}")

    # Normalise column names
    eco_df.columns = eco_df.columns.str.strip()
    required = {'treatment', 'cell_type_auto', 'ecotype'}
    missing = required - set(eco_df.columns)
    if missing:
        raise ValueError(f"Missing columns in ecotype CSV: {missing}")

    # Drop rows with NaN
    eco_df = eco_df.dropna(subset=['treatment', 'cell_type_auto', 'ecotype'])
    print(f"After dropping NaN: {len(eco_df):,} cells")
    print(f"  Treatments : {sorted(eco_df['treatment'].unique())}")
    print(f"  Ecotypes   : {sorted(eco_df['ecotype'].unique())}")
    print(f"  Cell types : {sorted(eco_df['cell_type_auto'].unique())}")

    for cfg in SANKEY_CONFIGS:
        label = 'small' if cfg['suffix'] else 'large (PPT)'
        print(f"\n--- Generating {label} Sankey figures ---")
        plot_sankey_matplotlib(eco_df, args.out_dir, cfg)
        for comparator in ['Sham', 'IT', 'GPH']:
            plot_sankey_pairwise(eco_df, comparator, args.out_dir, cfg)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collect_results import collect
    base_dir = args.out_dir.split('downstream_analysis')[0] or '.'
    collect(base_dir, {
        os.path.join('downstream_analysis', 'figures', 'panels_ecotype', 'sankey_gph_it_focus'): 'fig6_g',
    })


if __name__ == '__main__':
    main()
