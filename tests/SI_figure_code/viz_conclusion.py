# -*- coding: utf-8 -*-
"""
Conclusion figure: Risk reduction distribution — onshore vs offshore (VPTS vs Bauer).

Legacy single-figure script. Produces the SAME fig21 as
regenerate_offshore_figures.py (which also draws fig5 / fig14), so running either
script yields an identical, correct 3-group figure.

数据源：
  onshore_tradeoff_results.csv    (risk_reduction 0-100 百分比)
  tradeoff_offshore_55farms.csv   (risk_reduction_pct 0-100 百分比)
  offshore_farm_directions_55.csv (source: VPTS / Bauer_grid)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, '..', 'data', 'processed')
FIGS = os.path.join(HERE, '..', 'figures')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9,
})

C = {'onshore': '#2E86AB', 'vpts': '#27AE60', 'bauer': '#A23B72'}

# ---- Load ----
on = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'))
on1 = on[on['budget'] == 0.01].copy()
rr_on = on1['risk_reduction'].values  # 0-100

off = pd.read_csv(os.path.join(PROC, 'tradeoff_offshore_55farms.csv'))
off1 = off[off['budget'] == 0.01].copy()
directions = pd.read_csv(os.path.join(PROC, 'offshore_farm_directions_55.csv'))
vpts_ids = set(directions[directions['source'] == 'VPTS']['farm_id'])
bauer_ids = set(directions[directions['source'] == 'Bauer_grid']['farm_id'])
rr_vpts = off1[off1['farm_id'].isin(vpts_ids)]['risk_reduction_pct'].values
rr_bauer = off1[off1['farm_id'].isin(bauer_ids)]['risk_reduction_pct'].values

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ====== Panel 1: violins ======
ax = axes[0]
groups = [
    (0, rr_on, 'Onshore', C['onshore']),
    (1, rr_vpts, 'VPTS', C['vpts']),
    (2, rr_bauer, 'Bauer', C['bauer']),
]
for pos, rr, label, color in groups:
    vp = ax.violinplot(rr, positions=[pos], vert=True, widths=0.7, showmeans=True, showmedians=True)
    for body in vp['bodies']:
        body.set_facecolor(color); body.set_alpha(0.55)
    vp['cmeans'].set_color('#333'); vp['cmeans'].set_linewidth(1.5)
    vp['cmedians'].set_color('#333'); vp['cmedians'].set_linewidth(1.5)
    vp['cmedians'].set_linestyle('-')

for pos, rr, label, color in groups:
    p25, p50, p75 = np.percentile(rr, [25, 50, 75])
    mean = np.mean(rr); gt90 = (rr > 90).sum() / len(rr) * 100
    ax.annotate(
        f'{label}\nn={len(rr):,}\nMean {mean:.0f}%\nMedian {p50:.0f}%\nIQR [{p25:.0f}–{p75:.0f}%]\n>90%: {gt90:.0f}%',
        xy=(pos, 105), ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor=color, lw=1.2))

ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Onshore', 'Offshore\nVPTS', 'Offshore\nBauer'], fontsize=12, fontweight='bold')
ax.set_ylabel('Risk Reduction (%)')
ax.set_title('Full Distribution of Risk Reduction\n(1% AEP Budget)', fontweight='bold')
ax.set_ylim(-5, 130); ax.grid(True, axis='y', alpha=0.25)

# ====== Panel 2: grouped bars ======
ax2 = axes[1]
bins_edges = [0, 20, 50, 80, 90, 95, 99, 101]
bin_labels = ['0–20%', '20–50%', '50–80%', '80–90%', '90–95%', '95–99%', '99–100%']
x = np.arange(len(bin_labels)); w = 0.22
series = [('Onshore', rr_on, C['onshore']), ('VPTS', rr_vpts, C['vpts']), ('Bauer', rr_bauer, C['bauer'])]
for si, (label, rr, color) in enumerate(series):
    offset = (si - 1) * (w + 0.02)
    for bi in range(len(bin_labels)):
        lo, hi = bins_edges[bi], bins_edges[bi + 1]
        n = ((rr >= lo) & (rr < hi)).sum()
        pct = n / len(rr) * 100
        ax2.bar(bi + offset, pct, w, color=color, alpha=0.8, edgecolor='white', lw=0.5)
        if pct > 3:
            ax2.text(bi + offset, pct + 0.8, f'{pct:.0f}%', ha='center', fontsize=7,
                     fontweight='bold', color=color)

legend_elements = [Patch(facecolor=c, alpha=0.8, label=f'{lab} ({len(rr):,})') for lab, rr, c in series]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels(bin_labels, fontsize=9)
ax2.set_ylabel('Proportion of Farms (%)')
ax2.set_title('How Many Farms Achieve Each Risk Reduction Level?', fontweight='bold')
ax2.set_ylim(0, 70)
ax2.grid(axis='y', alpha=0.25)

fig.suptitle('Conclusion: Orientation Leverage — Onshore vs Offshore (VPTS vs Bauer)', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
out = os.path.join(FIGS, 'fig21_conclusion_distribution.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')