# -*- coding: utf-8 -*-
"""
重画海上相关图（VPTS / Bauer 两源分列），替代旧的"合并海上"画法。

产出（写入 our_work/figures/）：
  fig5_tradeoff_scatter.png        —— 海上散点：VPTS(37) vs Bauer(18) vs ERA5(116)
  fig14_onshore_tradeoff_scatter.png —— 陆上+海上散点：海上分 VPTS/Bauer 两色
  fig21_conclusion_distribution.png —— 小提琴+分组柱状：陆上 / VPTS / Bauer

数据源：
  onshore_tradeoff_results.csv   (risk_reduction 0-100 百分比)
  tradeoff_offshore_55farms.csv  (risk_reduction_pct 0-100 百分比)
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

C = {
    'onshore': '#2E86AB',
    'vpts': '#27AE60',       # VPTS 雷达
    'bauer': '#A23B72',      # Bauer 格网
    'era5': '#B0B0B0',       # ERA5 代理（已证伪）
}

# ---- 加载 ----
on = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'))
on1 = on[on['budget'] == 0.01].copy()
rr_on = on1['risk_reduction'].values  # 0-100

off = pd.read_csv(os.path.join(PROC, 'tradeoff_offshore_55farms.csv'))
off1 = off[off['budget'] == 0.01].copy()

directions = pd.read_csv(os.path.join(PROC, 'offshore_farm_directions_55.csv'))
vpts_ids = set(directions[directions['source'] == 'VPTS']['farm_id'])
bauer_ids = set(directions[directions['source'] == 'Bauer_grid']['farm_id'])

def grp(fid):
    if fid in vpts_ids:
        return 'vpts'
    if fid in bauer_ids:
        return 'bauer'
    return 'era5'

off1['grp'] = off1['farm_id'].apply(grp)
vpts = off1[off1['grp'] == 'vpts']
bauer = off1[off1['grp'] == 'bauer']
era5 = off1[off1['grp'] == 'era5']
print(f'offshore @1%: VPTS={len(vpts)} Bauer={len(bauer)} ERA5={len(era5)}  (total {len(off1)})')


# ======================================================================
# FIG 5: 海上散点，VPTS / Bauer / ERA5 三组
# ======================================================================
fig, ax = plt.subplots(figsize=(11, 7))
for sub, color, label, alpha, s, z in [
    (era5, C['era5'], f'ERA5 proxy (n={len(era5)})\ncoarse flyway, NOT validated', 0.25, 30, 2),
    (bauer, C['bauer'], f'Bauer grid <200km (n={len(bauer)})\nradar-measured direction', 0.75, 70, 3),
    (vpts, C['vpts'], f'VPTS radar (n={len(vpts)})\nradar-measured direction', 0.85, 80, 3),
]:
    ax.scatter(sub['aep_cost_pct'], sub['risk_reduction_pct'],
               c=color, s=s, alpha=alpha, edgecolors='#333', linewidth=0.3,
               label=label, zorder=z)

ax.axhline(y=50, color='#555', ls=':', alpha=0.5, lw=1)
ax.annotate('50% risk reduction', xy=(0.95, 52), fontsize=8, color='#555', ha='right')
ax.axvline(x=1.0, color='#555', ls=':', alpha=0.5, lw=1)
ax.annotate('1.0% AEP budget', xy=(1.02, 95), fontsize=8, color='#555', rotation=90)
ax.fill_between([0, 1.0], 50, 100, color=C['vpts'], alpha=0.05)

ax.set_xlabel('AEP Cost (% of maximum annual generation)')
ax.set_ylabel('Geometric Exposure Risk Reduction (%)')
ax.set_title('Figure 5: Offshore Energy-Ecology Trade-off at 1% AEP Budget\n'
             'VPTS vs Bauer direction sources reported separately', fontweight='bold')
ax.set_xlim(0, 1.05); ax.set_ylim(-5, 105)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(alpha=0.2)

stats = (
    f'VPTS radar ({len(vpts)}):  median RR {vpts.risk_reduction_pct.median():.0f}%  '
    f'AEP {vpts.aep_cost_pct.mean():.2f}%\n'
    f'Bauer grid ({len(bauer)}):  median RR {bauer.risk_reduction_pct.median():.0f}%  '
    f'AEP {bauer.aep_cost_pct.mean():.2f}%\n'
    f'ERA5 proxy ({len(era5)}):  NOT field-validated'
)
ax.text(0.98, 0.02, stats, transform=ax.transAxes, fontsize=8.5,
        va='bottom', ha='right', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FDF2F2', edgecolor='#999', alpha=0.9))

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig5_tradeoff_scatter.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved fig5_tradeoff_scatter.png')


# ======================================================================
# FIG 14: 陆上 + 海上散点，海上分 VPTS/Bauer
# ======================================================================
fig, ax = plt.subplots(figsize=(8.5, 6.2))
np.random.seed(1)
jx = np.random.normal(0, 0.006, len(on1)); jy = np.random.normal(0, 0.6, len(on1))
ax.scatter(on1['aep_cost_pct'] + jx, rr_on + jy, c=C['onshore'], alpha=0.12, s=5,
           edgecolors='none', label=f'Onshore (n={len(on1):,})')

for sub, color, lab in [
    (vpts, C['vpts'], f'Offshore VPTS (n={len(vpts)})'),
    (bauer, C['bauer'], f'Offshore Bauer (n={len(bauer)})'),
]:
    ojx = np.random.normal(0, 0.006, len(sub)); ojy = np.random.normal(0, 0.6, len(sub))
    ax.scatter(sub['aep_cost_pct'] + ojx, sub['risk_reduction_pct'] + ojy,
               c=color, alpha=0.6, s=28, edgecolors='white', linewidth=0.3, label=lab)

ax.axvline(on1['aep_cost_pct'].mean(), color=C['onshore'], ls='--', lw=1, alpha=0.8)
ax.axhline(rr_on.mean(), color=C['onshore'], ls='--', lw=1, alpha=0.8)
ax.axvline(vpts['aep_cost_pct'].mean(), color=C['vpts'], ls='--', lw=1, alpha=0.8)
ax.axhline(vpts['risk_reduction_pct'].mean(), color=C['vpts'], ls='--', lw=1, alpha=0.8)
ax.axvline(bauer['aep_cost_pct'].mean(), color=C['bauer'], ls='--', lw=1, alpha=0.8)
ax.axhline(bauer['risk_reduction_pct'].mean(), color=C['bauer'], ls='--', lw=1, alpha=0.8)

ax.text(0.98, 0.20,
    f'Onshore {on1["aep_cost_pct"].mean():.3f}% AEP -> {rr_on.mean():.1f}% RR (med {np.median(rr_on):.0f}%)\n'
    f'VPTS     {vpts["aep_cost_pct"].mean():.3f}% AEP -> {vpts["risk_reduction_pct"].mean():.1f}% RR (med {vpts["risk_reduction_pct"].median():.0f}%)\n'
    f'Bauer    {bauer["aep_cost_pct"].mean():.3f}% AEP -> {bauer["risk_reduction_pct"].mean():.1f}% RR (med {bauer["risk_reduction_pct"].median():.0f}%)',
    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.set_xlabel('AEP Cost (%)'); ax.set_ylabel('Risk Reduction (%)')
ax.set_title('Onshore vs Offshore Energy-Ecology Trade-off (1% AEP Budget)', fontweight='bold')
ax.legend(loc='upper left', markerscale=2)
ax.set_xlim(0, 1.02); ax.set_ylim(-5, 103)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig14_onshore_tradeoff_scatter.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved fig14_onshore_tradeoff_scatter.png')


# ======================================================================
# FIG 21: 小提琴 + 分组柱状，陆上 / VPTS / Bauer
# ======================================================================
rr_vpts = vpts['risk_reduction_pct'].values
rr_bauer = bauer['risk_reduction_pct'].values

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
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
fig.savefig(os.path.join(FIGS, 'fig21_conclusion_distribution.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('saved fig21_conclusion_distribution.png')
print('DONE')