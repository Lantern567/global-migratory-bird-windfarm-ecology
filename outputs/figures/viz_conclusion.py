"""
Conclusion figure: Risk reduction distribution — onshore vs offshore.
Shows the full data distribution, not just averages.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
})

C = {'onshore': '#2E86AB', 'offshore': '#A23B72', 'highlight': '#F18F01'}

# ---- Load ----
t = pd.read_csv(os.path.join(PROC_DIR, 'onshore_tradeoff_results.csv'))
b1 = t[t['budget'] == 0.01]
rr_on = b1['risk_reduction'].values  # 0-100 scale

off = pd.read_csv(os.path.join(PROC_DIR, 'tradeoff_all_171.csv'))
o1 = off[off['budget'] == 0.01]
oradar = o1[o1['group'] == 'Europe (direction data)']
rr_off = oradar['risk_reduction'].values * 100  # convert 0-1 -> 0-100

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ====== Panel 1: Split violins ======
ax = axes[0]
vp_on = ax.violinplot(rr_on, positions=[0], vert=True, widths=0.7,
                       showmeans=True, showmedians=True)
for body in vp_on['bodies']:
    body.set_facecolor(C['onshore']); body.set_alpha(0.55)
vp_on['cmeans'].set_color('#333333'); vp_on['cmeans'].set_linewidth(1.5)
vp_on['cmedians'].set_color('#333333'); vp_on['cmedians'].set_linewidth(1.5)
vp_on['cmedians'].set_linestyle('-')

vp_off = ax.violinplot(rr_off, positions=[1], vert=True, widths=0.7,
                        showmeans=True, showmedians=True)
for body in vp_off['bodies']:
    body.set_facecolor(C['offshore']); body.set_alpha(0.55)
vp_off['cmeans'].set_color('#333333'); vp_off['cmeans'].set_linewidth(1.5)
vp_off['cmedians'].set_color('#333333'); vp_off['cmedians'].set_linewidth(1.5)
vp_off['cmedians'].set_linestyle('-')

# Annotate key stats on violins
for pos, rr, label, color in [
    (0, rr_on, f'Onshore\nn={len(rr_on):,}', C['onshore']),
    (1, rr_off, f'Offshore\nn={len(rr_off)}', C['offshore']),
]:
    p25, p50, p75 = np.percentile(rr, [25, 50, 75])
    mean = np.mean(rr); gt90 = (rr > 90).sum() / len(rr) * 100
    ax.annotate(
        f'{label}\n'
        f'Mean {mean:.0f}%\n'
        f'Median {p50:.0f}%\n'
        f'IQR [{p25:.0f}–{p75:.0f}%]\n'
        f'>90%: {gt90:.0f}% farms',
        xy=(pos, 105), ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor=color, lw=1.2)
    )

ax.set_xticks([0, 1]); ax.set_xticklabels(['Onshore', 'Offshore'], fontsize=13, fontweight='bold')
ax.set_ylabel('Risk Reduction (%)')
ax.set_title('Full Distribution of Risk Reduction\n(1% AEP Budget)', fontweight='bold')
ax.set_ylim(-5, 125); ax.grid(True, axis='y', alpha=0.25)

# ====== Panel 2: Stacked bar — how many farms fall in each RR bin ======
ax2 = axes[1]
bins_edges = [0, 20, 50, 80, 90, 95, 99, 101]
bin_labels = ['0–20%', '20–50%', '50–80%', '80–90%', '90–95%', '95–99%', '99–100%']
colors_on = plt.cm.Blues(np.linspace(0.35, 0.95, len(bin_labels)))
colors_off = plt.cm.RdPu(np.linspace(0.35, 0.95, len(bin_labels)))

x = np.arange(len(bin_labels)); w = 0.35

for i in range(len(bin_labels)):
    lo, hi = bins_edges[i], bins_edges[i+1]
    n_on = ((rr_on >= lo) & (rr_on < hi)).sum()
    n_off = ((rr_off >= lo) & (rr_off < hi)).sum()
    pct_on = n_on / len(rr_on) * 100
    pct_off = n_off / len(rr_off) * 100

    b1 = ax2.bar(i - w/2, pct_on, w, color=C['onshore'], alpha=0.8, edgecolor='white', lw=0.5)
    b2 = ax2.bar(i + w/2, pct_off, w, color=C['offshore'], alpha=0.8, edgecolor='white', lw=0.5)

    if pct_on > 2:
        ax2.text(i - w/2, pct_on + 0.8, f'{pct_on:.0f}%', ha='center', fontsize=8, fontweight='bold', color=C['onshore'])
    if pct_off > 2:
        ax2.text(i + w/2, pct_off + 0.8, f'{pct_off:.0f}%', ha='center', fontsize=8, fontweight='bold', color=C['offshore'])

# Dummy bars for legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C['onshore'], alpha=0.8, label=f'Onshore ({len(rr_on):,} farms)'),
    Patch(facecolor=C['offshore'], alpha=0.8, label=f'Offshore ({len(rr_off)} farms)'),
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax2.set_xticks(x); ax2.set_xticklabels(bin_labels, fontsize=9)
ax2.set_ylabel('Proportion of Farms (%)')
ax2.set_title('How Many Farms Achieve Each Risk Reduction Level?', fontweight='bold')
ax2.set_ylim(0, max(
    max([((rr_on >= lo) & (rr_on < hi)).sum()/len(rr_on)*100 for lo,hi in zip(bins_edges[:-1],bins_edges[1:])]),
    max([((rr_off >= lo) & (rr_off < hi)).sum()/len(rr_off)*100 for lo,hi in zip(bins_edges[:-1],bins_edges[1:])])
) * 1.25)

fig.suptitle('Conclusion: Orientation Leverage Works Across All Wind Farms', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(FIGS_DIR, 'fig21_conclusion_distribution.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
