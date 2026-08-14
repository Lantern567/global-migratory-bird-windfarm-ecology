import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in fm.findSystemFonts():
    if 'simhei' in fp.lower():
        fm.fontManager.addfont(fp)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

out_dir = r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\outputs'
os.makedirs(out_dir, exist_ok=True)

td = pd.read_csv(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\processed\tradeoff_all_171.csv')
summary = pd.read_csv(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\processed\all_171_risk_summary.csv')
groups = pd.read_csv(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\processed\farm_groups_corrected.csv')
# td already has country, capacity_kW, group from earlier script
# Only add group if missing
if 'group' not in td.columns:
    td = td.merge(groups[['farm_id','group']], on='farm_id')

# ============================================================
# FIG 5: Risk Reduction vs AEP Cost (1% budget, all 171 farms)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

budget01 = td[td['budget'] == 0.01].copy()
budget01['evidence_type'] = 'Coarse flyway (134 farms)\nno field-measured\ndirection data'
budget01.loc[budget01['group'] == 'Europe (direction data)', 'evidence_type'] = 'Radar-measured (37 farms)\nVPTS 200m layer\nnighttime observations'

colors = {'Radar-measured (37 farms)\nVPTS 200m layer\nnighttime observations': '#27AE60',
          'Coarse flyway (134 farms)\nno field-measured\ndirection data': '#E67E22'}

for label, color in colors.items():
    sub = budget01[budget01['evidence_type'] == label]
    ax.scatter(sub['aep_cost_pct'], sub['relative_risk_reduction']*100,
              c=color, s=60, alpha=0.7, edgecolors='#333', linewidth=0.3, label=label, zorder=3)

# Annotate top radar farms
radar_sub = budget01[budget01['group']=='Europe (direction data)']
top5 = radar_sub.nlargest(5, 'relative_risk_reduction')
for _, r in top5.iterrows():
    ax.annotate(f'Farm {int(r["farm_id"])} ({r["country"]})',
               (r['aep_cost_pct'], r['relative_risk_reduction']*100),
               fontsize=7, ha='left', xytext=(5, 5), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))

# Shade regions
ax.axhline(y=50, color='#555', linestyle=':', alpha=0.5, lw=1)
ax.annotate('50% risk reduction', xy=(0.95, 52), fontsize=8, color='#555', ha='right')
ax.axvline(x=1.0, color='#555', linestyle=':', alpha=0.5, lw=1)
ax.annotate('1.0% AEP budget', xy=(1.02, 95), fontsize=8, color='#555', rotation=90)

# Highlight the high-value quadrant
ax.fill_between([0, 1.0], 50, 100, color='#27AE60', alpha=0.05)
ax.annotate('High-value zone:\n<1% AEP cost\n>50% risk reduction',
           xy=(0.5, 75), fontsize=9, ha='center', color='#27AE60', fontweight='bold')

ax.set_xlabel('AEP Cost (% of maximum annual generation)', fontsize=12)
ax.set_ylabel('Geometric Exposure Risk Reduction (%)', fontsize=12)
ax.set_title('Figure 5: Energy-Ecology Trade-off at 1% AEP Budget\n'
             'Green = radar-measured bird direction | Orange = coarse flyway proxy (NOT field-validated)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.set_ylim(-5, 105)
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(alpha=0.2)

# Stats box
stats_text = (
    f'Radar-measured (37 farms):\n'
    f'  Mean AEP cost: {radar_sub["aep_cost_pct"].mean():.3f}%\n'
    f'  Mean risk reduction: {radar_sub["relative_risk_reduction"].mean()*100:.1f}%\n'
    f'  Farms >50% reduction: {(radar_sub["relative_risk_reduction"]>0.5).sum()}/37\n\n'
    f'Coarse flyway (134 farms):\n'
    f'  Data are PROXY estimates only.\n'
    f'  Risk reduction driven by sin2\n'
    f'  model geometry, not bird data.\n'
    f'  NOT a scientific claim.'
)
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#FDF2F2', edgecolor='#E67E22', alpha=0.9),
        family='monospace')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_tradeoff_scatter.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig5_tradeoff_scatter.png')

# ============================================================
# FIG 6: Top 20 Radar Farms - bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
top20 = radar_sub.nlargest(20, 'relative_risk_reduction').sort_values('relative_risk_reduction')
y_labels = [f'Farm {int(r["farm_id"])} ({r["country"]})' for _, r in top20.iterrows()]

bars = ax.barh(range(20), top20['relative_risk_reduction']*100, color='#27AE60', alpha=0.85, edgecolor='white')
# Add AEP cost labels
for i, (_, r) in enumerate(top20.iterrows()):
    ax.text(r['relative_risk_reduction']*100 + 1, i, f'-{r["aep_cost_pct"]:.2f}% AEP', fontsize=8, va='center')

ax.set_yticks(range(20))
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Geometric Exposure Risk Reduction at 1% AEP Budget (%)', fontsize=12)
ax.set_title('Figure 6: Top 20 Radar-Measured Farms — Risk Reduction at <1% AEP Cost\n'
             '(nlhrw/deess VPTS 200m layer, nighttime high-density observations)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 105)
ax.grid(axis='x', alpha=0.2)
ax.axvline(x=50, color='#555', linestyle=':', alpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig6_radar_top20.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig6_radar_top20.png')

# ============================================================
# FIG 7: Budget sensitivity - how does risk reduction scale?
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

budget_levels = [0.005, 0.01, 0.02, 0.05]
for grp, color, label in [
    ('Europe (direction data)', '#27AE60', 'Radar-measured (37 farms)'),
    ('East Asia', '#E74C3C', 'East Asia (88 farms, proxy)'),
    ('Europe (no direction data)', '#F39C12', 'Europe no-data (41 farms, proxy)'),
]:
    sub = td[td['group'] == grp]
    means = [sub[sub['budget']==b]['relative_risk_reduction'].mean()*100 for b in budget_levels]
    stds = [sub[sub['budget']==b]['relative_risk_reduction'].std()*100 for b in budget_levels]
    ax.plot([b*100 for b in budget_levels], means, 'o-', color=color, lw=2.5, label=label, markersize=8)
    ax.fill_between([b*100 for b in budget_levels],
                    [max(0, m-s) for m, s in zip(means, stds)],
                    [min(100, m+s) for m, s in zip(means, stds)],
                    color=color, alpha=0.1)

ax.set_xlabel('AEP Budget (% of maximum annual generation)', fontsize=12)
ax.set_ylabel('Mean Risk Reduction (%)', fontsize=12)
ax.set_title('Figure 7: Diminishing Returns — Risk Reduction vs AEP Budget\n'
             '(error bands = +/-1 std across farms in group)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.grid(alpha=0.2)
ax.set_xlim(0.2, 5.5)
ax.set_ylim(0, 105)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig7_budget_sensitivity.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig7_budget_sensitivity.png')

# ============================================================
# FIG 8: Per-country summary of opportunities
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Only radar-measured countries
radar_countries = budget01[budget01['group']=='Europe (direction data)'].groupby('country').agg(
    n_farms=('farm_id', 'nunique'),
    mean_risk_red=('relative_risk_reduction', 'mean'),
    mean_aep_cost=('aep_cost_pct', 'mean'),
    total_cap_mw=('capacity_kW', 'sum'),
).reset_index()
radar_countries = radar_countries.sort_values('mean_risk_red')

x_pos = range(len(radar_countries))
width = 0.35

bars1 = ax.barh([x + width/2 for x in x_pos], radar_countries['mean_risk_red']*100,
                width, color='#27AE60', alpha=0.85, edgecolor='white', label='Mean risk reduction (%)')
ax.set_yticks([x + width/2 for x in x_pos])
ax.set_yticklabels([f'{r["country"]}\n({int(r["n_farms"])} farms, {r["total_cap_mw"]/1e3:.1f} GW)'
                     for _, r in radar_countries.iterrows()], fontsize=9)
ax.set_xlabel('Mean Risk Reduction at 1% AEP Budget (%)', fontsize=12)
ax.set_title('Figure 8: Radar-Measured Farms — Mean Trade-off by Country\n'
             '(nlhrw/deess VPTS radar, 200m layer, nighttime high-density events)',
             fontsize=13, fontweight='bold')
ax.axvline(x=50, color='#555', linestyle=':', alpha=0.5)
ax.grid(axis='x', alpha=0.2)

# Add AEP cost annotation
for i, (_, r) in enumerate(radar_countries.iterrows()):
    ax.text(r['mean_risk_red']*100 + 1, i + width/2,
            f'AEP -{r["mean_aep_cost"]:.3f}%', fontsize=8, va='center')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig8_radar_countries.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig8_radar_countries.png')

# Sync to shared folder
import shutil
dst = r'C:\Users\26841\Desktop\候鸟风场生态项目_数据共享\06_可视化图表'
for f in ['fig5_tradeoff_scatter.png','fig6_radar_top20.png','fig7_budget_sensitivity.png','fig8_radar_countries.png']:
    shutil.copy2(os.path.join(out_dir, f), os.path.join(dst, f))
    print(f'Synced: {f}')
print('Done')
