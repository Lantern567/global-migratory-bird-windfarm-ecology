"""
Generate individual onshore trade-off figures (split from fig14 dashboard).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

C = {'onshore': '#2E86AB', 'offshore': '#A23B72', 'highlight': '#F18F01', 'grid': '#CCCCCC'}

# ---- Load ----
t = pd.read_csv(os.path.join(PROC_DIR, 'onshore_tradeoff_results.csv'))
b1 = t[t['budget'] == 0.01].copy()
off = pd.read_csv(os.path.join(PROC_DIR, 'tradeoff_all_171.csv'))
o1 = off[off['budget'] == 0.01].copy()
oradar = o1[o1['group'] == 'Europe (direction data)'].copy()
# CRITICAL: offshore risk_reduction is stored as fraction (0-1), convert to % (0-100)
oradar['risk_reduction'] = oradar['risk_reduction'] * 100
print(f"Onshore: {len(b1)} farms, Offshore radar: {len(oradar)} farms")

# ======== FIG 14: Trade-off Scatter ========
fig, ax = plt.subplots(figsize=(8, 6))
np.random.seed(1)
jx = np.random.normal(0, 0.006, len(b1))
jy = np.random.normal(0, 0.6, len(b1))
ax.scatter(b1['aep_cost_pct']+jx, b1['risk_reduction']+jy,
           c=C['onshore'], alpha=0.12, s=5, edgecolors='none', label=f'Onshore (n={len(b1)})')
ojx = np.random.normal(0, 0.006, len(oradar))
ojy = np.random.normal(0, 0.6, len(oradar))
ax.scatter(oradar['aep_cost_pct']+ojx, oradar['risk_reduction']+ojy,
           c=C['offshore'], alpha=0.55, s=20, edgecolors='white', linewidth=0.3,
           label=f'Offshore (n={len(oradar)})')
ax.axvline(b1['aep_cost_pct'].mean(), color=C['onshore'], ls='--', lw=1, alpha=0.8)
ax.axhline(b1['risk_reduction'].mean(), color=C['onshore'], ls='--', lw=1, alpha=0.8)
ax.axvline(oradar['aep_cost_pct'].mean(), color=C['offshore'], ls='--', lw=1, alpha=0.8)
ax.axhline(oradar['risk_reduction'].mean(), color=C['offshore'], ls='--', lw=1, alpha=0.8)
ax.text(0.98, 0.22,
    f'Onshore  {b1["aep_cost_pct"].mean():.3f}% AEP  ->  {b1["risk_reduction"].mean():.1f}% RR (med {b1["risk_reduction"].median():.0f}%)\n'
    f'Offshore {oradar["aep_cost_pct"].mean():.3f}% AEP  ->  {oradar["risk_reduction"].mean():.1f}% RR (med {oradar["risk_reduction"].median():.0f}%)',
    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
ax.set_xlabel('AEP Cost (%)'); ax.set_ylabel('Risk Reduction (%)')
ax.set_title('Onshore vs Offshore Energy-Ecology Trade-off (1% AEP Budget)', fontweight='bold')
ax.legend(loc='upper left', markerscale=2)
ax.set_xlim(0, 1.02); ax.set_ylim(-5, 103)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig14_onshore_tradeoff_scatter.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig14 saved')

# ======== FIG 15: Risk Reduction Distribution + Budget Sensitivity ========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: ECDF — better than histogram for bimodal distribution
rr_on = np.sort(b1['risk_reduction'].values)
cy_on = np.arange(1, len(rr_on)+1) / len(rr_on)
ax1.step(rr_on, cy_on, where='post', color=C['onshore'], lw=1.5,
         label=f'Onshore (n={len(b1)}, med={b1["risk_reduction"].median():.0f}%)')
rr_off = np.sort(oradar['risk_reduction'].values)
cy_off = np.arange(1, len(rr_off)+1) / len(rr_off)
ax1.step(rr_off, cy_off, where='post', color=C['offshore'], lw=1.5,
         label=f'Offshore (n={len(oradar)}, med={oradar["risk_reduction"].median():.0f}%)')
ax1.axhline(0.5, color='gray', ls=':', lw=0.6, alpha=0.5)
ax1.axvline(b1['risk_reduction'].median(), color=C['onshore'], ls='--', lw=1, alpha=0.6)
ax1.axvline(oradar['risk_reduction'].median(), color=C['offshore'], ls='--', lw=1, alpha=0.6)
# Key fractions
p90_on = np.percentile(b1['risk_reduction'], 90)
ax1.annotate(f'90% farms\n>{p90_on:.0f}% RR', xy=(p90_on, 0.10),
             xytext=(p90_on-35, 0.25), fontsize=7, color=C['onshore'],
             arrowprops=dict(arrowstyle='->', color=C['onshore'], lw=0.8))
ax1.set_xlabel('Risk Reduction (%)'); ax1.set_ylabel('Cumulative Fraction')
ax1.set_title('Risk Reduction Cumulative Distribution', fontweight='bold')
ax1.legend(fontsize=7, loc='lower right')
ax1.set_xlim(-2, 102); ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, alpha=0.2)

# Panel 2: Budget sensitivity
budgets = sorted(t['budget'].unique())
mean_rr, med_rr, mean_aep, p25, p75 = [], [], [], [], []
for bv in budgets:
    sub = t[t['budget'] == bv]
    mean_rr.append(sub['risk_reduction'].mean()); med_rr.append(sub['risk_reduction'].median())
    p25.append(sub['risk_reduction'].quantile(0.25)); p75.append(sub['risk_reduction'].quantile(0.75))
    mean_aep.append(sub['aep_cost_pct'].mean())
bp = [b*100 for b in budgets]
ax2b = ax2.twinx()
l1, = ax2.plot(bp, mean_rr, 'o-', color=C['onshore'], lw=2, ms=5, label='Mean Risk Reduction')
l2, = ax2.plot(bp, med_rr, 's-', color=C['highlight'], lw=2, ms=5, label='Median Risk Reduction')
ax2.fill_between(bp, p25, p75, color=C['onshore'], alpha=0.12, label='IQR')
l3, = ax2b.plot(bp, mean_aep, 'D--', color=C['offshore'], lw=1.5, ms=5, label='Mean AEP Cost')
ax2.set_xlabel('AEP Budget (%)'); ax2.set_ylabel('Risk Reduction (%)')
ax2b.set_ylabel('AEP Cost (%)')
ax2.set_title('Budget Sensitivity', fontweight='bold')
lines = [l1, l2, l3]; ax2.legend(lines, [x.get_label() for x in lines], loc='lower right', fontsize=7)
ax2.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig15_onshore_distribution_sensitivity.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig15 saved')

# ======== FIG 16: Theta_econ vs Theta_eco ========
fig, ax = plt.subplots(figsize=(7, 6))
te = np.round(b1['theta_econ'] / 10) * 10 % 180
tco = np.round(b1['theta_eco'] / 10) * 10 % 180
xb = np.arange(-5, 185, 10)
hist, xe, ye = np.histogram2d(te, tco, bins=[xb, xb])
im = ax.pcolormesh(xe, ye, hist.T, cmap='Blues', shading='auto')
ax.plot([0, 180], [0, 180], 'k--', lw=0.6, alpha=0.4)
cbar = plt.colorbar(im, ax=ax, label='Number of Farms', shrink=0.85)
ax.set_xlabel('Economic Optimum Theta (deg)'); ax.set_ylabel('Ecological Optimum Theta (deg)')
ax.set_title('theta_econ vs theta_eco at 1% AEP Budget', fontweight='bold')
ax.text(0.03, 0.97, f'{hist.sum():.0f} farms', transform=ax.transAxes, va='top', fontsize=10, fontweight='bold')
# Annotate dominant region
cnt_40_60 = ((te >= 30) & (te <= 60) & (tco >= 30) & (tco <= 60)).sum()
ax.text(0.97, 0.03, f'theta_eco = 40-60 deg:\n{cnt_40_60} farms ({cnt_40_60/hist.sum()*100:.0f}%)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig16_onshore_theta_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig16 saved')

# ======== FIG 17: Stratification — Farm Size + Distance ========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Farm size
size_bins = [(3, 5, '3-5'), (5, 10, '5-10'), (10, 20, '10-20'), (20, 50, '20-50'), (50, 1000, '50+')]
labels = [s[2] for s in size_bins]
rr_m, aep_m, ns = [], [], []
for lo, hi, _ in size_bins:
    sub = b1[(b1['n_turbines'] >= lo) & (b1['n_turbines'] < hi)]
    rr_m.append(sub['risk_reduction'].mean()); aep_m.append(sub['aep_cost_pct'].mean()); ns.append(len(sub))
x = np.arange(len(labels)); w = 0.35
ax1b = ax1.twinx()
b1a = ax1.bar(x - w/2, rr_m, w, color=C['onshore'], alpha=0.85, label='Risk Reduction (%)')
b1b = ax1b.bar(x + w/2, aep_m, w, color=C['offshore'], alpha=0.85, label='AEP Cost (%)')
for i, (rr, aep, n) in enumerate(zip(rr_m, aep_m, ns)):
    ax1.text(i - w/2, rr + 0.5, f'{rr:.1f}%', ha='center', fontsize=7)
    ax1b.text(i + w/2, aep + 0.005, f'{aep:.3f}%', ha='center', fontsize=7)
    ax1.text(i, 2, f'n={n}', ha='center', fontsize=6, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_xlabel('Turbines per Farm'); ax1.set_ylabel('Risk Reduction (%)')
ax1b.set_ylabel('AEP Cost (%)')
ax1.set_title('By Farm Size', fontweight='bold')
ax1.legend(loc='upper left', fontsize=7); ax1b.legend(loc='upper right', fontsize=7)
ax1.set_ylim(75, 88)

# Distance decay
db = [(0, 25, '0-25'), (25, 50, '25-50'), (50, 100, '50-100'), (100, 200, '100-200')]
d_labels = [d[2] for d in db]
d_rr, d_n = [], []
for lo, hi, _ in db:
    sub = b1[(b1['grid_dist_km'] >= lo) & (b1['grid_dist_km'] < hi)]
    d_rr.append(sub['risk_reduction'].mean()); d_n.append(len(sub))
xd = np.arange(len(d_labels))
ax2.bar(xd, d_rr, color=C['onshore'], alpha=0.75)
for i, (rr, n) in enumerate(zip(d_rr, d_n)):
    ax2.text(i, rr + 0.5, f'{rr:.1f}%\n(n={n})', ha='center', fontsize=7)
ax2.set_xticks(xd); ax2.set_xticklabels(d_labels)
ax2.set_xlabel('Distance to Bauer Grid Cell (km)')
ax2.set_ylabel('Mean Risk Reduction (%)')
ax2.set_title('Distance Decay', fontweight='bold')
ax2.set_ylim(70, 86)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig17_onshore_stratification.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig17 saved')

# ======== FIG 18: Spatial Map ========
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(b1['centroid_lon'], b1['centroid_lat'],
                c=b1['risk_reduction'], cmap='RdYlGn', s=3, alpha=0.45, vmin=0, vmax=100)
cbar = plt.colorbar(sc, ax=ax, label='Risk Reduction (%)', shrink=0.8)
# Bauer grid outline
for y in [43, 55]:
    ax.axhline(y, color=C['grid'], ls='--', lw=0.6, alpha=0.4)
for x in [-5, 16]:
    ax.axvline(x, color=C['grid'], ls='--', lw=0.6, alpha=0.4)
ax.text(16.8, 43.5, 'Bauer\ngrid\nboundary', fontsize=7, alpha=0.4, ha='center')
# Country labels
countries = {'UK': (-3, 54.5), 'France': (2.5, 46.8), 'Germany': (9.5, 51),
             'Denmark': (9, 56), 'Netherlands': (5.5, 52.5), 'Belgium': (5, 50.5),
             'Spain': (-3.5, 40), 'Sweden': (15, 60.5), 'Poland': (18, 52)}
for name, (lon, lat) in countries.items():
    ax.text(lon, lat, name, fontsize=8, ha='center', alpha=0.35, fontweight='bold')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.set_title('Spatial Distribution of Risk Reduction at 1% AEP Budget', fontweight='bold')
ax.set_xlim(-10, 22); ax.set_ylim(36, 68)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig18_onshore_spatial_map.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig18 saved')

# ======== FIG 19: Wake Efficiency Polar (top 3 farms) ========
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onshore_aep_computation as oac

aep_df = pd.read_csv(os.path.join(PROC_DIR, 'onshore_aep_curves.csv'))
top4 = aep_df.nlargest(4, 'n_turbines')
turb_full = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'osm_turbines_clustered.csv'))
if 'rotor_diam' not in turb_full.columns:
    turb_full['rotor_diam'] = 70.0
turb_full['rotor_diam'] = turb_full['rotor_diam'].fillna(70.0)
turb_map = {fid: g for fid, g in turb_full.groupby('farm_id')}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'})

for ax_i, (_, farm) in enumerate(top4.iterrows()):
    if ax_i >= 3:
        break
    fid = farm['farm_id']
    turb = turb_map.get(fid)
    if turb is None:
        continue
    lat_m = turb['lat'].mean(); cos_lat = np.cos(np.radians(lat_m))
    tx = (turb['lon'].values - turb['lon'].mean()) * 111320 * cos_lat
    ty = (turb['lat'].values - turb['lat'].mean()) * 111320
    rd = turb['rotor_diam'].values

    rd_f = farm['rotor_diam']
    if pd.isna(rd_f) or rd_f <= 0: rd_f = 70.0
    cap = farm['capacity_kw'] if not pd.isna(farm['capacity_kw']) else 2000.0
    pc, vr, _ = oac.build_power_curve(rd_f, cap)
    tc = oac.build_thrust_curve(vr)
    oac.power_curve_global = pc

    wd_fine = np.arange(0, 360, 5)
    eta = np.array([oac.compute_farm_wake_efficiency(tx, ty, rd, wd, tc, vr) for wd in wd_fine])
    pc1 = farm['pc1_angle']; evr = farm['explained_var_ratio']

    ax = axes[ax_i]
    ax.plot(np.radians(wd_fine), eta, color=C['onshore'], lw=1.3)
    ax.fill(np.radians(wd_fine), eta, color=C['onshore'], alpha=0.12)
    ax.axvline(np.radians(pc1), color=C['highlight'], ls='--', lw=1.5, label=f'PCA axis = {pc1:.0f} deg')
    ax.axvline(np.radians((pc1 + 90) % 180), color=C['highlight'], ls=':', lw=1, alpha=0.4)
    ax.set_title(f'Farm {int(fid)}: {len(turb)} turbines, {rd_f:.0f}m rotor\nPCA EVR = {evr:.2f}', fontsize=10, fontweight='bold')
    ax.set_ylim(0.84, 1.02)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)

fig.suptitle('Jensen Wake Model — Farm Efficiency vs Wind Direction', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig19_wake_efficiency_polar.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig19 saved')

# ======== FIG 20: Onshore AEP Orientation Sensitivity ========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Best theta distribution
aep_cols = [f'aep_{int(t):03d}' for t in range(0, 180, 10)]
vals = aep_df[aep_cols].values
best_idx = np.argmax(vals, axis=1)
theta_best = np.arange(0, 180, 10)[best_idx]
theta_counts = [(theta_best == t).sum() for t in range(0, 180, 10)]
ax1.bar(np.arange(0, 180, 10), theta_counts, width=8, color=C['onshore'], alpha=0.8, edgecolor='white')
ax1.set_xlabel('Best Theta (deg)'); ax1.set_ylabel('Number of Farms')
ax1.set_title(f'Distribution of Economic Optimum Theta\n({len(aep_df)} farms)', fontweight='bold')

# Min/Max ratio histogram — trim x-axis to data-rich region (95% of farms >0.91)
ratio = vals.min(axis=1) / np.maximum(vals.max(axis=1), 1e-6)
# Use focused bins covering the actual data range
focused_bins = np.linspace(0.80, 1.005, 45)
ax2.hist(ratio, bins=focused_bins, color=C['offshore'], alpha=0.6, edgecolor='white')
ax2.axvline(ratio.mean(), color=C['offshore'], ls='--', lw=1.5, label=f'Mean = {ratio.mean():.4f}')
ax2.axvline(np.median(ratio), color=C['onshore'], ls='--', lw=1.5, label=f'Median = {np.median(ratio):.4f}')
# Mark the <0.80 tail count
n_tail = (ratio < 0.80).sum()
if n_tail > 0:
    ax2.text(0.805, 0.85, f'{n_tail} farms\n<0.80', transform=ax2.transAxes,
             fontsize=7, color='gray', ha='left', va='top')
ax2.set_xlabel('AEP(min) / AEP(max)'); ax2.set_ylabel('Number of Farms')
ax2.set_title('AEP Orientation Sensitivity', fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_xlim(0.80, 1.005)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'fig20_onshore_aep_sensitivity.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('fig20 saved')

print(f'\nAll figures saved to: {FIGS_DIR}')
print('Files: fig14-fig20')
