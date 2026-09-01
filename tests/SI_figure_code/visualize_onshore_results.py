"""
Onshore energy-ecology trade-off visualization figures.
Generates Figure 14: Onshore comprehensive dashboard (multi-panel)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, sys

# ---- Config ----
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
PROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'DejaVu Sans', 'Arial'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})

# ---- Load data ----
tradeoff = pd.read_csv(os.path.join(PROC_DIR, 'onshore_tradeoff_results.csv'))
budget_1pct = tradeoff[tradeoff['budget'] == 0.01].copy()
print(f"Onshore: {len(budget_1pct)} farms at 1% budget")

# Load offshore for comparison
offshore = pd.read_csv(os.path.join(PROC_DIR, 'tradeoff_all_171.csv'))
offshore_1pct = offshore[offshore['budget'] == 0.01].copy()
offshore_radar = offshore_1pct[offshore_1pct['group'] == 'Europe (direction data)'].copy()
print(f"Offshore radar: {len(offshore_radar)} farms at 1% budget")

Colors = {
    'onshore': '#2E86AB',
    'offshore': '#A23B72',
    'grid': '#DDDDDD',
    'highlight': '#F18F01',
}

# ================================================================
# FIGURE 14: Onshore Trade-off Dashboard (6 panels)
# ================================================================
fig = plt.figure(figsize=(14, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

# ---- Panel A: Scatter: AEP cost vs Risk Reduction ----
ax = fig.add_subplot(gs[0, :2])
# Onshore scatter
jitter_x = np.random.normal(0, 0.008, len(budget_1pct))
jitter_y = np.random.normal(0, 0.8, len(budget_1pct))
ax.scatter(
    budget_1pct['aep_cost_pct'] + jitter_x,
    budget_1pct['risk_reduction'] + jitter_y,
    c=Colors['onshore'], alpha=0.15, s=4, edgecolors='none',
    label=f'Onshore (n={len(budget_1pct)})'
)
# Offshore scatter
off_jitter_x = np.random.normal(0, 0.008, len(offshore_radar))
off_jitter_y = np.random.normal(0, 0.8, len(offshore_radar))
ax.scatter(
    offshore_radar['aep_cost_pct'] + off_jitter_x,
    offshore_radar['risk_reduction'] + off_jitter_y,
    c=Colors['offshore'], alpha=0.5, s=15, edgecolors='white', linewidth=0.3,
    label=f'Offshore (n={len(offshore_radar)})'
)
# Means
ax.axvline(budget_1pct['aep_cost_pct'].mean(), color=Colors['onshore'], linestyle='--', alpha=0.7, linewidth=1)
ax.axhline(budget_1pct['risk_reduction'].mean(), color=Colors['onshore'], linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(offshore_radar['aep_cost_pct'].mean(), color=Colors['offshore'], linestyle='--', alpha=0.7, linewidth=1)
ax.axhline(offshore_radar['risk_reduction'].mean(), color=Colors['offshore'], linestyle='--', alpha=0.7, linewidth=1)

# Annotate key stats
ax.text(0.98, 0.25,
    f'Onshore: {budget_1pct["aep_cost_pct"].mean():.3f}% AEP → {budget_1pct["risk_reduction"].mean():.1f}% RR\n'
    f'Offshore: {offshore_radar["aep_cost_pct"].mean():.3f}% AEP → {offshore_radar["risk_reduction"].mean():.1f}% RR',
    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.set_xlabel('AEP Cost (%)')
ax.set_ylabel('Risk Reduction (%)')
ax.set_title('A: Energy-Ecology Trade-off at 1% AEP Budget', fontweight='bold')
ax.legend(loc='upper left', markerscale=2)
ax.set_xlim(0, 1.0)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

# ---- Panel B: Risk Reduction histogram (onshore vs offshore) ----
ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 100, 41)
ax.hist(budget_1pct['risk_reduction'], bins=bins, color=Colors['onshore'], alpha=0.5,
        label=f'Onshore\n(med={budget_1pct["risk_reduction"].median():.0f}%)', density=True)
ax.hist(offshore_radar['risk_reduction'], bins=bins, color=Colors['offshore'], alpha=0.5,
        label=f'Offshore\n(med={offshore_radar["risk_reduction"].median():.0f}%)', density=True)
ax.axvline(budget_1pct['risk_reduction'].median(), color=Colors['onshore'], linestyle='--', linewidth=1)
ax.axvline(offshore_radar['risk_reduction'].median(), color=Colors['offshore'], linestyle='--', linewidth=1)
ax.set_xlabel('Risk Reduction (%)')
ax.set_ylabel('Density')
ax.set_title('B: Risk Reduction Distribution', fontweight='bold')
ax.legend(fontsize=7)

# ---- Panel C: Budget sensitivity ----
ax = fig.add_subplot(gs[1, 0])
budgets = sorted(tradeoff['budget'].unique())
means_rr, meds_rr, means_aep = [], [], []
p25_rr, p75_rr = [], []
for b in budgets:
    sub = tradeoff[tradeoff['budget'] == b]
    means_rr.append(sub['risk_reduction'].mean())
    meds_rr.append(sub['risk_reduction'].median())
    p25_rr.append(sub['risk_reduction'].quantile(0.25))
    p75_rr.append(sub['risk_reduction'].quantile(0.75))
    means_aep.append(sub['aep_cost_pct'].mean())

budget_pct = [b*100 for b in budgets]

ax2 = ax.twinx()
l1, = ax.plot(budget_pct, means_rr, 'o-', color=Colors['onshore'], linewidth=2, markersize=5, label='Mean RR')
l2, = ax.plot(budget_pct, meds_rr, 's-', color=Colors['highlight'], linewidth=2, markersize=5, label='Median RR')
ax.fill_between(budget_pct, p25_rr, p75_rr, color=Colors['onshore'], alpha=0.15, label='IQR')
l3, = ax2.plot(budget_pct, means_aep, 'D--', color=Colors['offshore'], linewidth=1.5, markersize=5, label='Mean AEP Cost')

ax.set_xlabel('AEP Budget (%)')
ax.set_ylabel('Risk Reduction (%)')
ax2.set_ylabel('AEP Cost (%)')
ax.set_title('C: Budget Sensitivity', fontweight='bold')
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='lower right', fontsize=7)
ax.set_ylim(50, 100)
ax.grid(True, alpha=0.3)

# ---- Panel D: theta_econ vs theta_eco ----
ax = fig.add_subplot(gs[1, 1])
theta_both = np.column_stack([budget_1pct['theta_econ'], budget_1pct['theta_eco']])
# Round to 10-degree bins
theta_econ_rounded = np.round(budget_1pct['theta_econ'] / 10) * 10 % 180
theta_eco_rounded = np.round(budget_1pct['theta_eco'] / 10) * 10 % 180

hist, xedges, yedges = np.histogram2d(theta_econ_rounded, theta_eco_rounded,
                                       bins=[np.arange(-5, 185, 10), np.arange(-5, 185, 10)])
im = ax.pcolormesh(xedges, yedges, hist.T, cmap='YlOrRd', shading='auto')
ax.plot([0, 180], [0, 180], 'k--', linewidth=0.5, alpha=0.5)
plt.colorbar(im, ax=ax, label='Farms', shrink=0.8)

# Annotate dominant quadrants
ax.text(0.05, 0.95, f'{hist.sum():.0f} farms', transform=ax.transAxes,
        va='top', fontsize=8, fontweight='bold')
ax.set_xlabel('Economic Optimum Theta (deg)')
ax.set_ylabel('Ecological Optimum Theta (deg)')
ax.set_title('D: Theta_econ vs Theta_eco (1% Budget)', fontweight='bold')

# ---- Panel E: Farm size stratification ----
ax = fig.add_subplot(gs[1, 2])
size_bins = [(3, 5, '3-5'), (5, 10, '5-10'), (10, 20, '10-20'), (20, 50, '20-50'), (50, 1000, '50+')]
size_labels = [b[2] for b in size_bins]
rr_means = []
rr_stds = []
aep_means = []
ns = []
budget_1pct_copy = budget_1pct.copy()
for (lo, hi, label) in size_bins:
    sub = budget_1pct_copy[(budget_1pct_copy['n_turbines'] >= lo) & (budget_1pct_copy['n_turbines'] < hi)]
    rr_means.append(sub['risk_reduction'].mean())
    rr_stds.append(sub['risk_reduction'].std())
    aep_means.append(sub['aep_cost_pct'].mean())
    ns.append(len(sub))

x = np.arange(len(size_labels))
w = 0.35
bars1 = ax.bar(x - w/2, rr_means, w, color=Colors['onshore'], alpha=0.8, label='Risk Reduction (%)')
ax2 = ax.twinx()
bars2 = ax2.bar(x + w/2, aep_means, w, color=Colors['offshore'], alpha=0.8, label='AEP Cost (%)')

for i, (rr, aep, n) in enumerate(zip(rr_means, aep_means, ns)):
    ax.text(i - w/2, rr + 1, f'{rr:.1f}%', ha='center', fontsize=6)
    ax2.text(i + w/2, aep + 0.01, f'{aep:.3f}%', ha='center', fontsize=6)
    ax.text(i, 3, f'n={n}', ha='center', fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels(size_labels)
ax.set_xlabel('Turbines per Farm')
ax.set_ylabel('Risk Reduction (%)')
ax2.set_ylabel('AEP Cost (%)')
ax.set_title('E: By Farm Size (1% Budget)', fontweight='bold')
ax.legend(loc='upper left', fontsize=7)
ax2.legend(loc='upper right', fontsize=7)
ax.set_ylim(75, 88)

# ---- Panel F: Spatial map (lat/lon colored by risk reduction) ----
ax = fig.add_subplot(gs[2, :2])
sc = ax.scatter(
    budget_1pct['centroid_lon'],
    budget_1pct['centroid_lat'],
    c=budget_1pct['risk_reduction'],
    cmap='RdYlGn', s=2, alpha=0.5, vmin=0, vmax=100
)
plt.colorbar(sc, ax=ax, label='Risk Reduction (%)', shrink=0.8)
# Bauer grid outline
ax.axhline(43, color=Colors['grid'], linestyle='--', linewidth=0.5, alpha=0.5)
ax.axhline(55, color=Colors['grid'], linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(-5, color=Colors['grid'], linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(16, color=Colors['grid'], linestyle='--', linewidth=0.5, alpha=0.5)
ax.text(16.5, 43.3, 'Bauer\ngrid', fontsize=6, alpha=0.5)
# Country labels
country_centers = {
    'UK': (-3, 54), 'France': (3, 47), 'Germany': (10, 51),
    'Denmark': (9, 56), 'Netherlands': (5, 52.5), 'Belgium': (4.5, 50.5),
    'Spain': (-3, 40.5), 'Sweden': (15, 60), 'Poland': (19, 52),
}
for name, (lon, lat) in country_centers.items():
    ax.text(lon, lat, name, fontsize=7, ha='center', alpha=0.4)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('F: Spatial Distribution of Risk Reduction (1% Budget)', fontweight='bold')
ax.set_xlim(-10, 22)
ax.set_ylim(36, 68)

# ---- Panel G: Distance decay ----
ax = fig.add_subplot(gs[2, 2])
dist_bins = [(0, 25, '0-25'), (25, 50, '25-50'), (50, 75, '50-75'),
             (75, 100, '75-100'), (100, 150, '100-150'), (150, 200, '150-200')]
dist_labels = [b[2] for b in dist_bins]
dist_rr = []
dist_n = []
for (lo, hi, label) in dist_bins:
    sub = budget_1pct_copy[(budget_1pct_copy['grid_dist_km'] >= lo) & (budget_1pct_copy['grid_dist_km'] < hi)]
    dist_rr.append(sub['risk_reduction'].mean())
    dist_n.append(len(sub))

x = np.arange(len(dist_labels))
bars = ax.bar(x, dist_rr, color=Colors['onshore'], alpha=0.7)
for i, (rr, n) in enumerate(zip(dist_rr, dist_n)):
    ax.text(i, rr + 1, f'{rr:.1f}%\n(n={n})', ha='center', fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(dist_labels)
ax.set_xlabel('Distance to Bauer Grid Cell (km)')
ax.set_ylabel('Mean Risk Reduction (%)')
ax.set_title('G: Distance Decay', fontweight='bold')
ax.set_ylim(70, 86)

# ---- Final touches ----
fig.suptitle('Onshore Wind Farm — Energy-Ecology Trade-off with Jensen Wake Model',
             fontsize=14, fontweight='bold', y=0.985)

# Add methodology annotation
fig.text(0.02, 0.005,
    f'Method: GWA wind resource (4,965 unique locs) + Jensen wake (α=0.075) + Bauer grid bird directions (2,025 cells) + sin² exposure model\n'
    f'Data: {len(budget_1pct)} onshore farms ({budget_1pct["n_turbines"].sum():,} turbines), 9 European countries. August 2026.',
    fontsize=6.5, style='italic', alpha=0.6)

out_path = os.path.join(FIGS_DIR, 'fig14_onshore_dashboard.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out_path}')

# ================================================================
# FIGURE 15: Jensen Wake Model Validation — Wake Efficiency Map
# ================================================================
# Take the 3 largest farms and show their wake efficiency as a function of wind direction
fig2, axes = plt.subplots(1, 3, figsize=(14, 4.5), subplot_kw={'projection': 'polar'})

# Find 3 largest farms with AEP data
aep_df = pd.read_csv(os.path.join(PROC_DIR, 'onshore_aep_curves.csv'))
top3 = aep_df.nlargest(3, 'n_turbines')

# For each, compute wake efficiency across 36 wind directions
turbines_full = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'osm_turbines_clustered.csv'))
if 'rotor_diam' not in turbines_full.columns:
    turbines_full['rotor_diam'] = 70.0
turbines_full['rotor_diam'] = turbines_full['rotor_diam'].fillna(70.0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onshore_aep_computation as oac

turb_farm_groups = {}
for fid, group in turbines_full.groupby('farm_id'):
    turb_farm_groups[fid] = group

for ax_i, (_, farm) in enumerate(top3.iterrows()):
    fid = farm['farm_id']
    turb = turb_farm_groups.get(fid)
    if turb is None:
        continue

    # Get turbine coordinates
    lat_mean = turb['lat'].mean()
    cos_lat = np.cos(np.radians(lat_mean))
    turbs_x = (turb['lon'].values - turb['lon'].mean()) * 111320 * cos_lat
    turbs_y = (turb['lat'].values - turb['lat'].mean()) * 111320
    rotor_diams = turb['rotor_diam'].values

    # Build thrust curve
    rotor_d = farm['rotor_diam']
    if pd.isna(rotor_d) or rotor_d <= 0:
        rotor_d = 70.0
    capacity = farm['capacity_kw'] if not pd.isna(farm['capacity_kw']) else 2000.0
    power_curve, v_rated, _ = oac.build_power_curve(rotor_d, capacity)
    thrust_curve = oac.build_thrust_curve(v_rated)

    # Compute efficiency for 72 wind directions
    wd_fine = np.arange(0, 360, 5)
    eta_vals = np.array([
        oac.compute_farm_wake_efficiency(turbs_x, turbs_y, rotor_diams, wd, thrust_curve, 8.0, power_curve)
        for wd in wd_fine
    ])

    # PCA angle for reference
    pc1 = farm['pc1_angle']
    evr = farm['explained_var_ratio']

    ax = axes[ax_i]
    ax.plot(np.radians(wd_fine), eta_vals, color=Colors['onshore'], linewidth=1.2)
    ax.fill(np.radians(wd_fine), eta_vals, color=Colors['onshore'], alpha=0.15)
    # Mark PCA axis direction
    ax.axvline(np.radians(pc1), color=Colors['highlight'], linestyle='--', linewidth=1.5, label=f'PCA={pc1:.0f}°')
    ax.axvline(np.radians((pc1 + 90) % 180), color=Colors['highlight'], linestyle=':', linewidth=1, alpha=0.5)

    ax.set_title(f'Farm {int(fid)} ({len(turb)} turb, hub={rotor_d:.0f}m)\nPCA EVR={evr:.2f}', fontsize=9, fontweight='bold')
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_rlabel_position(45)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

fig2.suptitle('Jensen Wake Model — Farm Efficiency vs Wind Direction',
              fontsize=13, fontweight='bold', y=1.02)
fig2.tight_layout()

out_path2 = os.path.join(FIGS_DIR, 'fig15_wake_efficiency_polar.png')
fig2.savefig(out_path2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'Saved: {out_path2}')

print('\nDone — figures 14-15 generated.')
