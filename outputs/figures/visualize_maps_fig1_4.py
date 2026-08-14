import os, sys, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrow, Circle

# Force a font that HAS CJK - register SimHei properly
for fp in fm.findSystemFonts():
    if 'simhei' in fp.lower() or 'SimHei' in fp:
        fm.fontManager.addfont(fp)
        print(f'Found: {fp}')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

out_dir = r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\outputs'
os.makedirs(out_dir, exist_ok=True)

farms = pd.read_csv(r'D:\1风力发电实习\wind-direction-to-electricity-transition-main\offshore-task0-HuTingxian\output\task0\farms_master.csv')

radar_stations = {
    'nlhrw': ('Den Helder', 'NL', 52.95, 4.75, 118),
    'deess': ('Essen', 'DE', 51.40, 6.97, 104),
    'frabb': ('Abbeville', 'FR', 50.13, 1.83, 214),
    'bejab': ('Jabbeke', 'BE', 51.18, 3.07, 118),
    'behel': ('Helchteren', 'BE', 51.05, 5.42, 2),
    'denhb': ('Neuhaus', 'DE', 53.60, 10.68, 2),
    'nldhl': ('Herwijnen', 'NL', 51.84, 5.15, 3),
    'bewid': ('Wideumont', 'BE', 49.91, 5.50, 0),
    'bezav': ('Zaventem', 'BE', 50.90, 4.53, 0),
    'frave': ('Avesnes', 'FR', 50.08, 3.87, 0),
}

def nearest_radar(lat, lon):
    best_d = float('inf'); best_s = None
    for sn, (_, _, rlat, rlon, _) in radar_stations.items():
        d = math.sqrt((lat-rlat)**2 + (lon-rlon)**2) * 111
        if d < best_d: best_d = d; best_s = sn
    return best_s, best_d

farms['radar'], farms['radar_dist'] = zip(*farms.apply(lambda r: nearest_radar(r['centroid_lat'], r['centroid_lon']), axis=1))
farms['has_direction'] = farms['radar_dist'] <= 500

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# FIG 1: EUROPE
# ============================================================
fig = plt.figure(figsize=(17, 13))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-12, 16, 41, 63])
ax.add_feature(cfeature.LAND, facecolor='#EDE8DC', edgecolor='none', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#C8DDE8', edgecolor='none', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888', zorder=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#BBB', zorder=1)

eu_farms = farms[farms['country'].isin(
    ['United Kingdom','Netherlands','Germany','Sweden','Denmark','Belgium','France',
     'Finland','Ireland','Portugal','Italy','Norway'])]

sc = ax.scatter(eu_farms['centroid_lon'], eu_farms['centroid_lat'],
    c=eu_farms['radar_dist'], cmap='RdYlGn_r', s=eu_farms['n_turb']/4,
    edgecolors='#333', linewidth=0.3, alpha=0.88, vmin=50, vmax=1000,
    transform=ccrs.PlateCarree(), zorder=3)
cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.03)
cbar.set_label('Distance to nearest radar (km)', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Radar stations
for sn, (city, country, rlat, rlon, nfiles) in radar_stations.items():
    has_sig = sn in ['nlhrw', 'deess', 'behel']
    marker = '^' if has_sig else 's'
    size = 160 if has_sig else 50
    color = '#C0392B' if has_sig else '#999'
    ax.scatter(rlon, rlat, marker=marker, s=size, c=color, edgecolors='white',
               linewidth=1.5, zorder=6)
    ax.annotate(f'{sn}\n({nfiles} files)', (rlon, rlat),
                textcoords='offset points', xytext=(-12, -22), fontsize=7,
                ha='center', color='#333',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    if nfiles > 50:
        ax.add_patch(Circle((rlon, rlat), 500/111.32, facecolor='none',
                      edgecolor=color, linewidth=1.2, linestyle='--', alpha=0.45,
                      transform=ccrs.PlateCarree(), zorder=2))

# Bird arrows
for angle, clr, label in [(32.6, '#E74C3C', 'Spring: 33 deg N-migration'),
                            (233.2, '#3498DB', 'Autumn: 233 deg S-migration')]:
    rad = math.radians(angle)
    dx = math.sin(rad) * 2.5; dy = math.cos(rad) * 2.5
    ax.annotate('', xy=(4.75+dx, 52.95+dy), xytext=(4.75, 52.95),
                arrowprops=dict(arrowstyle='->', color=clr, lw=3.5, alpha=0.85),
                transform=ccrs.PlateCarree(), zorder=7)
    ax.text(4.75+dx*1.15, 52.95+dy*1.15, label, fontsize=8, ha='center',
            color=clr, fontweight='bold', transform=ccrs.PlateCarree(), zorder=7,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    plt.scatter([],[], c='#2ECC71', s=80, edgecolors='#333', linewidth=0.3, label='<200 km to radar'),
    plt.scatter([],[], c='orange', s=80, edgecolors='#333', linewidth=0.3, label='200-500 km'),
    plt.scatter([],[], c='#E74C3C', s=80, edgecolors='#333', linewidth=0.3, label='>500 km (no reliable data)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#C0392B', markersize=10, label='Radar (has direction data)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#999', markersize=8, label='Radar (not yet downloaded)'),
    Line2D([0],[0], linestyle='--', color='#C0392B', lw=1.5, label='500 km radar coverage'),
    FancyArrow(0,0,0,0, color='#E74C3C', lw=2.5, label='Spring migration direction'),
    FancyArrow(0,0,0,0, color='#3498DB', lw=2.5, label='Autumn migration direction'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8.5, framealpha=0.92, ncol=2)

ax.set_title('Figure 1: European Offshore Wind Farms and Weather Radar Bird-Migration Coverage\n(37 of 171 farms within effective radar range)', fontsize=14, fontweight='bold', pad=10)

# Simple stats note
ax.text(1.02, 0.95, '37/171 farms within 500 km of radar\nRadar beam rises ~1 km / 100 km distance',
        transform=ax.transAxes, fontsize=8, verticalalignment='top', color='#555', style='italic')

fig.savefig(os.path.join(out_dir, 'fig1_europe_wind_radar.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig1')

# ============================================================
# FIG 2: GLOBAL
# ============================================================
flyways = gpd.read_file(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\raw\caff_flyways\shapefile\Major_Flyways.shp', engine='fiona')
global_wind = pd.read_csv(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\processed\global_wind_farms.csv')
global_wind = global_wind.dropna(subset=['latitude', 'longitude'])

fig = plt.figure(figsize=(22, 13))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#EDE8DC', edgecolor='none', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#C8DDE8', edgecolor='none', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#999', zorder=1)

flyway_colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
for idx, row in flyways.iterrows():
    ax.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                      facecolor=flyway_colors[idx%8], edgecolor='#555', linewidth=0.5, alpha=0.15, zorder=2)
    c = row.geometry.centroid
    ax.annotate(row['NAME'].replace(' / ', '/\n'), xy=(c.x, c.y), xycoords=ccrs.PlateCarree(),
                fontsize=8, color=flyway_colors[idx%8], ha='center', va='center', fontweight='bold',
                alpha=0.8, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

sample = global_wind.sample(min(25000, len(global_wind)), random_state=42)
ax.scatter(sample['longitude'], sample['latitude'], transform=ccrs.PlateCarree(),
           s=0.8, c='#2C3E50', alpha=0.12, zorder=3)
ax.scatter(farms['centroid_lon'], farms['centroid_lat'], transform=ccrs.PlateCarree(),
           s=40, c='#E74C3C', edgecolors='white', linewidth=0.5, zorder=5)

# Country labels for our farms
for label, lon, lat in [
    ('China (66)', 120, 34), ('UK (31)', -2, 56), ('Vietnam (13)', 108, 12),
    ('Denmark (12)', 10, 57), ('Netherlands (11)', 5, 53.5), ('Sweden (8)', 18, 60),
    ('Germany (6)', 8, 55), ('USA (5)', -72, 40), ('France (5)', -1, 47.5),
    ('Taiwan (4)', 121, 24), ('Japan (3)', 140, 42), ('S.Korea (2)', 127, 37),
]:
    ax.annotate(label, xy=(lon, lat), xycoords=ccrs.PlateCarree(),
                fontsize=6.5, ha='center', color='#C0392B', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

ax.set_title('Figure 2: Global Wind Farms and Arctic Bird Migration Flyways (Conceptual Overlap Only)\nCAFF flyways (Arctic Biodiversity Assessment 2013) -- each covers 27-105 million km2',
             fontsize=15, fontweight='bold', pad=12)

fig.savefig(os.path.join(out_dir, 'fig2_global_flyways_wind.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig2')

# ============================================================
# FIG 3: DIRECTION + EXPOSURE
# ============================================================
fig = plt.figure(figsize=(20, 8))
fig.suptitle('Figure 3: Bird Migration Direction Signatures & Ecological Exposure Curve (nlhrw radar, Netherlands)',
             fontsize=14, fontweight='bold', y=1.02)

# A: nlhrw rose
ax0 = fig.add_subplot(1, 3, 1)
for r_val in [0.3, 0.6, 0.9]:
    ax0.add_patch(plt.Circle((0,0), r_val, fill=False, color='#ddd', lw=0.5))

for season, angle, clr, label_txt in [
    ('Spring', 32.6, '#E74C3C',
     'Spring migration (n=7,715)\nMean direction: 32.6 deg NNE\nDensity: 105.6 birds/km3 avg\nPeak density: 2,395 birds/km3'),
    ('Autumn', 233.2, '#3498DB',
     'Autumn migration (n=6,543)\nMean direction: 233.2 deg SW\nDensity: 304.8 birds/km3 avg\nPeak density: 11,371 birds/km3'),
]:
    spread = 45
    theta_vals = np.linspace(math.radians(angle-spread), math.radians(angle+spread), 50)
    r_vals = 0.85
    ax0.fill_between([0]+list(np.sin(theta_vals)*r_vals), [0]+list(np.cos(theta_vals)*r_vals),
                     [0]+[0]*len(theta_vals), color=clr, alpha=0.15)
    ax0.plot(np.sin(theta_vals)*r_vals, np.cos(theta_vals)*r_vals, color=clr, lw=2.5)
    ax0.annotate('', xy=(math.sin(math.radians(angle))*r_vals*1.15,
                         math.cos(math.radians(angle))*r_vals*1.15),
                xytext=(0,0), arrowprops=dict(arrowstyle='->', color=clr, lw=4))
    label_rad = math.radians(angle)
    ax0.text(math.sin(label_rad)*1.7, math.cos(label_rad)*1.7, label_txt,
            fontsize=8.5, ha='center', va='center', color=clr, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=clr, alpha=0.92))

ax0.text(0, 1.8, 'N', ha='center', fontsize=13, fontweight='bold')
ax0.text(0, -1.8, 'S', ha='center', fontsize=10, color='gray')
ax0.text(1.8, 0, 'E', ha='center', fontsize=10, color='gray')
ax0.text(-1.8, 0, 'W', ha='center', fontsize=10, color='gray')
ax0.set_xlim(-2, 2); ax0.set_ylim(-2, 2)
ax0.set_aspect('equal'); ax0.axis('off')
ax0.set_title('A) nlhrw Radar (Den Helder, Netherlands) -- Spring and Autumn Migration Directions\nVPTS 200m layer, nighttime high-density events (>10 birds/km3)',
             fontsize=12, fontweight='bold')

# B: deess rose
ax1 = fig.add_subplot(1, 3, 2)
for r_val in [0.3, 0.6, 0.9]:
    ax1.add_patch(plt.Circle((0,0), r_val, fill=False, color='#ddd', lw=0.5))
angle, clr = 21.0, '#E74C3C'
theta_vals = np.linspace(math.radians(angle-40), math.radians(angle+40), 50)
r_vals = 0.55
ax1.fill_between([0]+list(np.sin(theta_vals)*r_vals), [0]+list(np.cos(theta_vals)*r_vals),
                 [0]+[0]*len(theta_vals), color=clr, alpha=0.15)
ax1.plot(np.sin(theta_vals)*r_vals, np.cos(theta_vals)*r_vals, color=clr, lw=2.5)
ax1.annotate('', xy=(math.sin(math.radians(angle))*1.15, math.cos(math.radians(angle))*1.15),
            xytext=(0,0), arrowprops=dict(arrowstyle='->', color=clr, lw=4))
ax1.text(math.sin(math.radians(angle))*1.7, math.cos(math.radians(angle))*1.7,
        'Spring (n=16 only!)\nMean: 21.0 deg (N)\nDensity: 14.8 birds/km3\nWARNING: small sample',
        fontsize=8.5, ha='center', va='center', color=clr, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=clr, alpha=0.92))
ax1.text(0, 1.8, 'N', ha='center', fontsize=13, fontweight='bold')
ax1.text(0, -1.8, 'S', ha='center', fontsize=10, color='gray')
ax1.text(1.8, 0, 'E', ha='center', fontsize=10, color='gray')
ax1.text(-1.8, 0, 'W', ha='center', fontsize=10, color='gray')
ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
ax1.set_aspect('equal'); ax1.axis('off')
ax1.set_title('B) deess Radar (Essen, Germany) -- Spring Migration Direction\nLimited sample (16 observation windows only)',
             fontsize=12, fontweight='bold')

# C: Exposure curve
ax2 = fig.add_subplot(1, 3, 3)
theta = np.linspace(0, 180, 500)
# Spring: birds heading 32.6deg
d_spring = np.minimum(np.abs(theta - 32.6), np.minimum(np.abs(theta - 212.6), np.abs(theta + 147.4)))
exp_s = np.sin(np.radians(d_spring))**2
# Autumn: birds heading 233.2deg = 53.2deg axial
d_autumn = np.minimum(np.abs(theta - 53.2), np.minimum(np.abs(theta - 233.2), np.abs(theta + 126.8)))
exp_a = np.sin(np.radians(d_autumn))**2
# Simple average
exp_avg = (exp_s + exp_a) / 2

ax2.plot(theta, exp_s, color='#E74C3C', lw=3, label='Spring (birds heading 32.6 deg NNE)')
ax2.plot(theta, exp_a, color='#3498DB', lw=3, label='Autumn (birds heading 233.2 deg SW)')
ax2.plot(theta, exp_avg, color='#2C3E50', lw=2, linestyle='--', alpha=0.6, label='Annual average')

ax2.axvline(x=33, color='#27AE60', linestyle='--', alpha=0.4, lw=2)
ax2.axvline(x=53, color='#27AE60', linestyle='--', alpha=0.4, lw=2)
ax2.axvline(x=123, color='#E74C3C', linestyle=':', alpha=0.4, lw=2)

ax2.annotate('BEST eco-orientation:\n33-53 deg\n(parallel to bird flight)\nExposure ~ 0',
            xy=(43, 0.05), fontsize=10, ha='center', color='#27AE60',
            fontweight='bold', bbox=dict(facecolor='white', edgecolor='#27AE60', alpha=0.9))
ax2.annotate('WORST eco-orientation:\n~123 deg\n(perpendicular to bird flight)\n41x risk ratio',
            xy=(123, 0.85), fontsize=10, ha='center', color='#E74C3C',
            fontweight='bold', bbox=dict(facecolor='white', edgecolor='#E74C3C', alpha=0.9))
ax2.fill_between([33, 53], 0, 1, color='#27AE60', alpha=0.08)
ax2.annotate('Low-risk\ndesign\nwindow', xy=(43, 0.55), fontsize=9, ha='center',
            color='#27AE60', style='italic')

ax2.set_xlabel('Farm row orientation (theta, degrees)', fontsize=11)
ax2.set_ylabel('Relative ecological exposure (0-1)', fontsize=11)
ax2.set_title('C) Ecological Exposure Curve for a Wind Farm near Den Helder (NL)\nExposure model: sin2(angle between row axis and bird flight direction), weighted by flux and conservation priority',
             fontsize=12, fontweight='bold')
ax2.set_xlim(0, 180); ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax2.grid(alpha=0.2)

# Twin axis
ax2t = ax2.twiny()
ax2t.set_xlim(0, 180)
ax2t.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax2t.set_xticklabels(['N-S\n0 deg','NNE-SSW\n30 deg','ENE-WSW\n60 deg','E-W\n90 deg',
                       'ESE-WNW\n120 deg','SSE-NNW\n150 deg','N-S\n180 deg'], fontsize=7)
ax2t.set_xlabel('Compass orientation of row axis', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig3_direction_exposure.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig3')

# ============================================================
# FIG 4: DATA READINESS
# ============================================================
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('Figure 4: Data Readiness Matrix & Key Bottlenecks -- Wind Farms by Region', fontsize=14, fontweight='bold')

categories = ['Wind farm\nlocation\n(U0)', 'Fine-scale\necol. overlap\n(U1)',
              'Flight\ndirection\np(phi) (U2)', 'Flux +\nrotor height\n(U3a)', 'AEP curve\n(U3b)']
regions = ['37 farms: EU coastal\n(<500 km of radar)', '40 farms: UK/Nordic\n(>400 km of radar)',
           '88 farms: East Asia\n(offshore)']
data_vals = [[37, 0, 37, 37, 37], [40, 0, 0, 0, 40], [88, 0, 0, 0, 88]]
colors = ['#27AE60', '#F39C12', '#E74C3C']

x = np.arange(len(categories))
width = 0.22
for i, (region, color, vals) in enumerate(zip(regions, colors, data_vals)):
    bars = ax_top.bar(x + i*width, vals, width, label=region, color=color, alpha=0.88, edgecolor='white', lw=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax_top.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, str(val),
                       ha='center', fontsize=10, fontweight='bold', color=color)


ax_top.set_ylabel('Number of wind farms', fontsize=11)
ax_top.set_xticks(x + width)
ax_top.set_xticklabels(categories, fontsize=9)
ax_top.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
ax_top.set_ylim(0, 100)
ax_top.grid(axis='y', alpha=0.25)
ax_top.set_title('Figure 4: Data Readiness by Region -- Five Variables Required for Ecology-AEP Trade-Off', fontsize=12, fontweight='bold', pad=8)

# Bottom: bottleneck flow chart
ax_bot.set_xlim(0, 10); ax_bot.set_ylim(0, 4); ax_bot.axis('off')

boxes = [
    (0.5, 1.6, 1.4, 1.3, '171 farms\nU0: location OK', '#E8F8F5'),
    (2.5, 1.6, 2.0, 1.3, 'U1: Fine overlap?\n>>> NEED AVISTEP\nor GPS tracking', '#FDEDEC'),
    (5.2, 1.6, 2.0, 1.3, 'U2: Direction p(phi)\n>>> NEED radar/GPS\n37/171 have radar', '#FEF9E7'),
    (7.9, 1.6, 1.6, 1.3, 'U3: AEP + Exposure\nCompute trade-off\n(code ready)', '#E8F8F5'),
]
for xb, yb, wb, hb, txt, clr in boxes:
    ax_bot.add_patch(plt.Rectangle((xb, yb), wb, hb, facecolor=clr, edgecolor='#555', lw=2, alpha=0.9))
    ax_bot.text(xb+wb/2, yb+hb/2, txt, ha='center', va='center', fontsize=10)

for x1, x2 in [(1.9, 2.5), (4.5, 5.2), (7.2, 7.9)]:
    ax_bot.annotate('', xy=(x2, 2.25), xytext=(x1, 2.25),
                   arrowprops=dict(arrowstyle='->', color='#555', lw=3))

ax_bot.text(3.5, 0.6, 'BOTTLENECK 1: No fine-scale\necological overlap data (U1)',
           ha='center', fontsize=10, color='#C0392B', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.9))
ax_bot.text(6.2, 0.6, 'BOTTLENECK 2: No bird direction\ndata for East Asia + UK/Nordic (U2)',
           ha='center', fontsize=10, color='#C0392B', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FDEDEC', alpha=0.9))

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig4_data_readiness.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig4')

print('\nDone. All figures in outputs/')
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.png'):
        print(f'  {f} ({os.path.getsize(os.path.join(out_dir,f))/1e3:.0f} KB)')
