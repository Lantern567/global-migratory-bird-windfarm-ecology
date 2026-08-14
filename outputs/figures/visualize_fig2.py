import os, sys, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import numpy as np

for fp in fm.findSystemFonts():
    if 'simhei' in fp.lower():
        fm.fontManager.addfont(fp)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

import cartopy.crs as ccrs
import cartopy.feature as cfeature

out_dir = r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\outputs'

farms = pd.read_csv(r'D:\1风力发电实习\wind-direction-to-electricity-transition-main\offshore-task0-HuTingxian\output\task0\farms_master.csv')
flyways = gpd.read_file(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\raw\caff_flyways\shapefile\Major_Flyways.shp', engine='fiona')
global_wind = pd.read_csv(r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\data\processed\global_wind_farms.csv')
global_wind = global_wind.dropna(subset=['latitude', 'longitude'])

fig = plt.figure(figsize=(22, 13))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#EDE8DC', edgecolor='none', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#C8DDE8', edgecolor='none', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#999', zorder=1)

# Flyways: red outlines + light fill
flyway_colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
for idx, row in flyways.iterrows():
    name = row['NAME']
    color = flyway_colors[idx % len(flyway_colors)]
    # Outline
    # Very faint fill + bold outline — shows extent but emphasizes it"s conceptual
    ax.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                      facecolor=color, edgecolor=color, linewidth=2.5, alpha=0.08, zorder=2)
    ax.add_geometries([row.geometry.boundary], crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor=color, linewidth=1.8, alpha=0.75, zorder=2)
    # Label
    c = row.geometry.centroid
    ax.annotate(name, xy=(c.x, c.y), xycoords=ccrs.PlateCarree(),
                fontsize=9, color=color, ha='center', va='center', fontweight='bold',
                alpha=0.9, path_effects=[pe.withStroke(linewidth=3, foreground='white')])

# Global wind - sample dots
sample = global_wind.sample(min(25000, len(global_wind)), random_state=42)
ax.scatter(sample['longitude'], sample['latitude'], transform=ccrs.PlateCarree(),
           s=0.8, c='#2C3E50', alpha=0.12, zorder=3)

# Our 171 farms - prominent red
ax.scatter(farms['centroid_lon'], farms['centroid_lat'], transform=ccrs.PlateCarree(),
           s=40, c='#E74C3C', edgecolors='white', linewidth=0.5, zorder=5)

# Country labels for our farms
for label, lon, lat in [
    ('China\n(66 farms)', 120, 34), ('United Kingdom\n(31 farms)', -2, 56),
    ('Vietnam\n(13 farms)', 108, 13), ('Denmark\n(12 farms)', 10, 56.5),
    ('Netherlands\n(11 farms)', 5, 53.5), ('Sweden\n(8 farms)', 18, 60),
    ('Germany\n(6 farms)', 8, 54), ('USA\n(5 farms)', -72, 40),
    ('France\n(5 farms)', -1, 48), ('Taiwan\n(4 farms)', 121, 25),
    ('Japan\n(3 farms)', 140, 42), ('South Korea\n(2 farms)', 127, 37),
]:
    ax.annotate(label, xy=(lon, lat), xycoords=ccrs.PlateCarree(),
                fontsize=7, ha='center', va='center', color='#C0392B', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

ax.set_title('Figure 2: Global Wind Farms and Arctic Bird Migration Flyways (Conceptual Overlap Only)\nCAFF flyways (Arctic Biodiversity Assessment 2013) -- each covers 27-105 million km2',
             fontsize=15, fontweight='bold', pad=12)

fig.savefig(os.path.join(out_dir, 'fig2_global_flyways_wind.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig2_global_flyways_wind.png')
