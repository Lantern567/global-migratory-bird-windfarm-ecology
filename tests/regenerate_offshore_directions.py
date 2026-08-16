# -*- coding: utf-8 -*-
"""
重生成海上方向源表（修正飞高层后）：VPTS(0-200m转子层) + Bauer<200km 两源。

产出：data/processed/offshore_farm_directions_55.csv
  字段：farm_id,country,centroid_lat,centroid_lon,source,spring_dir,autumn_dir,dist_km
  source ∈ {VPTS, Bauer_grid}；仅含"实测方向"场（VPTS 29 + Bauer 26 = 55）。
"""
import os, io, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
P = os.path.join(REPO, 'our_work', 'data', 'processed')
FARMS = r'D:\1风力发电实习\wind-direction-to-electricity-transition-main\offshore-task0-HuTingxian\output\task0\farms_master.csv'

farms = pd.read_csv(FARMS)
farms.columns = [c.strip('\ufeff') for c in farms.columns]
vpts = pd.read_csv(os.path.join(P, 'farm_direction_signatures_corrected.csv'))
grid = pd.read_csv(os.path.join(P, 'bauer_grid_cell_directions.csv')).dropna(subset=['spring_dir', 'autumn_dir'])

# VPTS 场 -> 春/秋方向
vpts_map = {}
for fid, g in vpts.groupby('farm_id'):
    sp = g[g['season'] == 'spring']['direction_deg'].iloc[0]
    au = g[g['season'] == 'autumn']['direction_deg'].iloc[0]
    vpts_map[fid] = (sp, au)

vpts_ids = set(vpts_map.keys())

# Bauer 格网匹配（200km 截断，等距投影）
KM = 111.0
COS = np.cos(np.radians(50.0))
gc = np.column_stack([grid['lat'].values * KM, grid['lon'].values * KM * COS])
fc = np.column_stack([farms['centroid_lat'].values * KM, farms['centroid_lon'].values * KM * COS])
tree = cKDTree(gc)
dist, idx = tree.query(fc, k=1, distance_upper_bound=200.0)
valid = np.isfinite(dist)
gm = grid.iloc[np.where(valid, idx, 0)].reset_index(drop=True)

rows = []
for i, r in farms.iterrows():
    fid = r['farm_id']
    if fid in vpts_ids:
        sp, au = vpts_map[fid]
        rows.append({'farm_id': fid, 'country': r['country'],
                     'centroid_lat': r['centroid_lat'], 'centroid_lon': r['centroid_lon'],
                     'source': 'VPTS', 'spring_dir': sp, 'autumn_dir': au, 'dist_km': np.nan})
    elif valid[i]:
        rows.append({'farm_id': fid, 'country': r['country'],
                     'centroid_lat': r['centroid_lat'], 'centroid_lon': r['centroid_lon'],
                     'source': 'Bauer_grid',
                     'spring_dir': gm.iloc[i]['spring_dir'],
                     'autumn_dir': gm.iloc[i]['autumn_dir'],
                     'dist_km': round(dist[i], 1)})

out = pd.DataFrame(rows)
out_path = os.path.join(P, 'offshore_farm_directions_55.csv')
out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'VPTS 场数: {len(vpts_ids)}')
print(f'Bauer 场数: {(out.source == "Bauer_grid").sum()}')
print(f'总计实测方向场数: {len(out)}')
print(f'已保存: {out_path}')