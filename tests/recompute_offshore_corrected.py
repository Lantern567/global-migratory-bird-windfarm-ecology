# -*- coding: utf-8 -*-
"""
重新推导海上侧 AEP-生态交换率（修正 B-7 及其更深层问题）。

修正内容：
1. 用风场真实坐标（farms_master.csv 的 centroid_lat/lon）重新匹配 Bauer 格网，
   替换旧 bauer_farm_directions.csv 中"把英国风场匹配到 300+ km 外比利时格网"的错误。
2. 采用与陆上一致的 200 km 匹配截断（KDTree 等距投影）。
3. 实测方向口径 = VPTS 雷达（29 场，0-200m 转子层校正，距站 400km 截断）+ Bauer 格网 <200 km（非 VPTS 场）。

实测方向场数 = 29 + 26 = 55 场（飞高层修正 + 400km 截断后，原 37+18=55）。
"""
import sys, os, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\src')

import pandas as pd, numpy as np
from scipy.spatial import cKDTree
from bird_wind_ecology.ecology_analysis import parallel_corridor_proxy_curve
from bird_wind_ecology.models import BirdDirectionSignature, AEPOrientationPoint, EcologyCurvePoint
from bird_wind_ecology.integration import optimize_under_aep_budget

P = r'D:\1风力发电实习\global-migratory-bird-windfarm-ecology-main\our_work\data\processed'
AEP_CSV = r'D:\1风力发电实习\offshore-task3\output\task3_s1_optimal_orientation.csv'
FARMS_CSV = r'D:\1风力发电实习\wind-direction-to-electricity-transition-main\offshore-task0-HuTingxian\output\task0\farms_master.csv'

farms = pd.read_csv(FARMS_CSV)
groups = pd.read_csv(os.path.join(P, 'farm_groups_updated.csv'))
radar_sigs = pd.read_csv(os.path.join(P, 'farm_direction_signatures_corrected.csv'))  # 飞高层修正：0-200m转子层 + 经验rhf
bauer_old = pd.read_csv(os.path.join(P, 'bauer_farm_directions.csv'))  # 仅取 conc 用
grid = pd.read_csv(os.path.join(P, 'bauer_grid_cell_directions.csv')).dropna(subset=['spring_dir', 'autumn_dir'])
birbase = pd.read_csv(os.path.join(P, 'birbase_conservation_weights.csv'))

radar_ids = set(radar_sigs['farm_id'].unique())
w_radar = birbase[birbase['order'].isin(['Charadriiformes', 'Anseriformes', 'Procellariiformes'])]['weight'].mean()
w_era5 = birbase[birbase['migratory'] == True]['weight'].mean()

# ---- 用真实坐标重新匹配 Bauer 格网（200 km 截断，等距投影）----
KM = 111.0
COS = np.cos(np.radians(50.0))
gcoords = np.column_stack([grid['lat'].values * KM, grid['lon'].values * KM * COS])
fcoords = np.column_stack([farms['centroid_lat'].values * KM, farms['centroid_lon'].values * KM * COS])
tree = cKDTree(gcoords)
dist, idx = tree.query(fcoords, k=1, distance_upper_bound=200.0)
valid = np.isfinite(dist)
gm = grid.iloc[np.where(valid, idx, 0)].reset_index(drop=True)

# 每场 conc（Bauer 场沿用旧文件里的 bauer_conc；其余用 0.15 下限）
conc_map = dict(zip(bauer_old['farm_id'], bauer_old['bauer_conc']))
def conc_for(fid):
    c = conc_map.get(fid, 0.15)
    return max(c, 0.15)

# ---- 构建签名 ----
all_sigs = []
# 1) VPTS 雷达（直接观测，35 场，0-200m 转子层校正）
for _, s in radar_sigs.iterrows():
    all_sigs.append(dict(s))

# 2) Bauer 格网校正（非 VPTS 且 <200 km，22 场）
non_vpts_valid = (~farms['farm_id'].isin(radar_ids)) & valid
for i, r in farms[non_vpts_valid].iterrows():
    fid = r['farm_id']
    c = r['country']
    sp = gm.iloc[i]['spring_dir'] if i < len(gm) else np.nan
    au = gm.iloc[i]['autumn_dir'] if i < len(gm) else np.nan
    conc = conc_for(fid)
    for season, d in [('spring', sp), ('autumn', au)]:
        for spread in [-20, -10, 0, 10, 20]:
            all_sigs.append({
                'farm_id': fid, 'receptor_id': f'bauer_grid_{c}',
                'season': season, 'direction_deg': (d + spread) % 360,
                'concentration': conc, 'flux': 50.0,
                'rotor_height_fraction': 0.5, 'evidence_level': 'radar',
                'conservation_weight': w_radar, 'n_observations': 5,
                'source': f'Bauer2026_grid_{c}',
            })

# 3) ERA5 代理（其余 116 场）
era5_mask = ~farms['farm_id'].isin(radar_ids) & (~non_vpts_valid)
for _, r in farms[era5_mask].iterrows():
    fid = r['farm_id']
    c = r['country']
    base_d = 60.0 if c in ['China', 'Vietnam'] else 45.0
    for season, bd in [('spring', base_d), ('autumn', (base_d + 180) % 360)]:
        for spread in [-30, -15, 0, 15, 30]:
            all_sigs.append({
                'farm_id': fid, 'receptor_id': f'era5_{c}',
                'season': season, 'direction_deg': (bd + spread) % 360,
                'concentration': 0.20, 'flux': 50.0,
                'rotor_height_fraction': 0.5, 'evidence_level': 'coarse-flyway',
                'conservation_weight': w_era5, 'n_observations': 5,
                'source': f'ERA5_{c}',
            })

sig_df = pd.DataFrame(all_sigs)
radar_fids = set(sig_df[sig_df['evidence_level'] == 'radar']['farm_id'].unique())
print(f'签名总数: {len(sig_df)}')
print(f'实测方向(radar)场数: {len(radar_fids)}')
print(f'ERA5 代理场数: {len(set(sig_df[~sig_df.farm_id.isin(radar_fids)].farm_id))}')

# ---- 暴露曲线 ----
theta_vals = list(range(0, 181, 10))
all_curves = []
for fid in sorted(sig_df['farm_id'].unique()):
    fs = sig_df[sig_df['farm_id'] == fid]
    objs = []
    for _, s in fs.iterrows():
        try:
            objs.append(BirdDirectionSignature(farm_id=str(fid), receptor_id=s['receptor_id'],
                season=s['season'], direction_deg=s['direction_deg'], concentration=s['concentration'],
                flux=s['flux'], rotor_height_fraction=s['rotor_height_fraction'],
                evidence_level=s['evidence_level'], conservation_weight=s['conservation_weight'],
                n_observations=int(s['n_observations']), source=s['source']))
        except Exception:
            pass
    if not objs:
        continue
    curve = parallel_corridor_proxy_curve(str(fid), theta_vals, objs)
    for p in curve:
        all_curves.append({'farm_id': fid, 'theta_deg': p.theta_deg, 'risk_score': round(p.risk_score, 4)})

curves = pd.DataFrame(all_curves)

# ---- 交换率 ----
s1 = pd.read_csv(AEP_CSV)
td = []
for fid in sorted(sig_df['farm_id'].unique()):
    ad = s1[s1['farm_id'] == fid]
    ed = curves[curves['farm_id'] == fid]
    if len(ed) == 0 or len(ad) == 0:
        continue
    ap = [AEPOrientationPoint(str(fid), float(r['angle_deg']), float(r['expected_AEP_kWh']) / 1e6)
          for _, r in ad.iterrows()]
    ep = [EcologyCurvePoint(str(fid), float(r['theta_deg']), float(r['risk_score']))
          for _, r in ed.iterrows()]
    for budget in [0.005, 0.01, 0.02, 0.05]:
        try:
            res = optimize_under_aep_budget(ap, ep, budget)
            td.append({'farm_id': fid, 'budget': budget,
                       'theta_econ': res.theta_econ_deg, 'theta_eco': res.theta_eco_deg,
                       'aep_cost_pct': round(res.aep_cost_gwh / res.aep_econ_gwh * 100, 4),
                       'risk_reduction': round(res.relative_risk_reduction, 4)})
        except Exception:
            pass

tr = pd.DataFrame(td)
tr['risk_reduction_pct'] = (tr['risk_reduction'] * 100).round(2)
tr = tr.merge(farms[['farm_id', 'country', 'capacity_kW']], on='farm_id')
tr.to_csv(os.path.join(P, 'tradeoff_offshore_55farms.csv'), index=False, encoding='utf-8-sig')

# ---- 汇总 ----
b01 = tr[tr['budget'] == 0.01]
measured = b01[b01['farm_id'].isin(radar_fids)]
era5 = b01[~b01['farm_id'].isin(radar_fids)]
print('\n=== 海上实测方向 55 场（VPTS 29 + Bauer<200km 26）1% 预算 ===')
print(f'场数: {len(measured)}')
print(f'AEP 代价 mean: {measured.aep_cost_pct.mean():.4f}%  median: {measured.aep_cost_pct.median():.4f}%')
print(f'RR mean: {measured.risk_reduction_pct.mean():.1f}%  median: {measured.risk_reduction_pct.median():.1f}%')
print(f'RR>90%: {(measured.risk_reduction_pct>90).sum()}/{len(measured)} = {(measured.risk_reduction_pct>90).mean()*100:.1f}%')
print(f'RR>50%: {(measured.risk_reduction_pct>50).sum()}/{len(measured)}')
print(f'RR 60-99%: {((measured.risk_reduction_pct>=60)&(measured.risk_reduction_pct<=99)).sum()}/{len(measured)}')
print(f'theta_eco 中位: {measured.theta_eco.median():.0f}°')

print(f'\nERA5 代理 {len(era5)} 场（不作为实测，仅供对比）:')
print(f'  RR mean: {era5.risk_reduction_pct.mean():.1f}%  median: {era5.risk_reduction_pct.median():.1f}%')
print('Done')