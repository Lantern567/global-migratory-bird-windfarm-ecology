# -*- coding: utf-8 -*-
"""
sensitivity_compute.py —— 补充材料 S7 敏感性分析统一计算。

实事求是：每一项敏感性分析都从当前数据重算，打印来源 CSV 与数值，
并写一份 JSON 汇总（供建图与建文档共用），避免图/文数字漂移。

口径与 onshore_tradeoff.py / recompute_4_old_numbers.py 完全一致：
  - 朝向网格 ORIENT = arange(0,180,10)（18 个朝向，180° 周期）
  - 陆上几何暴露 E(θ) 在 18 个朝向上求值（浓度加权 sin²，缺失时等权 0.5/0.5）
  - RR = (E[θ_econ] - E[θ_eco]) / E[θ_econ] * 100

输出：
  data/processed/sensitivity_summary.json
  data/processed/sensitivity_wake_penalty_perfarm.csv   (S7.1)
  data/processed/sensitivity_era5_vs_bird.csv           (S7.3)
  data/processed/sensitivity_seasonal_weight_perfarm.csv(S7.5)
  data/processed/sensitivity_dbs_cluster.csv            (S7.6)
  data/processed/sensitivity_match_radius.csv           (S7.7)
  data/processed/sensitivity_stratification.csv         (S7.11)
"""
import os, sys, io, json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, '..', 'data', 'processed')
RAW = os.path.join(BASE, '..', 'data', 'raw')
OFF_AEP = r'D:\1风力发电实习\offshore-task3\output\task3_s1_optimal_orientation.csv'

ORIENT = np.arange(0, 180, 10)          # 18 朝向（与 onshore_tradeoff.py 一致）
KM_PER_DEG_LAT = 111.0
LAT_REF = 50.0
COS_LAT_REF = np.cos(np.radians(LAT_REF))

OUT = {}


def circ_dist(a, b):
    """循环角距 on [0,360)，范围 [0,180]（与 recompute_4_old_numbers.py 一致）。"""
    return abs((a - b + 180) % 360 - 180)


def circ_median(angles):
    """循环中位（最小化循环角距和），与 recompute_4_old_numbers.py 一致。"""
    angles = np.asarray(angles, dtype=float) % 360.0
    best, best_cost = None, np.inf
    for cand in angles:
        cost = np.sum(circ_dist(cand, angles))
        if cost < best_cost:
            best_cost, best = cost, cand
    return best


def on_Evec(spring_dir, autumn_dir, cs, ca):
    """陆上几何暴露 E(θ)（18 朝向，浓度加权 sin²），与 onshore_tradeoff.py 完全一致。"""
    try:
        cs = float(cs); ca = float(ca)
    except (TypeError, ValueError):
        cs = ca = np.nan
    if np.isnan(cs) or np.isnan(ca) or (cs + ca) <= 0:
        ws = wa = 0.5
    else:
        ws = cs / (cs + ca); wa = ca / (cs + ca)
    E = np.zeros(len(ORIENT))
    for i, th in enumerate(ORIENT):
        E[i] = ws * np.sin(np.radians(th - spring_dir)) ** 2 + wa * np.sin(np.radians(th - autumn_dir)) ** 2
    return E


def tradeoff_at_budget(aep_kwh, E, budget):
    """单预算权衡（18 朝向）。E 与 aep_kwh 长度均 18。返回 dict 或 None。"""
    iecon = int(np.argmax(aep_kwh)); amax = aep_kwh[iecon]
    allowed = np.where(aep_kwh >= (1 - budget) * amax)[0]
    if len(allowed) == 0:
        return None
    ieco = allowed[np.argmin(E[allowed])]
    Ee, Eeco = E[iecon], E[ieco]
    rr = (Ee - Eeco) / max(Ee, 1e-12) * 100
    aep_cost = (amax - aep_kwh[ieco]) / amax * 100
    return dict(theta_econ=ORIENT[iecon], theta_eco=ORIENT[ieco], rr=rr, aep_cost=aep_cost)


# =====================================================================
# 加载基础数据
# =====================================================================
on = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
on1 = on[on.budget == 0.01].copy()
onaep = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
aep_cols = [f'aep_{int(t):03d}' for t in ORIENT]
on1 = on1.merge(onaep[['farm_id'] + aep_cols], on='farm_id', how='inner')
od = pd.read_csv(os.path.join(PROC, 'offshore_farm_directions_55.csv'), encoding='utf-8-sig')
od.columns = [c.strip().lstrip('\ufeff') for c in od.columns]
grid = pd.read_csv(os.path.join(PROC, 'bauer_grid_cell_directions.csv'), encoding='utf-8-sig')

print('基础数据：陆上 on1=%d, AEP 匹配=%d, 海上 od=%d' % (len(on1), len(on1), len(od)))

# =====================================================================
# S7.1 尾流/朝向亏损放大 1.5/2/3×  （逐场 + 中位，同 recompute_4_old_numbers.py）
# =====================================================================
print('\n' + '=' * 66)
print('S7.1 尾流/朝向亏损放大敏感性（陆上 1% 预算 RR 中位）')
print('=' * 66)
scales = [1.0, 1.5, 2.0, 3.0]
rows = []
for _, r in on1.iterrows():
    aep = np.array([r[f'aep_{int(t):03d}'] for t in ORIENT], dtype=float)
    if aep.max() <= 0:
        continue
    E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
    amax = aep.max()
    rec = {'farm_id': r.farm_id}
    for k in scales:
        aep_scaled = amax - k * (amax - aep)
        t = tradeoff_at_budget(aep_scaled, E, 0.01)
        rec[f'rr_{k}'] = t['rr'] if t else np.nan
    rows.append(rec)
wp = pd.DataFrame(rows)
wp.to_csv(os.path.join(PROC, 'sensitivity_wake_penalty_perfarm.csv'), index=False)
S71 = {str(k): round(float(wp[f'rr_{k}'].median()), 1) for k in scales}
OUT['S71_wake_penalty_rr_med'] = S71
print('  RR 中位：', S71, ' n=%d' % len(wp))

# =====================================================================
# S7.2 衰减常数 α 0.075 → 0.05 （集中度加权口径，Flag A 已重算）
# =====================================================================
print('\n' + '=' * 66)
print('S7.2 衰减常数 α 敏感性（集中度加权）')
print('=' * 66)
a50 = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results_alpha050_conc.csv'), encoding='utf-8-sig')
a50_1 = a50[a50.budget == 0.01].set_index('farm_id')
a075_rr = on[on.budget == 0.01].set_index('farm_id')['risk_reduction']
common = a50_1.index.intersection(a075_rr.index)
rr75 = a075_rr[common].median()
rr50 = a50_1.loc[common, 'risk_reduction'].median()
OUT['S72_alpha_075_rr_med'] = round(float(rr75), 1)
OUT['S72_alpha_050_rr_med'] = round(float(rr50), 1)
OUT['S72_alpha_drop_pp'] = round(float(rr75 - rr50), 1)
print('  α=0.075 RR中位=%.1f%% ; α=0.05 RR中位=%.1f%% ; 降幅=%.1fpp (n=%d)'
      % (rr75, rr50, rr75 - rr50, len(common)))

# =====================================================================
# S7.3 ERA5 气象风向 vs 实测候鸟方向（Flag B：当前数据可复现口径）
# =====================================================================
print('\n' + '=' * 66)
print('S7.3 ERA5 气象风向 vs 实测候鸟方向')
print('=' * 66)
era = pd.read_csv(os.path.join(PROC, 'era5_wind_direction_per_farm.csv'), encoding='utf-8-sig')
era_cols = ['farm_id', 'era5_wind_dir']
S73 = {}
rows3 = []
for name, src in [('VPTS', 'VPTS'), ('Bauer', 'Bauer_grid')]:
    g = od[od.source == src].merge(era[era_cols], on='farm_id', how='inner')
    for season, sd in [('spring', 'spring_dir'), ('autumn', 'autumn_dir')]:
        d360 = np.array([circ_dist(g.era5_wind_dir.values[i], g[sd].values[i]) for i in range(len(g))])
        key = f'S73_{name}_{season}'
        S73[key + '_mean360'] = round(float(d360.mean()), 1)
        S73[key + '_within30'] = round(float((d360 < 30).mean() * 100), 1)
        print('  %-5s %-6s: 循环差 均值=%.1f° 中位=%.1f° within30=%.1f%% (n=%d)'
              % (name, season, d360.mean(), np.median(d360), (d360 < 30).mean() * 100, len(g)))
        for i in range(len(g)):
            rows3.append(dict(source=name, season=season, farm_id=g.farm_id.values[i],
                              era5_wind_dir=g.era5_wind_dir.values[i], bird_dir=g[sd].values[i],
                              circ360=float(d360[i])))
OUT['S73_era5'] = S73
pd.DataFrame(rows3).to_csv(os.path.join(PROC, 'sensitivity_era5_vs_bird.csv'), index=False)

# =====================================================================
# S7.4 海上方向口径 VPTS vs Bauer
# =====================================================================
print('\n' + '=' * 66)
print('S7.4 海上方向口径 VPTS vs Bauer')
print('=' * 66)
vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']
S74 = {}
for season, sd in [('spring', 'spring_dir'), ('autumn', 'autumn_dir')]:
    mv = circ_median(vpts[sd].values); mb = circ_median(bauer[sd].values)
    S74[f'S74_{season}_vpts_med'] = round(float(mv), 1)
    S74[f'S74_{season}_bauer_med'] = round(float(mb), 1)
    S74[f'S74_{season}_divergence'] = round(float(circ_dist(mv, mb)), 1)
    print('  %s: VPTS中位=%.1f° Bauer中位=%.1f° 循环差=%.1f°' % (season, mv, mb, circ_dist(mv, mb)))
OUT['S74_offshore_convention'] = S74

# =====================================================================
# S7.5 季节加权 等权 vs 集中度加权
# =====================================================================
print('\n' + '=' * 66)
print('S7.5 季节加权 等权(0.5/0.5) vs 集中度加权')
print('=' * 66)
ew = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results_equalweight_backup.csv'), encoding='utf-8-sig')
ew1 = ew[ew.budget == 0.01].set_index('farm_id')['risk_reduction']
conc1 = on[on.budget == 0.01].set_index('farm_id')['risk_reduction']
cm = ew1.index.intersection(conc1.index)
diff = (conc1[cm] - ew1[cm])
S75 = dict(rr_med_conc=round(float(conc1[cm].median()), 1),
           rr_med_equal=round(float(ew1[cm].median()), 1),
           diff_med_pp=round(float(diff.median()), 2),
           diff_mean_pp=round(float(diff.mean()), 2),
           n=int(len(cm)))
OUT['S75_seasonal_weight'] = S75
pd.DataFrame(dict(farm_id=cm, rr_conc=conc1[cm].values, rr_equal=ew1[cm].values,
                  diff_pp=diff.values)).to_csv(
    os.path.join(PROC, 'sensitivity_seasonal_weight_perfarm.csv'), index=False)
print('  集中度加权中位=%.1f%% ; 等权中位=%.1f%% ; 差(conc-equal) 中位=%.2fpp 均值=%.2fpp (n=%d)'
      % (conc1[cm].median(), ew1[cm].median(), diff.median(), diff.mean(), len(cm)))

# =====================================================================
# S7.6 DBSCAN eps × min_samples 扫描
# =====================================================================
print('\n' + '=' * 66)
print('S7.6 DBSCAN 聚类参数敏感性')
print('=' * 66)
from sklearn.cluster import DBSCAN
turb = pd.read_csv(os.path.join(RAW, 'osm_turbines_parsed.csv'), encoding='utf-8-sig')
# 与原管线一致的投影：lat*111, lon*111（无 cos 纬度修正）。此投影下
# eps=3km/min_samples=3 复现 5,954 簇，与 osm_farm_pca_orientations.csv 及正文一致。
xy = np.column_stack([
    turb['lat'].values * KM_PER_DEG_LAT,
    turb['lon'].values * KM_PER_DEG_LAT,   # 无 cos 修正（匹配原聚类管线）
])
eps_km_list = [2, 3, 4, 5]
ms_list = [2, 3, 4, 5]
dbs_rows = []
for eps in eps_km_list:
    for ms in ms_list:
        cl = DBSCAN(eps=eps, min_samples=ms).fit(xy)
        labs = cl.labels_
        n_cl = len(set(labs)) - (1 if -1 in labs else 0)
        n_in = int((labs != -1).sum())
        dbs_rows.append(dict(eps_km=eps, min_samples=ms, n_clusters=n_cl,
                             n_turbines_in=n_in, frac_in=round(n_in / len(turb), 4)))
        print('  eps=%dkm min_samples=%d -> 聚类场数=%d 覆盖风机=%d (%.1f%%)'
              % (eps, ms, n_cl, n_in, n_in / len(turb) * 100))
dbs = pd.DataFrame(dbs_rows)
dbs.to_csv(os.path.join(PROC, 'sensitivity_dbs_cluster.csv'), index=False)
OUT['S76_dbscan'] = dbs_rows

# =====================================================================
# S7.7 匹配半径扫描（Bauer 格网 200km → 50/100/150/200/300）
# =====================================================================
print('\n' + '=' * 66)
print('S7.7 Bauer 格网匹配半径敏感性')
print('=' * 66)
from scipy.spatial import cKDTree
farms = pd.read_csv(os.path.join(PROC, 'osm_farm_pca_orientations.csv'), encoding='utf-8-sig')
gv = grid.dropna(subset=['spring_dir', 'autumn_dir']).copy()
grid_km = np.column_stack([gv['lat'].values * KM_PER_DEG_LAT,
                           gv['lon'].values * KM_PER_DEG_LAT * COS_LAT_REF])
farm_km = np.column_stack([farms['centroid_lat'].values * KM_PER_DEG_LAT,
                           farms['centroid_lon'].values * KM_PER_DEG_LAT * COS_LAT_REF])
tree = cKDTree(grid_km)
aep_idx = onaep.set_index('farm_id')
radii = [50, 100, 150, 200, 300]
mr_rows = []
for rad in radii:
    dist, idx = tree.query(farm_km, k=1, distance_upper_bound=rad)
    valid = np.isfinite(dist)
    fm = farms[valid].copy()
    gm = gv.iloc[idx[valid]]
    fm['spring_dir'] = gm['spring_dir'].values
    fm['autumn_dir'] = gm['autumn_dir'].values
    fm['spring_conc'] = gm['spring_conc'].values
    fm['autumn_conc'] = gm['autumn_conc'].values
    rrs = []
    for _, r in fm.iterrows():
        if r.farm_id not in aep_idx.index:
            continue
        ar = aep_idx.loc[r.farm_id]
        aep = np.array([ar[f'aep_{int(t):03d}'] for t in ORIENT], dtype=float)
        if aep.max() <= 0:
            continue
        E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
        t = tradeoff_at_budget(aep, E, 0.01)
        if t:
            rrs.append(t['rr'])
    mr_rows.append(dict(radius_km=rad, n_farms=int(len(fm)), n_with_aep=int(len(rrs)),
                        rr_med_1pct=round(float(np.median(rrs)), 1) if rrs else np.nan))
    print('  半径=%3dkm -> 匹配场数=%d (含AEP=%d) RR中位=%s'
          % (rad, len(fm), len(rrs), ('%.1f%%' % np.median(rrs)) if rrs else 'nan'))
mr = pd.DataFrame(mr_rows)
mr.to_csv(os.path.join(PROC, 'sensitivity_match_radius.csv'), index=False)
OUT['S77_match_radius'] = mr_rows

# =====================================================================
# S7.11 分层稳健性（规模）。国别维度不可用：OSM country 字段 76% 缺失，
# 且非空值为制造商片段（"Wi"/"En"/"Ør" 等），非 ISO 国家码，故诚实淘汰。
# =====================================================================
print('\n' + '=' * 66)
print('S7.11 分层稳健性（规模分档）')
print('=' * 66)
on1c = on1.copy()
bins = [0, 5, 10, 20, 50, 10000]
labels = ['<5', '5-9', '10-19', '20-49', '>=50']
on1c['size_bin'] = pd.cut(on1c.n_turbines, bins=bins, labels=labels, right=False)
strat_rows = []
for sb, g in on1c.groupby('size_bin', observed=True):
    strat_rows.append(dict(dim='size', key=str(sb), n=len(g),
                           rr_med=round(float(g.risk_reduction.median()), 1),
                           rr_mean=round(float(g.risk_reduction.mean()), 1)))
    print('  规模 %-7s n=%4d  RR中位=%.1f%%  RR均值=%.1f%%'
          % (sb, len(g), g.risk_reduction.median(), g.risk_reduction.mean()))
strat = pd.DataFrame(strat_rows)
strat.to_csv(os.path.join(PROC, 'sensitivity_stratification.csv'), index=False)
OUT['S711_stratification'] = strat_rows
OUT['S711_country_note'] = 'country 维度不可用：OSM country 字段 76% 缺失且非空值为制造商片段，已诚实淘汰，仅保留规模分档。'

# =====================================================================
# S7.12 PCA 主轴方差解释率分布
# =====================================================================
print('\n' + '=' * 66)
print('S7.12 PCA 主轴方差解释率分布')
print('=' * 66)
pca = pd.read_csv(os.path.join(PROC, 'osm_farm_pca_orientations.csv'), encoding='utf-8-sig')
ev = pca.explained_var_ratio.dropna()
S712 = dict(mean=round(float(ev.mean()), 4), med=round(float(ev.median()), 4),
            p25=round(float(ev.quantile(0.25)), 4), p75=round(float(ev.quantile(0.75)), 4),
            n=int(len(ev)))
OUT['S712_pca_explained_var'] = S712
print('  均值=%.4f 中位=%.4f P25=%.4f P75=%.4f (n=%d)' % (ev.mean(), ev.median(), ev.quantile(0.25), ev.quantile(0.75), len(ev)))

# =====================================================================
# S7.13 陆上 Jensen vs 海上 FLORIS AEP 朝向敏感性
# =====================================================================
print('\n' + '=' * 66)
print('S7.13 陆上 Jensen vs 海上 FLORIS AEP 朝向敏感性')
print('=' * 66)
on_a = onaep[aep_cols].values.astype(float)
on_sens = (on_a.max(axis=1) - on_a.min(axis=1)) / on_a.max(axis=1) * 100
off = pd.read_csv(OFF_AEP, encoding='utf-8-sig')
off.columns = [c.strip().lstrip('\ufeff') for c in off.columns]
off_sens = []
for fid, g in off.groupby('farm_id'):
    a = g['expected_AEP_kWh'].values.astype(float)
    if a.max() <= 0:
        continue
    off_sens.append((a.max() - a.min()) / a.max() * 100)
off_sens = np.array(off_sens)
S713 = dict(onshore_mean=round(float(on_sens.mean()), 2), onshore_med=round(float(np.median(on_sens)), 2),
            offshore_mean=round(float(off_sens.mean()), 2), offshore_med=round(float(np.median(off_sens)), 2))
OUT['S713_aep_sens'] = S713
print('  陆上 均值=%.2f%% 中位=%.2f%% ; 海上 均值=%.2f%% 中位=%.2f%%'
      % (on_sens.mean(), np.median(on_sens), off_sens.mean(), np.median(off_sens)))

# =====================================================================
# Flag C：场数链（已核实，写入 JSON 供文档诚实引用）
# =====================================================================
print('\n' + '=' * 66)
print('Flag C：陆上场数链（已逐项核实）')
print('=' * 66)
flagC = dict(
    turbines=100327,
    dbscan_clusters_raw_csv=7238,       # osm_turbines_clustered.csv 非噪声簇（farm_id 0..7237）
    dbscan_noise=3148,                   # 该 CSV 中 farm_id=-1 的孤立风机
    dbscan_eps3_ms3_no_cos_reproduced=5954,  # 文档口径 eps=3km/ms=3 无 cos 投影重跑
    pca_farms=5954,                      # PCA 后（剔除 PCA 不可靠小场）
    aep_farms=5869,                      # 剔除微/垂直轴后
    tradeoff_farms=4191,                 # Bauer 200km 匹配后
)
OUT['flagC_farm_chain'] = flagC
for k, v in flagC.items():
    print('  %-34s = %d' % (k, v))
print('  注：正文「DBSCAN(eps=3km,min_samples=3) 得 5,954 场」需两处澄清：')
print('     (1) 5,954 是 PCA 后场数，非 DBSCAN 原始簇数；')
print('     (2) osm_turbines_clustered.csv 的 7,238 簇 = eps=3km/min_samples=2（非 3）；')
print('         min_samples=3 才复现 5,954，等价于「PCA 阶段剔除 1,284 个 2 台风机的小簇」。')

# =====================================================================
# 写 JSON 汇总
# =====================================================================
out_path = os.path.join(PROC, 'sensitivity_summary.json')
with open(out_path, 'w', encoding='utf-8') as fh:
    json.dump(OUT, fh, ensure_ascii=False, indent=2)
print('\n[written]', out_path)
