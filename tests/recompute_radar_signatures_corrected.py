# -*- coding: utf-8 -*-
"""
飞高层修正：重算 VPTS 雷达方向签名（用正确的旋翼扫掠高度层）。

原管线 bug：build_upload.py 用 ROTOR_ALT=200（即 HGHT=200 → 200-400m 层，位于旋翼上方）
提取方向与通量；IEA-10MW 旋翼扫掠 30-208m，实际位于 0-200m 层（HGHT=0）。

本脚本：
  1. 方向/转子带通量 取 HGHT=0（0-200m，旋翼扫掠层）的密度加权圆均值；
  2. 通量 flux = 全高度列（25 层）dens 求和（总迁徙通量）；
  3. rotor_height_fraction = 转子带(0-200m)通量 / 全列通量（逐站逐季经验值，替换硬编码 0.5）；
  4. 夜间(20:00-06:00)、dens>10、春(3-5月)/秋(8-11月)过滤不变；
  5. 重新校验方向（春北迁/秋南迁）；deess 站旋翼带无鸟 → 排除其对应 2 场。

输出：data/processed/farm_direction_signatures_corrected.csv
"""
import os, io, sys, math
from collections import defaultdict
from datetime import datetime
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
VPTS_DIR = os.path.join(REPO, 'our_work', 'data', 'raw', 'radar_vpts')
OUT = os.path.join(REPO, 'our_work', 'data', 'processed')
FARMS_CSV = r'D:\1风力发电实习\wind-direction-to-electricity-transition-main\offshore-task0-HuTingxian\output\task0\farms_master.csv'

ROTOR_LAYER = 0        # HGHT=0 -> 0-200m 层，IEA-10MW 旋翼 30-208m 所在层
DENSITY_THRESHOLD = 10
VALID_STATIONS = {
    'nlhrw': (52.95, 4.75),   # Den Helder
    'bejab': (51.18, 3.07),   # Jabbeke
    'deess': (51.40, 6.97),   # Essen
    'frabb': (50.13, 1.83),   # Abbeville
}

# station -> season -> {rotor_bearings, rotor_dens, total_flux}
data = defaultdict(lambda: defaultdict(lambda: {
    'rotor_b': [], 'rotor_d': [], 'total': 0.0}))

for fname in sorted(os.listdir(VPTS_DIR)):
    if not fname.endswith('.txt'):
        continue
    st = fname.split('_')[0]
    if st not in VALID_STATIONS:
        continue
    with open(os.path.join(VPTS_DIR, fname), encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 12:
                continue
            try:
                hght = int(p[2]); dd = p[7]; dens = p[11]
                if dd == 'nan' or dens == 'nan':
                    continue
                dd = float(dd); dens = float(dens)
                if dens < DENSITY_THRESHOLD:
                    continue
                hour = int(p[1][:2])
                if 6 < hour < 20:
                    continue
                m = datetime.strptime(p[0], '%Y%m%d').month
                if 3 <= m <= 5:
                    season = 'spring'
                elif 8 <= m <= 11:
                    season = 'autumn'
                else:
                    continue
                d = data[st][season]
                d['total'] += dens            # 全列通量
                if hght == ROTOR_LAYER:
                    d['rotor_b'].append(dd)
                    d['rotor_d'].append(dens)
            except (ValueError, IndexError):
                continue


def circ_mean(bearings, weights):
    tot = sum(weights)
    x = sum(w * math.cos(math.radians(b)) for b, w in zip(bearings, weights))
    y = sum(w * math.sin(math.radians(b)) for b, w in zip(bearings, weights))
    if tot == 0:
        return float('nan'), float('nan'), 0.0
    return (math.degrees(math.atan2(y, x)) % 360, math.hypot(x, y) / tot, tot)


print('=== 修正后的逐站逐季签名（0-200m 转子层） ===')
print(f"{'station':8} {'season':8} {'n':>7} {'dir':>8} {'conc':>6} {'rotor_flux':>12} {'total_flux':>12} {'rhf':>6}  校验")
print('-' * 95)
sigs = {}
for st in VALID_STATIONS:
    for season in ['spring', 'autumn']:
        d = data[st][season]
        direc, conc, rotor_flux = circ_mean(d['rotor_b'], d['rotor_d'])
        total = d['total']
        rhf = rotor_flux / total if total > 0 else float('nan')
        n = len(d['rotor_b'])
        north = (315 <= direc <= 360 or 0 <= direc <= 45)
        south = (135 <= direc <= 270)
        ok = (season == 'spring' and north) or (season == 'autumn' and south)
        sigs[(st, season)] = dict(direction=direc, concentration=conc,
                                  flux=round(total, 1), rotor_height_fraction=round(rhf, 4),
                                  n=n, ok=bool(ok) if n > 0 else False)
        print(f"{st:8} {season:8} {n:>7} {direc:8.1f} {conc:6.3f} {rotor_flux:12.1f} {total:12.1f} {rhf:6.3f}  "
              f"{'OK' if ok else ('SUSPECT' if n > 0 else 'NO-DATA')}")

# ---- 农场映射（沿用最近站规则，但 deess 无转子带鸟 -> 排除）----
farms = pd.read_csv(FARMS_CSV)
farms.columns = [c.strip('\ufeff') for c in farms.columns]
EXCLUDE_STATION = {'deess'}   # 旋翼带无鸟，方向不可靠
USABLE = {s: c for s, c in VALID_STATIONS.items() if s not in EXCLUDE_STATION}


def nearest(lat, lon):
    best_d, best_s = float('inf'), None
    for sn, (rlat, rlon) in USABLE.items():
        dd_km = math.hypot(lat - rlat, lon - rlon) * 111
        if dd_km < best_d:
            best_d, best_s = dd_km, sn
    return best_s, best_d


rows = []
for _, farm in farms.iterrows():
    fid = farm['farm_id']; c = farm['country']
    if c in ['China', 'Vietnam', 'Taiwan', 'Japan', 'South Korea', 'United States of America']:
        continue
    lat, lon = farm['centroid_lat'], farm['centroid_lon']
    sn, sd = nearest(lat, lon)
    if sd > 400:
        continue
    farm_rows = []
    for season in ['spring', 'autumn']:
        s = sigs[(sn, season)]
        if not s['ok']:
            continue
        farm_rows.append({
            'farm_id': fid, 'receptor_id': f'radar_{sn}', 'season': season,
            'direction_deg': round(s['direction'], 1),
            'concentration': round(s['concentration'], 3),
            'flux': s['flux'],
            'rotor_height_fraction': s['rotor_height_fraction'],
            'evidence_level': 'radar', 'conservation_weight': 1.0,
            'n_observations': s['n'], 'source': f'VPTS_{sn}_2020_rotor_dist{int(sd)}km',
        })
    # 仅保留春、秋两季方向均有效的场（单季缺失 -> 归入 Bauer/ERA5 兜底）
    if len(farm_rows) == 2:
        rows.extend(farm_rows)

out_df = pd.DataFrame(rows)
out_path = os.path.join(OUT, 'farm_direction_signatures_corrected.csv')
out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n修正后 VPTS 实测场数: {out_df['farm_id'].nunique()} (原 37 -> 排除 deess 2 场)")
print(f"各站场数:\n{out_df.drop_duplicates('farm_id')['receptor_id'].value_counts()}")
print(f"签名行数: {len(out_df)}")
print(f"已保存: {out_path}")