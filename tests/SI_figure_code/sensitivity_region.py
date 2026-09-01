# -*- coding: utf-8 -*-
"""
S7.17 区域分层：陆上风场按地理区域（UK / Central Europe / Nordic / Iberia / France）
分层，统计 budget=0.01 的 RR 中位/均值与场数。替代被废弃的国别维度（OSM country 字段
76% 缺失且多为厂商名片段，非 ISO 国别码）。

区域判定复用 onshore_tradeoff.py 第 258–271 行的经纬度框逻辑。数字从数据注入。
"""
import os
import json
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, '..', 'data', 'processed')


def assign_region(lat, lon):
    if 50 < lat < 60 and -5 < lon < 2:
        return 'UK'
    if 47 < lat < 55 and 2 < lon < 15:
        return 'Central Europe'
    if 55 < lat < 70 and 5 < lon < 30:
        return 'Nordic'
    if 36 < lat < 44 and -10 < lon < 5:
        return 'Iberia'
    if 43 < lat < 47 and -2 < lon < 8:
        return 'France'
    return 'Other'


def main():
    df = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'))
    one = df[df.budget == 0.01].copy()
    one['region'] = [assign_region(la, lo)
                     for la, lo in zip(one['centroid_lat'], one['centroid_lon'])]

    order = ['UK', 'Central Europe', 'Nordic', 'Iberia', 'France', 'Other']
    rows = []
    for r in order:
        g = one[one.region == r]
        if len(g) == 0:
            continue
        rows.append({
            'region': r,
            'n': int(len(g)),
            'rr_med': round(float(g['risk_reduction'].median()), 2),
            'rr_mean': round(float(g['risk_reduction'].mean()), 2),
            'aep_cost_med': round(float(g['aep_cost_pct'].median()), 3),
        })

    out_csv = os.path.join(PROC, 'sensitivity_region.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {'n_farms': int(len(one)), 'regions': rows}
    sj = os.path.join(PROC, 'sensitivity_summary.json')
    with open(sj, encoding='utf-8') as fh:
        allsum = json.load(fh)
    allsum['S717_region'] = summary
    with open(sj, 'w', encoding='utf-8') as fh:
        json.dump(allsum, fh, ensure_ascii=False, indent=1)

    print('=== S7.17 区域分层 ===')
    for r in rows:
        print(f"  {r['region']:16s} n={r['n']:5d}  rr_med={r['rr_med']:6.2f}  "
              f"rr_mean={r['rr_mean']:6.2f}  aep_cost_med={r['aep_cost_med']}")
    print(f'  saved -> {out_csv}; updated -> {sj}')


if __name__ == '__main__':
    main()
