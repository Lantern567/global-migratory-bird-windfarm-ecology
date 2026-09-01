# -*- coding: utf-8 -*-
"""
S7.15 春秋单季分解敏感性：spring-only / autumn-only / 集中度加权（full）三口径的 RR。

原理：AEP 最优朝向 θ_econ 仅由风资源决定，与迁徙方向无关，因此三口径共用同一 θ_econ；
只有 E(θ) 曲线随口径变化。对陆上 4,191 场（budget=0.01）分别以单季方向重算暴露曲线并
套 onshore_tradeoff.compute_tradeoff 得到各口径 RR，比较单季口径与正文口径的差异。

不重跑 AEP（转子/轮毂缺省值不变）。数字一律从数据注入，不手工转写。
"""
import os
import json
import numpy as np
import pandas as pd

import onshore_tradeoff as ot

PROC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')


def circ(a, b):
    """循环角距（0–180°，180° 周期）。"""
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)


def main():
    aep = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
    trade = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
    one = trade[trade.budget == 0.01].copy()

    # AEP 曲线字典：farm_id -> 18 朝向 AEP 数组
    aep_dict = {}
    for _, r in aep.iterrows():
        aep_dict[r['farm_id']] = np.array([r[f'aep_{int(t):03d}'] for t in ot.ORIENTATIONS])

    rows = []
    max_diff_full = 0.0
    for _, r in one.iterrows():
        fid = r['farm_id']
        if fid not in aep_dict:
            continue
        aep_kwh = aep_dict[fid]
        if aep_kwh.max() <= 0:
            continue
        sd, ad = r['spring_dir'], r['autumn_dir']
        cs, ca = r['spring_conc'], r['autumn_conc']

        E_full = ot.compute_exposure_curve(sd, ad, cs, ca)
        E_sp = ot.compute_exposure_curve(sd, ad, 1.0, 0.0)
        E_au = ot.compute_exposure_curve(sd, ad, 0.0, 1.0)

        t_full = ot.compute_tradeoff(aep_kwh, E_full, budgets=[0.01])[0]
        t_sp = ot.compute_tradeoff(aep_kwh, E_sp, budgets=[0.01])[0]
        t_au = ot.compute_tradeoff(aep_kwh, E_au, budgets=[0.01])[0]

        # 一致性校验：重算 full 应等于 CSV 中的 risk_reduction
        max_diff_full = max(max_diff_full, abs(t_full['risk_reduction'] - r['risk_reduction']))

        rows.append({
            'farm_id': fid,
            'rr_full': t_full['risk_reduction'],
            'rr_spring': t_sp['risk_reduction'],
            'rr_autumn': t_au['risk_reduction'],
            'theta_eco_full': t_full['theta_eco'],
            'theta_eco_spring': t_sp['theta_eco'],
            'theta_eco_autumn': t_au['theta_eco'],
            'axis_sep_deg': circ(sd, ad),
        })

    df = pd.DataFrame(rows)
    n = len(df)
    out_csv = os.path.join(PROC, 'onshore_seasonal_decompose.csv')
    df.to_csv(out_csv, index=False)

    summary = {
        'n': n,
        'rr_med_full': round(float(df['rr_full'].median()), 2),
        'rr_med_spring': round(float(df['rr_spring'].median()), 2),
        'rr_med_autumn': round(float(df['rr_autumn'].median()), 2),
        'rr_mean_full': round(float(df['rr_full'].mean()), 2),
        'rr_mean_spring': round(float(df['rr_spring'].mean()), 2),
        'rr_mean_autumn': round(float(df['rr_autumn'].mean()), 2),
        'diff_spring_autumn_med_pp': round(float((df['rr_spring'] - df['rr_autumn']).median()), 2),
        'diff_spring_autumn_mean_pp': round(float((df['rr_spring'] - df['rr_autumn']).mean()), 2),
        'axis_sep_med_deg': round(float(df['axis_sep_deg'].median()), 2),
        'recompute_full_max_abs_diff_pp': round(float(max_diff_full), 4),
    }

    # 注入 sensitivity_summary.json
    sj = os.path.join(PROC, 'sensitivity_summary.json')
    with open(sj, encoding='utf-8') as fh:
        allsum = json.load(fh)
    allsum['S715_seasonal_decompose'] = summary
    with open(sj, 'w', encoding='utf-8') as fh:
        json.dump(allsum, fh, ensure_ascii=False, indent=1)

    print('=== S7.15 春秋单季分解 ===')
    print(f'  farms (budget=0.01): {n}')
    print(f'  RR median  full={summary["rr_med_full"]}  spring={summary["rr_med_spring"]}  '
          f'autumn={summary["rr_med_autumn"]}')
    print(f'  RR mean    full={summary["rr_mean_full"]}  spring={summary["rr_mean_spring"]}  '
          f'autumn={summary["rr_mean_autumn"]}')
    print(f'  spring-autumn median diff = {summary["diff_spring_autumn_med_pp"]} pp')
    print(f'  spring-autumn mean   diff = {summary["diff_spring_autumn_mean_pp"]} pp')
    print(f'  axis separation median   = {summary["axis_sep_med_deg"]} deg')
    print(f'  consistency: max|recomputed_full - csv_rr| = {summary["recompute_full_max_abs_diff_pp"]} pp')
    print(f'  saved -> {out_csv}')
    print(f'  updated -> {sj}')


if __name__ == '__main__':
    main()