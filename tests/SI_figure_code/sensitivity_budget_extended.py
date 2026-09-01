# -*- coding: utf-8 -*-
"""
S7.18 更细 AEP 预算扫描：在现有 AEP 曲线上扩展预算档位（0.1%–20%），
输出各预算下 RR 中位（18 点网格口径）。纯 numpy 复用 compute_tradeoff，不重跑 AEP。

基线 tradeoff 只含 {0.5%, 1%, 2%, 5%} 四档；此处扩展到 8 档，把「RR 随预算」关系
画成连续曲线。
"""
import os
import json
import numpy as np
import pandas as pd
import onshore_tradeoff as ot

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, '..', 'data', 'processed')

BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def main():
    aep = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
    trade = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
    one = trade[trade.budget == 0.01].copy()

    aep_dict = {}
    for _, r in aep.iterrows():
        aep_dict[r['farm_id']] = np.array([r[f'aep_{int(t):03d}'] for t in ot.ORIENTATIONS])

    budget_rr = {b: [] for b in BUDGETS}
    n = 0
    for _, r in one.iterrows():
        fid = r['farm_id']
        if fid not in aep_dict:
            continue
        aep_kwh = aep_dict[fid]
        if aep_kwh.max() <= 0:
            continue
        E = ot.compute_exposure_curve(r['spring_dir'], r['autumn_dir'],
                                      r['spring_conc'], r['autumn_conc'])
        ts = ot.compute_tradeoff(aep_kwh, E, budgets=BUDGETS)
        for t in ts:
            budget_rr[t['budget']].append(t['risk_reduction'])
        n += 1

    rows = []
    for b in BUDGETS:
        rr = np.array(budget_rr[b])
        rows.append({
            'budget': b,
            'n': int(len(rr)),
            'rr_med': round(float(np.median(rr)), 2),
            'rr_mean': round(float(np.mean(rr)), 2),
        })

    out_csv = os.path.join(PROC, 'sensitivity_budget_extended.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {'n_farms': n, 'budgets': rows}
    sj = os.path.join(PROC, 'sensitivity_summary.json')
    with open(sj, encoding='utf-8') as fh:
        allsum = json.load(fh)
    allsum['S718_budget_extended'] = summary
    with open(sj, 'w', encoding='utf-8') as fh:
        json.dump(allsum, fh, ensure_ascii=False, indent=1)

    print('=== S7.18 更细预算扫描 ===')
    for r in rows:
        print(f"  budget={r['budget']*100:6.3f}%  n={r['n']}  "
              f"rr_med={r['rr_med']:6.2f}  rr_mean={r['rr_mean']:6.2f}")
    print(f'  saved -> {out_csv}; updated -> {sj}')


if __name__ == '__main__':
    main()
