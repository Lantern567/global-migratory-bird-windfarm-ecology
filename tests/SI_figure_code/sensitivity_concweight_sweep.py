# -*- coding: utf-8 -*-
"""
S7.19 浓度权重连续扫描：λ∈[0,1] 线性组合春/秋单季暴露，
E_λ = λ·sin²(θ-sd) + (1-λ)·sin²(θ-ad)，套 compute_tradeoff 得 budget=0.01 的 RR 中位。

λ=0 纯秋、λ=1 纯春、λ=0.5 等权；把 S7.5（等权 vs 集中度）与 S7.15（春秋单季）的离散
口径泛化为连续曲线。复用 onshore_tradeoff.compute_exposure_curve（其内部把 (cs,ca) 归一化
为 (cs,ca)/(cs+ca)，故传入 (λ,1-λ) 即得 w_s=λ、w_a=1-λ）。
"""
import os
import json
import numpy as np
import pandas as pd
import onshore_tradeoff as ot

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, '..', 'data', 'processed')

LAMBDAS = [round(0.1 * i, 2) for i in range(11)]  # 0.0 .. 1.0


def main():
    aep = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
    trade = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
    one = trade[trade.budget == 0.01].copy()

    aep_dict = {}
    for _, r in aep.iterrows():
        aep_dict[r['farm_id']] = np.array([r[f'aep_{int(t):03d}'] for t in ot.ORIENTATIONS])

    farms = []
    for _, r in one.iterrows():
        fid = r['farm_id']
        if fid not in aep_dict:
            continue
        aep_kwh = aep_dict[fid]
        if aep_kwh.max() <= 0:
            continue
        farms.append((r['spring_dir'], r['autumn_dir'], aep_kwh))

    rows = []
    for lam in LAMBDAS:
        rr_list = []
        for sd, ad, aep_kwh in farms:
            E = ot.compute_exposure_curve(sd, ad, lam, 1.0 - lam)
            t = ot.compute_tradeoff(aep_kwh, E, budgets=[0.01])[0]
            rr_list.append(t['risk_reduction'])
        rows.append({
            'lambda': lam,
            'n': len(rr_list),
            'rr_med': round(float(np.median(rr_list)), 2),
            'rr_mean': round(float(np.mean(rr_list)), 2),
        })

    out_csv = os.path.join(PROC, 'sensitivity_concweight_sweep.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {'n_farms': len(farms),
               'lambda': [r['lambda'] for r in rows],
               'rr_med': [r['rr_med'] for r in rows],
               'rr_mean': [r['rr_mean'] for r in rows]}
    sj = os.path.join(PROC, 'sensitivity_summary.json')
    with open(sj, encoding='utf-8') as fh:
        allsum = json.load(fh)
    allsum['S719_concweight_sweep'] = summary
    with open(sj, 'w', encoding='utf-8') as fh:
        json.dump(allsum, fh, ensure_ascii=False, indent=1)

    print('=== S7.19 浓度权重连续扫描 ===')
    for r in rows:
        print(f"  λ={r['lambda']:4.2f}  n={r['n']}  rr_med={r['rr_med']:6.2f}  "
              f"rr_mean={r['rr_mean']:6.2f}")
    print(f'  saved -> {out_csv}; updated -> {sj}')


if __name__ == '__main__':
    main()
