# -*- coding: utf-8 -*-
"""
S7.16 缺省值扫描：转子直径 / 轮毂高度 / 容量 缺省值对陆上权衡 RR 的敏感性。

对三个缺省值各做一维扫描（1D），逐点重算 onshore_aep_curves_<tag>.csv（约 4 分钟/点），
再套 onshore_tradeoff.py 得 budget=0.01 的 RR 中位（18 点网格口径，与正文 96.9 一致）。
基线复用现有 onshore_aep_curves.csv / onshore_tradeoff_results.csv（70 m / 70 m / 2000 kW），
不重跑。数字一律从数据注入，不手工转写。

注意：缺省值只影响「场级中位数缺失」的场（场级中位数已由 Bauer 库插补，落缺省者极少），
故转子/轮毂扫描效应预期 ≈ 0，容量扫描（~32% 场插补）效应更大。结论写实。
"""
import os
import json
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, '..', 'data', 'processed')


def _run(cmd):
    print('  $ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def run_aep(flag, val, tag):
    out = f'onshore_aep_curves_{tag}.csv'
    _run([sys.executable, os.path.join(HERE, 'onshore_aep_computation.py'),
          f'--{flag}', str(val), '--out', out])
    return out


def run_tradeoff(aep_file, tag):
    out = f'onshore_tradeoff_results_{tag}.csv'
    _run([sys.executable, os.path.join(HERE, 'onshore_tradeoff.py'),
          '--aep', aep_file, '--out', out])
    return out


def _one(tradeoff_csv):
    df = pd.read_csv(os.path.join(PROC, tradeoff_csv))
    return df[df.budget == 0.01]


def rr_median(tradeoff_csv):
    one = _one(tradeoff_csv)
    return float(one['risk_reduction'].median()), int(len(one))


def _n_true(series):
    return int(series.astype(str).str.lower().eq('true').sum())


def main():
    # 基线（不重跑）：现有 70/70/2000 口径
    base_csv = 'onshore_tradeoff_results.csv'
    rr_base, n_base = rr_median(base_csv)
    base_one = _one(base_csv)
    n_imp = {
        'rotor': _n_true(base_one['rotor_diam_imputed']),
        'hub': _n_true(base_one['hub_height_imputed']),
        'capacity': _n_true(base_one['capacity_imputed']),
    }
    print(f'baseline (70/70/2000): RR median={rr_base:.2f}%, n={n_base}')
    print(f'  n_imputed in budget=0.01 subset: rotor={n_imp["rotor"]}, '
          f'hub={n_imp["hub"]}, capacity={n_imp["capacity"]}\n')

    summary = {'n_farms_budget1pct': n_base, 'baseline_rr_med': round(rr_base, 2)}

    spec = [
        ('rotor', 60.0, 90.0, 110.0, 70.0, 'rotor', 'rotor'),
        ('hub', 60.0, 90.0, 110.0, 70.0, 'hub', 'hub'),
        ('capacity', 1500.0, 2500.0, 3000.0, 2000.0, 'capacity', 'cap'),
    ]

    for flag, v1, v2, v3, base_val, key, tagpref in spec:
        # 值列表：非基线 3 点 + 基线 1 点，排序后输出
        vals = sorted([v1, v2, v3, base_val])
        rr = {}
        for v in vals:
            if abs(v - base_val) < 1e-9:
                # 基线复用
                rr[v] = rr_base
                continue
            tag = f'{tagpref}{int(v):03d}'
            aep_file = run_aep(flag, v, tag)
            trade_file = run_tradeoff(aep_file, tag)
            m, n = rr_median(trade_file)
            rr[v] = round(m, 2)
            print(f'  {flag}={v}: RR median={m:.2f}% (n={n})', flush=True)

        defaults = [v for v in vals]
        rr_med = [rr[v] for v in vals]
        delta_pp = [round(rr[v] - rr_base, 2) for v in vals]
        summary[key] = {
            'defaults': defaults,
            'rr_med_1pct': rr_med,
            'delta_pp': delta_pp,
            'n_imputed': n_imp[key],
            'max_abs_delta_pp': round(max(abs(d) for d in delta_pp), 2),
        }
        print(f'\n{key} scan done: defaults={defaults}, rr_med={rr_med}, '
              f'delta_pp={delta_pp}, n_imputed={n_imp[key]}\n')

    # 注入 sensitivity_summary.json
    sj = os.path.join(PROC, 'sensitivity_summary.json')
    with open(sj, encoding='utf-8') as fh:
        allsum = json.load(fh)
    allsum['S716_defaults_scan'] = summary
    with open(sj, 'w', encoding='utf-8') as fh:
        json.dump(allsum, fh, ensure_ascii=False, indent=1)

    print('=== S7.16 缺省值扫描完成 ===')
    for key in ('rotor', 'hub', 'capacity'):
        d = summary[key]
        print(f"  {key}: defaults={d['defaults']}  rr_med={d['rr_med_1pct']}  "
              f"delta_pp={d['delta_pp']}  n_imputed={d['n_imputed']}")
    print(f'  updated -> {sj}')


if __name__ == '__main__':
    main()
