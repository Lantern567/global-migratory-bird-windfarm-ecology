# -*- coding: utf-8 -*-
"""
bootstrap_confidence_intervals.py — bootstrap 95% CIs for the headline energy-ecology
trade-off metrics (1% AEP budget), three groups (onshore / VPTS / Bauer).

Mirrors the "bootstrap 95% CI on the key fit parameter" pattern used in the 69-page
reference SI: resample farms with replacement, recompute the group median (and mean),
repeat 10,000 times, report the 2.5/50/97.5 percentiles of the bootstrap distribution.

Inputs (processed/):
  onshore_tradeoff_results.csv      (budget == 0.01, col `risk_reduction`, `aep_cost_pct`)
  tradeoff_offshore_55farms.csv     (budget == 0.01, col `risk_reduction_pct`, `aep_cost_pct`)
  offshore_farm_directions_55.csv   (source: VPTS / Bauer_grid)
Outputs:
  bootstrap_ci.json                 (written next to sensitivity_summary.json)
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, '..', 'data', 'processed')
SEED = 20260826
N = 10000


def boot_ci(values, n=N, seed=SEED):
    """Percentile-method bootstrap 95% CI of the median and mean of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    meds = np.empty(n)
    means = np.empty(n)
    k = len(values)
    for i in range(n):
        s = values[rng.integers(0, k, size=k)]
        meds[i] = np.median(s)
        means[i] = np.mean(s)
    return {
        'n': int(k),
        'point_median': float(np.median(values)),
        'point_mean': float(np.mean(values)),
        'median_ci_lo': float(np.percentile(meds, 2.5)),
        'median_ci_hi': float(np.percentile(meds, 97.5)),
        'mean_ci_lo': float(np.percentile(means, 2.5)),
        'mean_ci_hi': float(np.percentile(means, 97.5)),
    }


def main():
    on = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
    on1 = on[on.budget == 0.01].copy()
    od = pd.read_csv(os.path.join(PROC, 'offshore_farm_directions_55.csv'), encoding='utf-8-sig')
    to = pd.read_csv(os.path.join(PROC, 'tradeoff_offshore_55farms.csv'), encoding='utf-8-sig')
    to1 = to[to.budget == 0.01].copy()
    to1 = to1.merge(od[['farm_id', 'source']], on='farm_id', how='left')

    out = {
        'onshore_rr_1pct': boot_ci(on1['risk_reduction']),
        'onshore_aepcost_1pct': boot_ci(on1['aep_cost_pct']),
        'vpts_rr_1pct': boot_ci(to1[to1.source == 'VPTS']['risk_reduction_pct']),
        'vpts_aepcost_1pct': boot_ci(to1[to1.source == 'VPTS']['aep_cost_pct']),
        'bauer_rr_1pct': boot_ci(to1[to1.source == 'Bauer_grid']['risk_reduction_pct']),
        'bauer_aepcost_1pct': boot_ci(to1[to1.source == 'Bauer_grid']['aep_cost_pct']),
    }

    dst = os.path.join(PROC, 'bootstrap_ci.json')
    with open(dst, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'wrote {dst}')
    for k, v in out.items():
        print(f"{k}: n={v['n']} median={v['point_median']:.2f} "
              f"[{v['median_ci_lo']:.2f}, {v['median_ci_hi']:.2f}] "
              f"mean={v['point_mean']:.2f} [{v['mean_ci_lo']:.2f}, {v['mean_ci_hi']:.2f}]")


if __name__ == '__main__':
    main()
