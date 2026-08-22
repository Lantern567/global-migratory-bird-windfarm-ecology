# -*- coding: utf-8 -*-
"""
FINAL authoritative numbers for the advisor's R1-R4 framework.
All values computed from CORRECTED data:
  - onshore: 4191 farms (concentration-weighted geometric exposure, 0-1)
  - offshore: VPTS 29 (radar), Bauer 26 (grid), absolute signature exposure
Writes a clean UTF-8 report.
"""
import os
import numpy as np
import pandas as pd

P = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'final_numbers.txt'))

L = []
def p(s=''):
    L.append(str(s))

# ---------------- load ----------------
on = pd.read_csv(os.path.join(P, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
on1 = on[on.budget == 0.01].copy()
od = pd.read_csv(os.path.join(P, 'offshore_farm_directions_55.csv'), encoding='utf-8-sig')
vpts = set(od[od.source == 'VPTS']['farm_id'])
bauer = set(od[od.source == 'Bauer_grid']['farm_id'])
to = pd.read_csv(os.path.join(P, 'tradeoff_offshore_55farms.csv'), encoding='utf-8-sig')
to1 = to[to.budget == 0.01].copy()
cur = pd.read_csv(os.path.join(P, 'all_171_exposure_curves_corrected.csv'), encoding='utf-8-sig')
grid = pd.read_csv(os.path.join(P, 'bauer_grid_cell_directions.csv'), encoding='utf-8-sig')

thetas = np.arange(0, 180, 1)

def circ(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)

# ---------------- R1.1 ----------------
gv = grid.dropna(subset=['spring_dir', 'autumn_dir'])
sp, au = gv.spring_dir.median(), gv.autumn_dir.median()
p('=' * 60)
p('R1.1 迁徙方向结构 (Bauer 格网, %d 单元)' % len(gv))
p('  春季中位方向 %.1f deg ; 秋季中位方向 %.1f deg' % (sp, au))
p('  春季集中度中位 %.3f ; 秋季集中度中位 %.3f' % (gv.spring_conc.median(), gv.autumn_conc.median()))
p('  有向春秋差 %.1f deg ; 暴露几何同轴差 %.1f deg' % (circ(sp, (au+180) % 360), circ(sp, au)))

# ---------------- analyze helpers ----------------
def on_Evec(sd, ad, cs, ca):
    cs = float(cs); ca = float(ca)
    if np.isnan(cs) or np.isnan(ca) or (cs+ca) <= 0:
        ws = wa = 0.5
    else:
        ws = cs/(cs+ca); wa = ca/(cs+ca)
    return ws*np.sin(np.radians(thetas-sd))**2 + wa*np.sin(np.radians(thetas-ad))**2

def off_Evec(fid):
    sub = cur[cur.farm_id == fid].sort_values('theta_deg')
    return np.interp(thetas, sub.theta_deg.values, sub.risk_score.values, period=180.0)

def analyze(te, E):
    imin = int(np.argmin(E)); Emin = E[imin]; th_min = thetas[imin]
    Emax = E.max()
    Ee = E[int(round(te)) % 180]
    avoid = (Ee - Emin)/Ee if Ee > 1e-12 else 0.0
    d_full = circ(te, th_min)
    sgn = 1
    for s in (1, -1):
        if circ((te+s*d_full) % 180, th_min) < 1.0:
            sgn = s; break
    out = dict(Ee=Ee, Emin=Emin, Emax=Emax, avoid=avoid, d_full=d_full)
    for dth in (5, 10, 20, 30):
        th = (te + sgn*dth) % 180
        Ev = E[int(round(th)) % 180]
        red = (Ee - Ev)/Ee if Ee > 1e-12 else 0.0   # (a) exposure reduction
        frac = (Ee - Ev)/(Ee - Emin) if (Ee - Emin) > 1e-12 else 0.0  # (b) fraction of max
        out[f'red{dth}'] = red
        out[f'frac{dth}'] = frac
    d50 = d80 = None
    for dth in range(0, 91):
        th = (te + sgn*dth) % 180
        Ev = E[int(round(th)) % 180]
        frac = (Ee - Ev)/(Ee - Emin) if (Ee - Emin) > 1e-12 else 0.0
        if d50 is None and frac >= 0.5: d50 = dth
        if d80 is None and frac >= 0.8: d80 = dth
    out['d50'] = d50; out['d80'] = d80
    return out

def run(name, rows):
    df = pd.DataFrame(rows)
    p('=' * 60)
    p(f'[{name}]  n={len(df)}')
    p('  R1.2 相对变化 (Emax-Emin)/Emax 中位 %.1f%% ; >50%% 场址 %.1f%% ; >80%% 场址 %.1f%%'
      % (df.rel.median()*100, (df.rel>0.5).mean()*100, (df.rel>0.8).mean()*100))
    p('  R2.1 角差 中位 %.0f deg (P25 %.0f, P75 %.0f)' % (df.d_full.median(), df.d_full.quantile(.25), df.d_full.quantile(.75)))
    p('  R2.2 E(θ_econ)中位 %.4g ; E_min中位 %.4g ; 比值中位 %.1fx' % (df.Ee.median(), df.Emin.median(), (df.Ee/df.Emin).median()))
    p('  R2.3 可削减比例(无约束) 中位 %.1f%% ; 均值 %.1f%%' % (df.avoid.median()*100, df.avoid.mean()*100))
    for dth in (5, 10, 20, 30):
        p('  R3.1 %2d deg: 暴露下降(%% of E_econ) 中位 %5.1f%% ; 占最大可削减 中位 %5.1f%%'
          % (dth, df[f'red{dth}'].median()*100, df[f'frac{dth}'].median()*100))
    p('  R3.2 Δθ50 = %.0f deg ; Δθ80 = %.0f deg (中位)' % (df.d50.median(), df.d80.median()))
    p('  R3.3 ≤20deg 内>50%%: %.1f%% ; ≤30deg 内>50%%: %.1f%% ; ≤30deg 内>80%%: %.1f%%'
      % ((df.frac20>0.5).mean()*100, (df.frac30>0.5).mean()*100, (df.frac30>0.8).mean()*100))
    p('  R4.1 1%% AEP: 代价 中位 %.3f%%/均值 %.3f%% ; RR 中位 %.1f%%/均值 %.1f%%'
      % (df.aep.median(), df.aep.mean(), df.rr.median(), df.rr.mean()))
    p('  R4.2 1%%预算捕获最大可削减 中位 %.1f%%' % ((df.rr/100/df.avoid).median()*100))
    return df

# ---------------- onshore ----------------
p('正在计算陆上 %d 场...' % len(on1))
onrows = []
for _, r in on1.iterrows():
    E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
    m = analyze(r.theta_econ, E)
    m['rel'] = (m['Emax']-m['Emin'])/m['Emax'] if m['Emax']>1e-12 else 0
    m['aep'] = r.aep_cost_pct
    m['rr'] = r.risk_reduction
    onrows.append(m)
run('陆上', onrows)

# ---------------- offshore ----------------
for name, ids in [('海上·VPTS', vpts), ('海上·Bauer', bauer)]:
    rows = []
    for f in sorted(ids):
        t = to1[to1.farm_id == f]
        if len(t) == 0: continue
        r = t.iloc[0]
        E = off_Evec(f)
        m = analyze(r.theta_econ, E)
        m['rel'] = (m['Emax']-m['Emin'])/m['Emax'] if m['Emax']>1e-12 else 0
        m['aep'] = r.aep_cost_pct
        m['rr'] = r.risk_reduction_pct
        rows.append(m)
    run(name, rows)

# ---------------- R4.3 ----------------
p('=' * 60)
p('R4.3  2%% -> 5%% AEP 无新增暴露收益场址比例')
for name, df_all, ids in [('陆上', on, None), ('VPTS', to, vpts), ('Bauer', to, bauer)]:
    sub = df_all if ids is None else df_all[df_all.farm_id.isin(ids)]
    col = 'risk_reduction' if ids is None else 'risk_reduction_pct'
    b2 = sub[sub.budget == 0.02].set_index('farm_id')[col]
    b5 = sub[sub.budget == 0.05].set_index('farm_id')[col]
    ids2 = b2.index.intersection(b5.index)
    delta = (b5[ids2]-b2[ids2])
    p('  %s: 无新增(<0.01pp) %.1f%%  (n=%d)' % (name, (delta.abs()<0.01).mean()*100, len(ids2)))

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
print('written', OUT)