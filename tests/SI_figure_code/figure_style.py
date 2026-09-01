# -*- coding: utf-8 -*-
"""
figure_style.py —— 论文制图共享层（实事求是，严谨）。

职责：
  1. 全局样式（SciencePlots `nature` + no-latex，色盲安全 Okabe-Ito 色板，≥8pt，矢量导出）。
  2. 数据加载（与 final_numbers.py 完全同源、同口径）。
  3. 指标计算（逐场 E(θ)、相对变化、角差、可削减比例、Δθ50/Δθ80、
     1% 预算 RR/代价、R4.3 饱和比例），确保图中数字与 final_numbers.txt 一致。

图内文字一律英文（学长要求），正文仍中文。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, '..', 'data', 'processed')
FIG = os.path.join(BASE, '..', 'figures_v2')
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. 全局样式
# ---------------------------------------------------------------------------
try:
    import scienceplots  # noqa: F401
    plt.style.use(['nature', 'no-latex'])
except Exception:
    pass

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.unicode_minus': False,
})

# ---------------------------------------------------------------------------
# 2. 色盲安全色板（Okabe-Ito）
# ---------------------------------------------------------------------------
C_LAND   = '#0072B2'   # 陆上 onshore（蓝）
C_VPTS   = '#009E73'   # 海上·VPTS（绿）
C_BAUER  = '#CC79A7'   # 海上·Bauer（紫）
C_SPRING = '#E69F00'   # 春季（橙）—— 全局固定
C_AUTUMN = '#56B4E9'   # 秋季（天蓝）—— 全局固定
C_ECON   = '#444444'   # AEP 最优 θ_econ（深灰）
C_ECO    = '#D55E00'   # 生态最优 θ_min（朱红）—— 全局固定

GROUP_COLORS = {'Onshore': C_LAND, 'VPTS': C_VPTS, 'Bauer': C_BAUER}
GROUP_ORDER = ['Onshore', 'VPTS', 'Bauer']

# 连续色标（色盲安全）：方向集中度 viridis，暴露降/可削减比例 cividis
CMAP_CONC = 'viridis'
CMAP_RISK = 'cividis'

# ---------------------------------------------------------------------------
# 3. 数据加载（与 final_numbers.py 同源）
# ---------------------------------------------------------------------------
THETAS = np.arange(0, 180, 1)          # 1° 分辨率，180° 周期


def circ(a, b):
    """循环角距（0–180°，180° 周期）。"""
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)


def on_Evec(sd, ad, cs, ca):
    """陆上几何暴露 E(θ)（浓度加权 sin²），同 onshore_tradeoff.py / final_numbers.py。"""
    cs = float(cs); ca = float(ca)
    if np.isnan(cs) or np.isnan(ca) or (cs + ca) <= 0:
        ws = wa = 0.5
    else:
        ws = cs / (cs + ca); wa = ca / (cs + ca)
    return ws * np.sin(np.radians(THETAS - sd)) ** 2 + wa * np.sin(np.radians(THETAS - ad)) ** 2


def load():
    """加载全部数据，返回 (on, on1, od, to, to1, cur, grid, radar, on_aep)。"""
    on = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
    on1 = on[on.budget == 0.01].copy()

    od = pd.read_csv(os.path.join(PROC, 'offshore_farm_directions_55.csv'), encoding='utf-8-sig')
    to = pd.read_csv(os.path.join(PROC, 'tradeoff_offshore_55farms.csv'), encoding='utf-8-sig')
    to1 = to[to.budget == 0.01].copy()

    cur = pd.read_csv(os.path.join(PROC, 'all_171_exposure_curves_corrected.csv'), encoding='utf-8-sig')
    grid = pd.read_csv(os.path.join(PROC, 'bauer_grid_cell_directions.csv'), encoding='utf-8-sig')
    radar = pd.read_csv(os.path.join(PROC, 'radar_station_signatures_v3.csv'), encoding='utf-8-sig')
    on_aep = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
    return on, on1, od, to, to1, cur, grid, radar, on_aep


def off_Evec(fid, cur):
    """海上几何暴露 E(θ)（绝对 signature 暴露，1° 插值）。"""
    sub = cur[cur.farm_id == fid].sort_values('theta_deg')
    return np.interp(THETAS, sub.theta_deg.values, sub.risk_score.values, period=180.0)


def analyze(theta_econ, E):
    """逐场指标（与 final_numbers.py 的 analyze 完全一致）。返回 dict。

    theta_econ: AEP 最优朝向（°）
    E: E(θ) 曲线，长度 180（对应 THETAS）
    """
    imin = int(np.argmin(E)); Emin = E[imin]; th_min = THETAS[imin]
    Emax = E.max()
    Ee = E[int(round(theta_econ)) % 180]
    avoid = (Ee - Emin) / Ee if Ee > 1e-12 else 0.0
    d_full = circ(theta_econ, th_min)

    # 判断从 θ_econ 往哪个方向走能更短到达 θ_min（用于 Δθ 系列）
    sgn = 1
    for s in (1, -1):
        if circ((theta_econ + s * d_full) % 180, th_min) < 1.0:
            sgn = s
            break

    out = dict(Ee=Ee, Emin=Emin, Emax=Emax, avoid=avoid, d_full=d_full, th_min=th_min)
    for dth in (5, 10, 20, 30):
        th = (theta_econ + sgn * dth) % 180
        Ev = E[int(round(th)) % 180]
        red = (Ee - Ev) / Ee if Ee > 1e-12 else 0.0
        frac = (Ee - Ev) / (Ee - Emin) if (Ee - Emin) > 1e-12 else 0.0
        out[f'red{dth}'] = red
        out[f'frac{dth}'] = frac

    d50 = d80 = None
    for dth in range(0, 91):
        th = (theta_econ + sgn * dth) % 180
        Ev = E[int(round(th)) % 180]
        frac = (Ee - Ev) / (Ee - Emin) if (Ee - Emin) > 1e-12 else 0.0
        if d50 is None and frac >= 0.5:
            d50 = dth
        if d80 is None and frac >= 0.8:
            d80 = dth
    out['d50'] = d50
    out['d80'] = d80
    return out


def compute_metrics():
    """计算三组逐场指标，返回 (on_df, vp_df, ba_df, ctx)。

    ctx 含：grid（方向场）、radar（雷达站）、on1（陆上元数据+θ/代价/RR）、
    vpts/bauer 元数据（质心、source）、off_aep（海上 AEP 曲线）。
    """
    on, on1, od, to, to1, cur, grid, radar, on_aep = load()

    # 陆上
    onrows = []
    for _, r in on1.iterrows():
        E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
        m = analyze(r.theta_econ, E)
        m['rel'] = (m['Emax'] - m['Emin']) / m['Emax'] if m['Emax'] > 1e-12 else 0.0
        m['aep'] = r.aep_cost_pct
        m['rr'] = r.risk_reduction
        m['theta_econ'] = r.theta_econ
        m['theta_eco'] = r.theta_eco
        m['centroid_lat'] = r.centroid_lat
        m['centroid_lon'] = r.centroid_lon
        m['farm_id'] = r.farm_id
        onrows.append(m)
    on_df = pd.DataFrame(onrows)

    # 海上
    vpts_ids = set(od[od.source == 'VPTS']['farm_id'])
    bauer_ids = set(od[od.source == 'Bauer_grid']['farm_id'])

    def _off_rows(ids):
        rows = []
        for f in sorted(ids):
            t = to1[to1.farm_id == f]
            if len(t) == 0:
                continue
            r = t.iloc[0]
            E = off_Evec(f, cur)
            m = analyze(r.theta_econ, E)
            m['rel'] = (m['Emax'] - m['Emin']) / m['Emax'] if m['Emax'] > 1e-12 else 0.0
            m['aep'] = r.aep_cost_pct
            m['rr'] = r.risk_reduction_pct
            m['theta_econ'] = r.theta_econ
            m['theta_eco'] = r.theta_eco
            m['farm_id'] = f
            rows.append(m)
        return pd.DataFrame(rows)

    vp_df = _off_rows(vpts_ids)
    ba_df = _off_rows(bauer_ids)

    ctx = dict(on=on, on1=on1, od=od, to=to, to1=to1, cur=cur, grid=grid,
               radar=radar, on_aep=on_aep, vpts_ids=vpts_ids, bauer_ids=bauer_ids)
    return on_df, vp_df, ba_df, ctx


def r43_saturation(ctx):
    """R4.3：2% -> 5% AEP 无新增暴露收益场址比例（与 final_numbers.py 一致）。"""
    on, _, _, to, _, _, _, _, _ = (ctx['on'], ctx['on1'], ctx['od'], ctx['to'],
                                   ctx['to1'], ctx['cur'], ctx['grid'], ctx['radar'],
                                   ctx['on_aep'])
    res = {}
    for name, df_all, ids in [('Onshore', on, None),
                              ('VPTS', to, ctx['vpts_ids']),
                              ('Bauer', to, ctx['bauer_ids'])]:
        sub = df_all if ids is None else df_all[df_all.farm_id.isin(ids)]
        col = 'risk_reduction' if ids is None else 'risk_reduction_pct'
        b2 = sub[sub.budget == 0.02].set_index('farm_id')[col]
        b5 = sub[sub.budget == 0.05].set_index('farm_id')[col]
        ids2 = b2.index.intersection(b5.index)
        delta = (b5[ids2] - b2[ids2])
        res[name] = (delta.abs() < 0.01).mean() * 100
    return res


# ---------------------------------------------------------------------------
# 4. 面板工具
# ---------------------------------------------------------------------------
def style_ax(ax, top=False, right=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6, length=2.5, direction='out')


def panel_label(ax, txt, x=0.02, y=0.98, fs=12, ha='left', va='top'):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs, fontweight='bold',
            va=va, ha=ha, color='black')


def savefig(fig, name):
    """同时导出 PDF（矢量）与 PNG（300dpi，Word 用）。"""
    pdf = os.path.join(FIG, name + '.pdf')
    png = os.path.join(FIG, name + '.png')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=300)
    return pdf, png