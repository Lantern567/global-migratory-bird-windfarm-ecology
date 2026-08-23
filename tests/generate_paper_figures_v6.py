# -*- coding: utf-8 -*-
"""
generate_paper_figures_v6.py —— Nature Energy 级重设计（v7 稿件专用）。

相对 v5 的实质升级：
  * 多子图混合图形（hexbin + ridgeline + Pareto + bee-swarm + saturation heatmap
    + directional rose overlay + case-study inset + derivative inset），
    而非单一 violin/bar/scatter 重复。
  * Panel 尺寸分层（大 panel + 小 panel + inset），信息密度贴近 Nature Energy。
  * sequential / divergent / qualitative 三层配色分离使用。
  * 表格 heatmap 化：Fig 5 直接把「预算 × 组」削减率作 heatmap 呈现，替代原 Table 2。

所有数字继续经 figure_style.compute_metrics()，与 final_numbers.txt 一致。
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN, C_ECON, C_ECO,
                          GROUP_COLORS, GROUP_ORDER, THETAS, CMAP_CONC, CMAP_RISK,
                          style_ax, panel_label, circ, on_Evec, off_Evec, r43_saturation)

from generate_paper_figures_v2 import (add_basemap, RADAR_LOC, rep_farm, rep_curve,
                                       HAS_CARTOPY, figS1_threat)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    pass

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
FIG6 = os.path.join(BASE, '..', 'figures_v6')
os.makedirs(FIG6, exist_ok=True)

W_SINGLE, W_ONEHALF, W_DOUBLE, H_MAX = 3.46, 4.72, 7.09, 9.72

plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.2,
    'axes.labelpad': 2.0,
    'axes.titlepad': 3.0,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.1,
    'lines.markersize': 3.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _sty(ax):
    style_ax(ax)
    ax.tick_params(labelsize=6.5, width=0.5, length=2.2, pad=1.5)


def _lab(ax, txt, x=-0.22, y=1.12):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='left', color='black')


def _fig_lab(fig, txt, x, y):
    fig.text(x, y, txt, fontsize=9, fontweight='bold', va='bottom', ha='left', color='black')


def savefig(fig, name):
    pdf = os.path.join(FIG6, name + '.pdf')
    png = os.path.join(FIG6, name + '.png')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=450)
    return pdf, png


def _on_rr_at(ctx, budget):
    sub = ctx['on'][ctx['on'].budget == budget]
    return sub[['farm_id', 'centroid_lat', 'centroid_lon', 'risk_reduction',
                'aep_cost_pct', 'n_turbines']].copy()


def _off_rr_at(ctx, budget):
    od = ctx['od'][['farm_id', 'centroid_lat', 'centroid_lon', 'source']]
    sub = ctx['to'][ctx['to'].budget == budget][['farm_id', 'risk_reduction_pct', 'aep_cost_pct']]
    return od.merge(sub, on='farm_id', how='inner')


def _off_metric(od, df, col):
    return od[['farm_id', 'centroid_lat', 'centroid_lon']].merge(
        df[['farm_id', col]], on='farm_id', how='inner')


def _median_rr_vs_budget(ctx):
    on, to = ctx['on'], ctx['to']
    budgets = [0.005, 0.01, 0.02, 0.05]
    out = {}
    for b in budgets:
        out[b] = [on[on.budget == b]['risk_reduction'].median(),
                  to[(to.budget == b) & to.farm_id.isin(ctx['vpts_ids'])]['risk_reduction_pct'].median(),
                  to[(to.budget == b) & to.farm_id.isin(ctx['bauer_ids'])]['risk_reduction_pct'].median()]
    return budgets, out


def _mean_aep_vs_budget(ctx):
    on, to = ctx['on'], ctx['to']
    budgets = [0.005, 0.01, 0.02, 0.05]
    out = {}
    for b in budgets:
        out[b] = [on[on.budget == b]['aep_cost_pct'].mean(),
                  to[(to.budget == b) & to.farm_id.isin(ctx['vpts_ids'])]['aep_cost_pct'].mean(),
                  to[(to.budget == b) & to.farm_id.isin(ctx['bauer_ids'])]['aep_cost_pct'].mean()]
    return budgets, out


def _iqr_curve(all_curves):
    """all_curves: 2D array (n_farms, 180); returns 25/50/75 percentile arrays."""
    q25 = np.nanpercentile(all_curves, 25, axis=0)
    q50 = np.nanpercentile(all_curves, 50, axis=0)
    q75 = np.nanpercentile(all_curves, 75, axis=0)
    return q25, q50, q75


def _capture_curves_iqr(df, on1, cur, on_flag):
    """给定 group df，返回从 θ_econ 出发的旋转-归一化削减曲线（0–45°）的 IQR。"""
    dths = np.arange(0, 46)
    fids = df.farm_id.values
    mat = np.full((len(fids), len(dths)), np.nan)
    for i, fid in enumerate(fids):
        if on_flag:
            row = on1[on1.farm_id == fid]
            if row.empty:
                continue
            r = row.iloc[0]
            E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
            te = r.theta_econ
        else:
            E = off_Evec(fid, cur)
            te = df.loc[df.farm_id == fid, 'theta_econ'].iloc[0]
        Ee = E[int(round(te)) % 180]
        Emin = E.min()
        if Ee - Emin < 1e-9:
            continue
        # 选择方向
        d_full = circ(te, THETAS[np.argmin(E)])
        sgn = 1
        for s in (1, -1):
            if circ((te + s * d_full) % 180, THETAS[np.argmin(E)]) < 1.0:
                sgn = s
                break
        for j, d in enumerate(dths):
            th = (te + sgn * d) % 180
            Ev = E[int(round(th)) % 180]
            mat[i, j] = (Ee - Ev) / (Ee - Emin) * 100
    return dths, mat


def _hexbin_over(ax, lon, lat, C, cmap, vmin, vmax, gridsize=32, mincnt=1, reduce_C_function=np.median):
    hb = ax.hexbin(lon, lat, C=C, cmap=cmap, gridsize=gridsize, mincnt=mincnt,
                   vmin=vmin, vmax=vmax, edgecolors='none', linewidths=0,
                   reduce_C_function=reduce_C_function, transform=ccrs.PlateCarree(), zorder=3)
    return hb


def _ridge(ax, series_list, labels, colors, x_lo, x_hi, x_label, alpha_fill=0.55):
    """Simple ridge plot: overlaid vertically-shifted KDEs."""
    from scipy.stats import gaussian_kde
    n = len(series_list)
    offset_step = 1.0
    ymax = 0
    for i, (data, lab, c) in enumerate(zip(series_list, labels, colors)):
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if len(data) < 2:
            continue
        try:
            kde = gaussian_kde(data)
        except Exception:
            continue
        xs = np.linspace(x_lo, x_hi, 300)
        ys = kde(xs)
        ys = ys / ys.max() * 0.85
        y_off = (n - 1 - i) * offset_step
        ax.fill_between(xs, y_off, y_off + ys, color=c, alpha=alpha_fill, linewidth=0)
        ax.plot(xs, y_off + ys, color=c, lw=0.7)
        ax.axhline(y_off, color='#888888', lw=0.3, zorder=0)
        ax.text(x_lo - (x_hi - x_lo) * 0.03, y_off + 0.35, lab, ha='right', va='center',
                fontsize=6.3, color=c, fontweight='bold')
        ymax = max(ymax, y_off + 1.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.15, ymax)
    ax.set_xlabel(x_label)
    ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.5, width=0.5, length=2.2)


def _country_bbox(lat, lon):
    """粗略国家归类（陆上样本），依 bbox；仅用于 Fig 4 分组绘图。"""
    if lat > 55 and -8 < lon < 3:
        return 'UK'
    if 49 < lat <= 55 and -8 < lon < 2:
        return 'UK'
    if 50.5 < lat < 55 and 2 <= lon < 7:
        return 'NL/BE'
    if 47 < lat < 55 and 6 <= lon < 15:
        return 'DE'
    if 43 < lat < 51 and -5 < lon < 8:
        return 'FR'
    if lat < 44 and -10 < lon < 4:
        return 'ES'
    if lat > 49 and 14 <= lon < 24:
        return 'PL'
    if 54 < lat < 58 and 7 < lon < 13:
        return 'DK'
    return 'other'


# =====================================================================
# Fig 1 — mechanism + regional evidence (double col, 3 rows)
# =====================================================================
def fig1_mechanism(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; od = ctx['od']; grid = ctx['grid']; cur = ctx['cur']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']

    fig = plt.figure(figsize=(W_DOUBLE, 8.6))

    # 三行 GridSpec，高度不等：概念行、frontier 行、regional evidence 行
    gs_a = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.965, bottom=0.755,
                            wspace=0.42, width_ratios=[1.05, 1.20, 0.95])
    gs_b = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.685, bottom=0.435,
                            wspace=0.44, width_ratios=[0.9, 1.05, 1.15])
    gs_c = fig.add_gridspec(1, 2, left=0.035, right=0.985, top=0.375, bottom=0.045,
                            wspace=0.08, width_ratios=[1.35, 1.0])

    # ---- (a) top-down farm sketch parallel vs perpendicular ----
    axa = fig.add_subplot(gs_a[0, 0])
    axa.set_xlim(0, 10); axa.set_ylim(-1.5, 11.2); axa.set_aspect('equal')
    axa.axis('off')

    def _farm_flat(x0, y0, w, h, rows_vert):
        axa.add_patch(Rectangle((x0, y0), w, h, fill=True, facecolor='#F5F3EA',
                                edgecolor='#666666', lw=0.6))
        if rows_vert:
            xs = np.linspace(x0 + 0.55, x0 + w - 0.55, 3)
            for xi in xs:
                ys = np.linspace(y0 + 0.4, y0 + h - 0.4, 5)
                axa.plot([xi] * len(ys), ys, 'o', color=C_ECON, markersize=3.4, zorder=3)
        else:
            ys = np.linspace(y0 + 0.55, y0 + h - 0.55, 3)
            for yi in ys:
                xs = np.linspace(x0 + 0.4, x0 + w - 0.4, 5)
                axa.plot(xs, [yi] * len(xs), 'o', color=C_ECON, markersize=3.4, zorder=3)

    _farm_flat(0.4, 0.5, 3.4, 8.0, rows_vert=True)
    axa.annotate('', xy=(2.1, 10.6), xytext=(2.1, 8.9),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=1.8, mutation_scale=13))
    _farm_flat(6.2, 0.5, 3.4, 8.0, rows_vert=False)
    axa.annotate('', xy=(7.9, 10.6), xytext=(7.9, 8.9),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=1.8, mutation_scale=13))
    axa.text(5.0, 10.7, 'migration', fontsize=6.2, color=C_SPRING, ha='center', va='center')
    axa.text(2.1, -0.35, 'parallel', fontsize=6.8, ha='center', va='top', fontweight='bold')
    axa.text(2.1, -1.0, 'low exposure', fontsize=5.8, ha='center', va='top', color='#1E8449')
    axa.text(7.9, -0.35, 'perpendicular', fontsize=6.8, ha='center', va='top', fontweight='bold')
    axa.text(7.9, -1.0, 'high exposure', fontsize=5.8, ha='center', va='top', color='#B03A2E')
    axa.set_title('Array geometry sets exposure', fontsize=7.5)
    _lab(axa, 'a', x=-0.02, y=1.05)

    # ---- (b) exposure surface heatmap: E vs (theta, migration dir phi) ----
    axb = fig.add_subplot(gs_a[0, 1])
    phi = np.arange(0, 181, 2)
    theta = np.arange(0, 181, 2)
    T, P = np.meshgrid(theta, phi)
    Z = np.sin(np.radians(T - P)) ** 2
    im = axb.imshow(Z, origin='lower', extent=(0, 180, 0, 180),
                    aspect='auto', cmap='cividis_r', vmin=0, vmax=1)
    # contour
    cs = axb.contour(T, P, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='white',
                     linewidths=0.4, alpha=0.6)
    axb.clabel(cs, fontsize=5, inline=True, fmt='%.1f')
    # 假想 migration phi=100° 处的两个 optima
    phi_ex = 100
    axb.axhline(phi_ex, color='white', ls='--', lw=0.6, alpha=0.8)
    axb.text(3, phi_ex + 3, 'example migration φ = 100°', color='white', fontsize=5.5)
    # 在 phi=100° 上：eco-opt = 100° (E=0)，AEP-opt = 45° (假设 wind resource)
    axb.scatter([45], [phi_ex], marker='X', s=45, c=C_ECON, edgecolors='white',
                linewidths=0.8, zorder=6, label='AEP-opt')
    axb.scatter([100], [phi_ex], marker='o', s=42, c=C_ECO, edgecolors='white',
                linewidths=0.7, zorder=6, label='eco-opt')
    axb.annotate('', xy=(100, phi_ex + 5), xytext=(45, phi_ex + 5),
                 arrowprops=dict(arrowstyle='<->', color='white', lw=0.9))
    axb.text(72.5, phi_ex + 10, r'$\Delta\theta$ = 55°', ha='center', fontsize=6, color='white',
             fontweight='bold')
    axb.set_xlabel(r'Array orientation $\theta$ (°)')
    axb.set_ylabel(r'Migration direction $\varphi$ (°)')
    axb.set_xticks([0, 45, 90, 135, 180]); axb.set_yticks([0, 45, 90, 135, 180])
    axb.set_title(r'Exposure surface $\sin^{2}(\theta{-}\varphi)$', fontsize=7.5)
    axb.legend(loc='upper right', fontsize=5.8, frameon=True, framealpha=0.85,
               edgecolor='none', handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    _sty(axb); _lab(axb, 'b', x=-0.20, y=1.06)
    cax = make_axes_locatable(axb).append_axes('right', size='4%', pad=0.05, axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax); cb.set_label('Norm. exposure', fontsize=6.2)
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    # ---- (c) misalignment compass ring ----
    axc = fig.add_subplot(gs_a[0, 2], projection='polar')
    axc.set_theta_zero_location('N'); axc.set_theta_direction(-1)
    axc.set_thetamin(0); axc.set_thetamax(180)
    # 数据：三组 median AEP-opt vs eco-opt（每组占一个 ring）
    rings = [(on_df, C_LAND, 'Onshore', 0.95),
             (vp_df, C_VPTS, 'VPTS', 0.72),
             (ba_df, C_BAUER, 'Bauer', 0.48)]
    for grp_df, c, lab, off in rings:
        te = np.radians(grp_df.theta_econ.median())
        tm = np.radians(grp_df.th_min.median())
        arc = np.linspace(min(te, tm), max(te, tm), 40)
        axc.plot(arc, [off] * len(arc), color=c, lw=2.0, alpha=0.6,
                 solid_capstyle='round')
        axc.scatter([te], [off], marker='X', s=42, c=C_ECON, edgecolors='white',
                    linewidths=0.7, zorder=5)
        axc.scatter([tm], [off], marker='o', s=38, c=C_ECO, edgecolors='white',
                    linewidths=0.6, zorder=5)
        # 组标签置于弧最左端（0°方向）
        axc.text(np.radians(-3), off, lab, fontsize=6.0, color=c,
                 fontweight='bold', ha='right', va='center')
        # Δ 标签置于弧最右端
        axc.text(np.radians(183), off, f'Δ={abs(np.degrees(te - tm)):.0f}°',
                 fontsize=5.8, color='#333333', ha='left', va='center')
    axc.set_rlim(0, 1.1)
    axc.set_rticks([])
    axc.tick_params(labelsize=5.8, pad=1)
    axc.set_xticks(np.radians([0, 45, 90, 135, 180]))
    axc.set_xticklabels(['N', '45°', 'E', '135°', 'S'], fontsize=5.8)
    axc.set_title('AEP-opt vs eco-opt (compass)', fontsize=7.5, pad=6)
    axc.legend([Line2D([], [], marker='X', ls='', color=C_ECON, markersize=6,
                       markeredgecolor='white', markeredgewidth=0.6),
                Line2D([], [], marker='o', ls='', color=C_ECO, markersize=5,
                       markeredgecolor='white', markeredgewidth=0.5)],
               ['AEP-opt', 'eco-opt'], loc='lower center', bbox_to_anchor=(0.5, -0.20),
               fontsize=5.8, frameon=False, ncol=2, handletextpad=0.3, columnspacing=1.0)
    _fig_lab(fig, 'c', 0.70, 0.955)

    # ---- (d) sin² geometry callout ----
    axd = fig.add_subplot(gs_b[0, 0])
    axd.set_xlim(-0.05, 1.05); axd.set_ylim(-0.05, 1.05); axd.set_aspect('equal')
    axd.axis('off')
    # 圆
    ang = np.linspace(0, 2 * np.pi, 200)
    axd.plot(0.5 + 0.4 * np.cos(ang), 0.5 + 0.4 * np.sin(ang), color='#888888', lw=0.5)
    # 涡轮列（水平）
    axd.plot([0.14, 0.86], [0.5, 0.5], color='#333333', lw=1.4)
    axd.scatter([0.20, 0.35, 0.50, 0.65, 0.80], [0.5] * 5, s=15, c=C_ECON, zorder=3)
    # 迁徙方向 phi
    phi = np.radians(115)
    axd.annotate('', xy=(0.5 + 0.4 * np.cos(phi), 0.5 + 0.4 * np.sin(phi)),
                 xytext=(0.5, 0.5),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=1.4))
    # theta 标注
    axd.plot([0.5, 0.9], [0.5, 0.5], color=C_ECON, lw=1.0)
    axd.text(0.68, 0.44, r'$\theta$', color=C_ECON, fontsize=8, ha='center')
    axd.text(0.5 + 0.42 * np.cos(phi), 0.5 + 0.42 * np.sin(phi),
             r'$\varphi$ migration', color=C_SPRING, fontsize=6.5, ha='left', va='bottom')
    # 公式
    axd.text(0.5, 0.05, r'$E\propto\sin^{2}(\theta-\varphi)$', ha='center', fontsize=8,
             color='#222222')
    axd.set_title(r'Geometric model', fontsize=7.5)
    _lab(axd, 'd', x=-0.02, y=1.05)

    # ---- (e) capture curve IQR bands (three groups, from θ_econ) ----
    axe = fig.add_subplot(gs_b[0, 1])
    for df, c, lab, is_on in [(on_df, C_LAND, 'Onshore', True),
                              (vp_df, C_VPTS, 'VPTS', False),
                              (ba_df, C_BAUER, 'Bauer', False)]:
        dths, mat = _capture_curves_iqr(df, on1, cur, is_on)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q50 = np.nanpercentile(mat, 50, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        axe.fill_between(dths, q25, q75, color=c, alpha=0.18, linewidth=0)
        axe.plot(dths, q50, color=c, lw=1.4, label=lab)
    axe.axhline(50, color='#888888', lw=0.6, ls=':')
    axe.text(1.5, 53, 'half of max gain', fontsize=5.8, color='#555555')
    axe.axvspan(0, 20, color=C_ECO, alpha=0.08, zorder=0)
    axe.axvline(20, color=C_ECO, lw=0.8, ls='--')
    axe.text(20, 103, r'$\leq 20°$', color=C_ECO, ha='center', fontsize=6.5, fontweight='bold')
    axe.set_xlabel(r'Rotation from AEP-opt $\Delta\theta$ (°)')
    axe.set_ylabel('Share of max exposure cut (%)')
    axe.set_xlim(0, 45); axe.set_ylim(0, 108)
    axe.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.4)
    axe.set_title('Small rotation captures most gain', fontsize=7.5)
    _sty(axe); _lab(axe, 'e')

    # ---- (f) Pareto frontier: AEP loss vs exposure reduction ----
    axf = fig.add_subplot(gs_b[0, 2])
    # 汇合所有场址 all budgets：x = aep_cost_pct, y = risk_reduction(%)
    on_all = ctx['on'][['farm_id', 'aep_cost_pct', 'risk_reduction']].copy()
    on_all['group'] = 'Onshore'
    on_all = on_all.rename(columns={'risk_reduction': 'rr'})
    to_all = ctx['to'][['farm_id', 'aep_cost_pct', 'risk_reduction_pct', 'source' if 'source' in ctx['to'].columns else 'farm_id']].copy()
    to_all = ctx['to'].merge(ctx['od'][['farm_id', 'source']], on='farm_id', how='left')
    to_all['group'] = to_all['source'].map({'VPTS': 'VPTS', 'Bauer_grid': 'Bauer'})
    to_all = to_all.rename(columns={'risk_reduction_pct': 'rr'})[['farm_id', 'aep_cost_pct', 'rr', 'group']]
    pa = pd.concat([on_all[['farm_id', 'aep_cost_pct', 'rr', 'group']], to_all], ignore_index=True)

    for grp, c, m, sz, alp in [('Onshore', C_LAND, 'o', 3, 0.10),
                               ('VPTS', C_VPTS, 'o', 10, 0.55),
                               ('Bauer', C_BAUER, 's', 10, 0.55)]:
        sub = pa[pa.group == grp]
        axf.scatter(sub.aep_cost_pct, sub.rr, s=sz, c=c, alpha=alp, edgecolors='none',
                    marker=m, rasterized=True, label=grp)
    # 1% AEP 竖线
    axf.axvspan(0, 1, color=C_ECO, alpha=0.10, zorder=0)
    axf.axvline(1, color=C_ECO, lw=0.8, ls='--')
    axf.text(1, 103, r'$\leq 1\%$ AEP', color=C_ECO, ha='center', fontsize=6.5, fontweight='bold')
    # median markers 每组在每个 budget
    for grp, c, m in [('Onshore', C_LAND, 'o'), ('VPTS', C_VPTS, 's'), ('Bauer', C_BAUER, 'D')]:
        sub = pa[pa.group == grp]
        for b in [0.005, 0.01, 0.02, 0.05]:
            bx = sub[(sub.aep_cost_pct <= b * 100 * 1.05) & (sub.aep_cost_pct > b * 100 * 0.95)]
            if not bx.empty:
                axf.scatter([bx.aep_cost_pct.median()], [bx.rr.median()], s=32, c=c,
                            marker=m, edgecolors='white', linewidths=0.7, zorder=5)
    axf.set_xlabel('AEP loss (%)')
    axf.set_ylabel('Exposure reduction (%)')
    axf.set_xlim(-0.15, 5.5); axf.set_ylim(0, 108)
    axf.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.2,
               scatterpoints=1)
    axf.set_title('Energy–exposure Pareto frontier', fontsize=7.5)
    _sty(axf); _lab(axf, 'f')

    # ---- (g) regional map with spring migration quiver ----
    if HAS_CARTOPY:
        axg = fig.add_subplot(gs_c[0, 0], projection=ccrs.PlateCarree())
        add_basemap(axg, [-12, 20, 39, 60])
        # 迁徙方向场（抽稀）
        gsub = grid[(grid.row % 2 == 0) & (grid.col % 2 == 0)]
        u = np.sin(np.radians(gsub.spring_dir.values))
        v = np.cos(np.radians(gsub.spring_dir.values))
        q = axg.quiver(gsub.lon.values, gsub.lat.values, u, v, gsub.spring_conc.values,
                       cmap=CMAP_CONC, scale=45, width=0.0022, headwidth=2.5,
                       headlength=2.8, alpha=0.65,
                       transform=ccrs.PlateCarree(), zorder=3)
        q.set_clim(0.55, 0.9)
        # 陆上
        axg.scatter(on1.centroid_lon, on1.centroid_lat, s=0.5, c=C_LAND, alpha=0.35,
                    edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)
        # 海上
        axg.scatter(vpts.centroid_lon, vpts.centroid_lat, s=14, c=C_VPTS,
                    edgecolors='white', linewidths=0.3, transform=ccrs.PlateCarree(),
                    zorder=5)
        axg.scatter(bauer.centroid_lon, bauer.centroid_lat, s=14, c=C_BAUER, marker='s',
                    edgecolors='white', linewidths=0.3, transform=ccrs.PlateCarree(),
                    zorder=5)
        for sn, (la, lo) in RADAR_LOC.items():
            axg.scatter(lo, la, marker='^', s=22, c='#C0392B', edgecolors='white',
                        linewidths=0.35, transform=ccrs.PlateCarree(), zorder=6)
        # 北海放大框
        axg.plot([-3.5, 9.5, 9.5, -3.5, -3.5], [49.5, 49.5, 56.5, 56.5, 49.5],
                 color='#C0392B', lw=0.7, ls='-', transform=ccrs.PlateCarree(), zorder=7)
        axg.set_title('Study region — spring migration & farms (n$_{on}$=4,191, n$_{VPTS}$=29, n$_{Bauer}$=26)',
                      fontsize=7.5)
        _fig_lab(fig, 'g', 0.035, 0.38)
        # colorbar for quiver
        cax = make_axes_locatable(axg).append_axes('right', size='2.5%', pad=0.04,
                                                    axes_class=plt.Axes)
        cb = fig.colorbar(q, cax=cax); cb.set_label('Direction concentration', fontsize=6.2)
        cb.set_ticks([0.6, 0.7, 0.8, 0.9]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)
        # legend
        leg = [
            Line2D([], [], marker='o', ls='', color=C_LAND, markersize=3.5, label='Onshore'),
            Line2D([], [], marker='o', ls='', color=C_VPTS, markersize=4, label='Offshore VPTS'),
            Line2D([], [], marker='s', ls='', color=C_BAUER, markersize=4, label='Offshore Bauer'),
            Line2D([], [], marker='^', ls='', color='#C0392B', markersize=4.5, label='Radar'),
        ]
        axg.legend(handles=leg, loc='lower left', fontsize=5.6, frameon=True, framealpha=0.85,
                   edgecolor='none', handletextpad=0.3, borderpad=0.3, labelspacing=0.2)

    # ---- (h) Δθ50 hex-bin spatial map (onshore) ----
    if HAS_CARTOPY:
        axh = fig.add_subplot(gs_c[0, 1], projection=ccrs.PlateCarree())
        ext = [on_df.centroid_lon.min() - 0.8, on_df.centroid_lon.max() + 0.8,
               on_df.centroid_lat.min() - 0.8, on_df.centroid_lat.max() + 0.8]
        add_basemap(axh, ext)
        sub = on_df.dropna(subset=['d50']).copy()
        hb = axh.hexbin(sub.centroid_lon, sub.centroid_lat, C=sub.d50, cmap='magma',
                        gridsize=28, mincnt=1, vmin=0, vmax=45, edgecolors='none',
                        reduce_C_function=np.median, transform=ccrs.PlateCarree(), zorder=3)
        # 海上叠加
        offm = _off_metric(od, pd.concat([vp_df, ba_df]), 'd50').dropna(subset=['d50'])
        axh.scatter(offm.centroid_lon, offm.centroid_lat, c=offm.d50, cmap='magma',
                    s=28, edgecolors='white', linewidths=0.5, vmin=0, vmax=45,
                    transform=ccrs.PlateCarree(), zorder=4)
        axh.set_title(r'Rotation for half the gain ($\Delta\theta_{50}$)', fontsize=7.5)
        _fig_lab(fig, 'h', 0.585, 0.38)
        cax = make_axes_locatable(axh).append_axes('right', size='3.5%', pad=0.05,
                                                    axes_class=plt.Axes)
        cb = fig.colorbar(hb, cax=cax); cb.set_label(r'Median $\Delta\theta_{50}$ (°)', fontsize=6.2)
        cb.set_ticks([0, 15, 30, 45]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    pdf, png = savefig(fig, 'fig1_mechanism')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 2 — Directional structure & cross-source robustness
# =====================================================================
def fig2_directional(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']; grid = ctx['grid']; od = ctx['od']

    fig = plt.figure(figsize=(W_DOUBLE, 5.4))
    gs_top = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.94, bottom=0.52,
                              wspace=0.42, width_ratios=[0.95, 1.20, 1.10])
    gs_bot = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.44, bottom=0.09,
                              wspace=0.42, width_ratios=[1.0, 1.25, 1.0])

    # (a) rose spring/autumn
    axa = fig.add_subplot(gs_top[0, 0], projection='polar')
    gdir = grid.dropna(subset=['spring_dir', 'autumn_dir'])
    bins = np.arange(0, 361, 15)
    sp_hist, _ = np.histogram(gdir.spring_dir.values, bins=bins)
    au_hist, _ = np.histogram(gdir.autumn_dir.values, bins=bins)
    tcent = np.radians((bins[:-1] + bins[1:]) / 2)
    axa.set_theta_zero_location('N'); axa.set_theta_direction(-1)
    axa.bar(tcent, sp_hist, width=np.radians(15), color=C_SPRING, alpha=0.75,
            edgecolor='white', linewidth=0.3)
    axa.bar(tcent, au_hist, width=np.radians(15), color=C_AUTUMN, alpha=0.6,
            edgecolor='white', linewidth=0.3)
    # 向量均值箭头
    def _mean(x):
        s = np.sin(np.radians(x)); c = np.cos(np.radians(x))
        return np.degrees(np.arctan2(s.mean(), c.mean())) % 360
    sp_mean = _mean(gdir.spring_dir.values); au_mean = _mean(gdir.autumn_dir.values)
    axa.annotate('', xy=(np.radians(sp_mean), max(sp_hist) * 0.9), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=1.4, mutation_scale=10))
    axa.annotate('', xy=(np.radians(au_mean), max(au_hist) * 0.9), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='-|>', color=C_AUTUMN, lw=1.4, mutation_scale=10))
    axa.set_yticks([])
    axa.tick_params(labelsize=6, pad=0.5)
    axa.set_title('Migration direction rose\n(2,025 grid cells)', fontsize=7.5, pad=8)
    axa.legend([Line2D([], [], color=C_SPRING, lw=4, alpha=0.75),
                Line2D([], [], color=C_AUTUMN, lw=4, alpha=0.6)],
               ['Spring', 'Autumn'], loc='upper right', bbox_to_anchor=(1.15, 1.10),
               fontsize=6, frameon=False, handlelength=1.2)
    _fig_lab(fig, 'a', 0.045, 0.94)

    # (b) ridgeline: normalized exposure vs orientation for representative farms per group
    axb = fig.add_subplot(gs_top[0, 1])
    # 抽 12 条陆上（分位数）、8 条 VPTS、8 条 Bauer 的归一化 E(θ)
    def _samples(df, k, on_flag):
        if on_flag:
            ord_df = df.dropna(subset=['d_full']).sort_values('d_full')
            idx = np.linspace(0, len(ord_df) - 1, k).astype(int)
            sel = ord_df.iloc[idx]
        else:
            sel = df.sort_values('d_full')
            idx = np.linspace(0, len(sel) - 1, min(k, len(sel))).astype(int)
            sel = sel.iloc[idx]
        out = []
        for _, r in sel.iterrows():
            if on_flag:
                row = on1[on1.farm_id == r.farm_id].iloc[0]
                E = on_Evec(row.spring_dir, row.autumn_dir, row.spring_conc, row.autumn_conc)
            else:
                E = off_Evec(r.farm_id, cur)
            E = E / max(E.max(), 1e-9)
            out.append(E)
        return out

    n_show = 10
    lanes = [(_samples(on_df, n_show, True), C_LAND, 'Onshore'),
             (_samples(vp_df, min(n_show, len(vp_df)), False), C_VPTS, 'VPTS'),
             (_samples(ba_df, min(n_show, len(ba_df)), False), C_BAUER, 'Bauer')]
    # 每组做 y 偏移的 curve stack
    offset_between = 1.0
    within = 0.85
    ymax = 0
    yticks = []; yticklabels = []
    for gi, (curves, c, lab) in enumerate(lanes):
        base = gi * offset_between
        for i, E in enumerate(curves):
            y = base + within * (i / max(len(curves) - 1, 1))
            axb.fill_between(THETAS, y, y + within * 0.9 * E, color=c, alpha=0.20,
                             linewidth=0)
            axb.plot(THETAS, y + within * 0.9 * E, color=c, lw=0.5, alpha=0.85)
        yticks.append(base + within * 0.5); yticklabels.append(lab)
        ymax = max(ymax, base + within * 1.1)
    axb.set_yticks(yticks); axb.set_yticklabels(yticklabels)
    axb.set_xlim(0, 180); axb.set_ylim(-0.1, ymax)
    axb.set_xticks([0, 45, 90, 135, 180])
    axb.set_xlabel(r'Array orientation $\theta$ (°)')
    axb.set_title('Family of exposure curves (representative farms)', fontsize=7.5)
    _sty(axb); _lab(axb, 'b', x=-0.10, y=1.05)

    # (c) sensitivity jitter + violin (rel by group)
    axc = fig.add_subplot(gs_top[0, 2])
    dfa = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'rel': np.concatenate([on_df.rel.values * 100, vp_df.rel.values * 100,
                               ba_df.rel.values * 100]),
    })
    sns.violinplot(data=dfa, x='Group', y='rel', ax=axc, order=GROUP_ORDER,
                   hue='Group', palette=GROUP_COLORS, inner=None, linewidth=0.4,
                   saturation=0.95, cut=0, legend=False)
    # jitter dots
    rng = np.random.default_rng(0)
    for i, (grp, c) in enumerate(zip(GROUP_ORDER, [C_LAND, C_VPTS, C_BAUER])):
        vals = dfa[dfa.Group == grp].rel.values
        x = i + rng.uniform(-0.14, 0.14, len(vals))
        axc.scatter(x, vals, s=1.6 if grp == 'Onshore' else 5, c=c, alpha=0.35,
                    edgecolors='none', zorder=3, rasterized=(grp == 'Onshore'))
    for i, (grp, df) in enumerate(zip(GROUP_ORDER, [on_df, vp_df, ba_df])):
        m = df.rel.median() * 100
        axc.hlines(m, i - 0.34, i + 0.34, color='black', lw=1.1, zorder=5)
        axc.text(i, m + 0.5, f'{m:.1f}%', ha='center', va='bottom', fontsize=6.8,
                 fontweight='bold')
    axc.set_ylim(85, 102.5)
    axc.set_xlabel('')
    axc.set_ylabel('Relative exposure change (%)')
    axc.set_title('Sensitivity amplitude (per farm)', fontsize=7.5)
    _sty(axc); _lab(axc, 'c')

    # (d) cumulative CDF of rel with bootstrap CI band per group
    axd = fig.add_subplot(gs_bot[0, 0])
    rng = np.random.default_rng(1)
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        v = np.sort(df.rel.values * 100)
        cdf = np.arange(1, len(v) + 1) / len(v)
        # bootstrap CI
        boot = []
        n = len(v); reps = 200
        thr_axis = np.arange(85, 100.1, 0.5)
        for _ in range(reps):
            s = rng.choice(v, size=n, replace=True)
            boot.append([(s >= t).mean() for t in thr_axis])
        boot = np.array(boot) * 100
        q_lo = np.percentile(boot, 2.5, axis=0); q_hi = np.percentile(boot, 97.5, axis=0)
        med = [(v >= t).mean() * 100 for t in thr_axis]
        axd.fill_between(thr_axis, q_lo, q_hi, color=c, alpha=0.15, linewidth=0)
        axd.plot(thr_axis, med, color=c, lw=1.4, label=lab)
    axd.axhline(100, color='#888888', lw=0.5, ls=':')
    axd.set_xlim(85, 100); axd.set_ylim(0, 106)
    axd.set_xlabel('Relative-change threshold (%)')
    axd.set_ylabel('Share of farms above (%)')
    axd.set_title('Universality (bootstrap 95% CI)', fontsize=7.5)
    axd.legend(loc='lower left', fontsize=6, frameon=False, handlelength=1.4)
    _sty(axd); _lab(axd, 'd')

    # (e) hex-bin rel across space (onshore)
    if HAS_CARTOPY:
        axe = fig.add_subplot(gs_bot[0, 1], projection=ccrs.PlateCarree())
        ext = [on_df.centroid_lon.min() - 0.8, on_df.centroid_lon.max() + 0.8,
               on_df.centroid_lat.min() - 0.8, on_df.centroid_lat.max() + 0.8]
        add_basemap(axe, ext)
        sub = on_df.dropna(subset=['rel'])
        hb = axe.hexbin(sub.centroid_lon, sub.centroid_lat, C=sub.rel * 100,
                        cmap=CMAP_RISK, gridsize=28, mincnt=1, vmin=85, vmax=100,
                        edgecolors='none', reduce_C_function=np.median,
                        transform=ccrs.PlateCarree(), zorder=3)
        offm = _off_metric(od, pd.concat([vp_df, ba_df]), 'rel').dropna(subset=['rel'])
        axe.scatter(offm.centroid_lon, offm.centroid_lat, c=offm.rel * 100,
                    cmap=CMAP_RISK, s=26, edgecolors='white', linewidths=0.5,
                    vmin=85, vmax=100, transform=ccrs.PlateCarree(), zorder=4)
        axe.set_title('Sensitivity across space', fontsize=7.5)
        _fig_lab(fig, 'e', 0.36, 0.44)
        cax = make_axes_locatable(axe).append_axes('right', size='3%', pad=0.05,
                                                    axes_class=plt.Axes)
        cb = fig.colorbar(hb, cax=cax); cb.set_label('Median rel. change (%)', fontsize=6.2)
        cb.set_ticks([85, 90, 95, 100]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    # (f) mechanism: direction concentration vs sensitivity
    axf = fig.add_subplot(gs_bot[0, 2])
    # 陆上：每场 spring_conc（来自 on1）与 rel
    on1_slim = on1[['farm_id', 'spring_conc']].drop_duplicates(subset='farm_id')
    on_m = on_df.merge(on1_slim, on='farm_id', how='left').dropna(subset=['spring_conc', 'rel'])
    axf.scatter(on_m.spring_conc, on_m.rel * 100, s=1.6, c=C_LAND, alpha=0.20,
                edgecolors='none', rasterized=True)
    # binned median
    q = np.linspace(on_m.spring_conc.min(), on_m.spring_conc.max(), 12)
    xs = 0.5 * (q[:-1] + q[1:])
    med = []; lo = []; hi = []
    for i in range(len(xs)):
        s = on_m[(on_m.spring_conc >= q[i]) & (on_m.spring_conc < q[i + 1])]
        if len(s) < 5:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(s.rel.median() * 100)
        lo.append(s.rel.quantile(0.25) * 100); hi.append(s.rel.quantile(0.75) * 100)
    axf.fill_between(xs, lo, hi, color=C_LAND, alpha=0.25, linewidth=0)
    axf.plot(xs, med, color=C_LAND, lw=1.4, label='Onshore (binned median±IQR)')
    axf.set_xlabel('Directional concentration (spring, grid)')
    axf.set_ylabel('Relative exposure change (%)')
    axf.set_ylim(85, 102)
    axf.set_title('Mechanism: concentration ⇒ sensitivity', fontsize=7.5)
    axf.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.2)
    _sty(axf); _lab(axf, 'f')

    pdf, png = savefig(fig, 'fig2_directional')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 3 — Systematic misalignment (double col, 2 rows)
# =====================================================================
def fig3_misalign(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']; on_aep = ctx['on_aep']

    fig = plt.figure(figsize=(W_DOUBLE, 6.4))
    gs_top = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.955, bottom=0.55,
                              wspace=0.30, width_ratios=[1.3, 1.0])
    gs_bot = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.47, bottom=0.06,
                              wspace=0.40, width_ratios=[0.9, 1.05, 1.15])

    # (a) hex-bin scatter θ_min vs θ_econ
    axa = fig.add_subplot(gs_top[0, 0])
    all_df = pd.concat([on_df.assign(grp='Onshore'),
                        vp_df.assign(grp='VPTS'), ba_df.assign(grp='Bauer')])
    x = all_df.theta_econ.values + np.random.uniform(-1.5, 1.5, len(all_df))
    y = all_df.th_min.values + np.random.uniform(-1.5, 1.5, len(all_df))
    hb = axa.hexbin(x, y, gridsize=42, cmap='cividis_r', mincnt=1, edgecolors='none',
                    bins='log')
    # 对角线
    axa.plot([0, 180], [0, 180], color='#E74C3C', ls='--', lw=1.0, zorder=3, label='aligned')
    # ±45° 参考带
    for off in (45, -45):
        y0 = np.clip(np.array([0, 180]) + off, 0, 180)
        axa.plot([0, 180], y0, color='#888888', ls=':', lw=0.5, zorder=2)
    # 三组 median 大点
    for grp, c, m in [('Onshore', C_LAND, 'o'), ('VPTS', C_VPTS, 's'), ('Bauer', C_BAUER, 'D')]:
        sub = all_df[all_df.grp == grp]
        axa.scatter([sub.theta_econ.median()], [sub.th_min.median()], s=55, c=c, marker=m,
                    edgecolors='white', linewidths=0.9, zorder=6, label=f'{grp} median')
    axa.set_xlabel(r'AEP-opt orientation $\theta_{\mathrm{econ}}$ (°)')
    axa.set_ylabel(r'Eco-opt orientation $\theta_{\mathrm{min}}$ (°)')
    axa.set_xlim(0, 180); axa.set_ylim(0, 180)
    axa.set_xticks([0, 45, 90, 135, 180]); axa.set_yticks([0, 45, 90, 135, 180])
    axa.set_aspect('equal')
    axa.set_title('AEP-opt vs eco-opt orientation (all 4,246 farms)', fontsize=7.5)
    axa.legend(loc='lower right', fontsize=5.8, frameon=True, framealpha=0.85,
               edgecolor='none', handletextpad=0.3, borderpad=0.25, labelspacing=0.2,
               scatterpoints=1)
    _sty(axa); _lab(axa, 'a', x=-0.10, y=1.03)
    cax = inset_axes(axa, width='32%', height='2.5%', loc='upper left',
                     bbox_to_anchor=(0.03, -0.05, 1, 1), bbox_transform=axa.transAxes)
    cb = fig.colorbar(hb, cax=cax, orientation='horizontal')
    cb.set_label('n farms per hex (log)', fontsize=5.5, labelpad=1)
    cb.ax.tick_params(width=0.4, length=1.2, labelsize=5.2, pad=1)

    # (b) dumbbell + top marginal histogram of d_full
    # 用嵌套 gridspec 拆分为上下：上=marginal, 下=dumbbell
    inner = gs_top[0, 1].subgridspec(2, 1, height_ratios=[0.35, 1.0], hspace=0.05)
    axb_top = fig.add_subplot(inner[0, 0])
    axb_bot = fig.add_subplot(inner[1, 0])
    bins = np.arange(0, 91, 4)
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        axb_top.hist(df.d_full.values, bins=bins, color=c, alpha=0.5, density=True,
                     edgecolor='white', linewidth=0.2, label=lab)
    axb_top.set_xlim(0, 90)
    axb_top.set_xticks([])
    axb_top.set_yticks([])
    for s in ('top', 'right', 'left'):
        axb_top.spines[s].set_visible(False)
    axb_top.set_title('AEP-opt vs eco-opt misalignment', fontsize=7.5, pad=4)
    axb_top.legend(loc='upper right', fontsize=5.8, frameon=False, handlelength=1.2,
                   ncol=3, columnspacing=0.7)
    _fig_lab(fig, 'b', 0.605, 0.955)

    # dumbbell rows (3)
    for i, (df, c, lab) in enumerate([(on_df, C_LAND, 'Onshore'),
                                       (vp_df, C_VPTS, 'VPTS'),
                                       (ba_df, C_BAUER, 'Bauer')]):
        y = 2 - i
        te = df.theta_econ.median(); tm = df.th_min.median()
        axb_bot.plot([te, tm], [y, y], color='#999999', lw=1.2, zorder=2)
        axb_bot.scatter([te], [y], s=46, marker='X', c=C_ECON, edgecolors='white',
                        linewidths=0.6, zorder=3)
        axb_bot.scatter([tm], [y], s=40, marker='o', c=C_ECO, edgecolors='white',
                        linewidths=0.5, zorder=3)
        axb_bot.text(-8, y, lab, ha='right', va='center', fontsize=6.5, color=c,
                     fontweight='bold')
        axb_bot.text((te + tm) / 2, y + 0.22, f'{abs(te - tm):.0f}°', ha='center',
                     va='bottom', fontsize=6.2, color='#333333')
    axb_bot.set_yticks([]); axb_bot.set_xlim(-15, 180)
    axb_bot.set_xticks([0, 45, 90, 135, 180])
    axb_bot.set_xlabel('Median orientation (°)')
    axb_bot.set_ylim(-0.6, 2.6)
    for s in ('top', 'right', 'left'):
        axb_bot.spines[s].set_visible(False)
    axb_bot.tick_params(labelsize=6.5, width=0.5, length=2.2, pad=1.5)
    axb_bot.legend([Line2D([], [], marker='X', ls='', color=C_ECON, markersize=6,
                           markeredgecolor='white', markeredgewidth=0.6),
                    Line2D([], [], marker='o', ls='', color=C_ECO, markersize=5,
                           markeredgecolor='white', markeredgewidth=0.5)],
                   ['AEP-opt', 'eco-opt'], loc='lower right', fontsize=5.8,
                   frameon=False, handletextpad=0.3)

    # (c) log-ratio bars E(θ_econ)/E_min
    axc = fig.add_subplot(gs_bot[0, 0])
    ratio = [(df.Ee / df.Emin.replace(0, np.nan)).median()
             for df in [on_df, vp_df, ba_df]]
    counts = [len(on_df), len(vp_df), len(ba_df)]
    iqr = [((df.Ee / df.Emin.replace(0, np.nan)).quantile(0.25),
            (df.Ee / df.Emin.replace(0, np.nan)).quantile(0.75))
           for df in [on_df, vp_df, ba_df]]
    xs = np.arange(3)
    bars = axc.bar(xs, ratio, width=0.55, color=[C_LAND, C_VPTS, C_BAUER], alpha=0.9,
                   edgecolor='white', linewidth=0.6, zorder=3)
    for xi, v, (lo, hi) in zip(xs, ratio, iqr):
        axc.errorbar([xi], [v], yerr=[[v - lo], [hi - v]], color='#333333', lw=0.7,
                     capsize=2, capthick=0.7)
    axc.set_yscale('log'); axc.set_ylim(1.5, 6000)
    for xi, v, n in zip(xs, ratio, counts):
        axc.text(xi, v * 2.2, f'{v:.0f}×', ha='center', fontsize=7.5, fontweight='bold')
        axc.text(xi, 3200, f'n={n}', ha='center', fontsize=5.8, color='#555555')
    axc.set_xticks(xs); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel(r'$E(\theta_{\mathrm{econ}})/E_{\mathrm{min}}$  (log, median±IQR)')
    axc.set_title('AEP-opt exposure over minimum', fontsize=7.5)
    _sty(axc); _lab(axc, 'c')

    # (d) wind-rose vs migration-rose overlay for one representative onshore farm
    axd = fig.add_subplot(gs_bot[0, 1], projection='polar')
    # 挑一个 spring_conc 分布中位的陆上场
    reprow = on1.iloc[(on1.spring_conc - on1.spring_conc.median()).abs().argsort()].iloc[0]
    sp = reprow.spring_dir; au = reprow.autumn_dir
    te = reprow.theta_econ
    axd.set_theta_zero_location('N'); axd.set_theta_direction(-1)
    # migration rose 用玫瑰扇区
    axd.bar([np.radians(sp)], [1.0], width=np.radians(30), color=C_SPRING, alpha=0.5,
            edgecolor='none', bottom=0.0)
    axd.bar([np.radians(au)], [0.7], width=np.radians(30), color=C_AUTUMN, alpha=0.5,
            edgecolor='none', bottom=0.0)
    # 涡轮阵列朝向 θ_econ 一条粗线（跨圆通过原点两端）
    for t in [te, te + 180]:
        axd.plot([np.radians(t), np.radians(t)], [0, 1.0], color=C_ECON, lw=2.0,
                 solid_capstyle='round')
    axd.set_rlim(0, 1.05); axd.set_rticks([]); axd.tick_params(labelsize=5.8, pad=0.5)
    axd.set_title(f'Rose vs array (farm {int(reprow.farm_id)})', fontsize=7.5, pad=8)
    axd.legend([Patch(facecolor=C_SPRING, alpha=0.5),
                Patch(facecolor=C_AUTUMN, alpha=0.5),
                Line2D([], [], color=C_ECON, lw=2)],
               ['Spring dir.', 'Autumn dir.', 'AEP-opt array'],
               loc='upper right', bbox_to_anchor=(1.30, 1.14), fontsize=5.8, frameon=False)
    _fig_lab(fig, 'd', 0.36, 0.47)

    # (e) case-study: worst- vs best-aligned onshore farm — E(θ) + AEP(θ) overlays
    axe = fig.add_subplot(gs_bot[0, 2])
    # 只在有 AEP 曲线的陆上场里挑
    aep_ids = set(on_aep.farm_id.unique())
    on_ca = on_df[on_df.farm_id.isin(aep_ids)].copy()
    worst = on_ca.sort_values('d_full', ascending=False).iloc[0]
    best = on_ca.sort_values('d_full').iloc[0]
    aep_cols = [c for c in on_aep.columns if c.startswith('aep_')]
    aep_angles = np.array([int(c.split('_')[1]) for c in aep_cols])

    def _twin(ax, r, colour, ls, lab):
        row1 = on1[on1.farm_id == r.farm_id].iloc[0]
        E = on_Evec(row1.spring_dir, row1.autumn_dir, row1.spring_conc, row1.autumn_conc)
        E = E / max(E.max(), 1e-9)
        ax.plot(THETAS, E, color=colour, lw=1.3, ls=ls, label=f'{lab}: exposure')
        aep_row = on_aep[on_aep.farm_id == r.farm_id].iloc[0]
        aep = np.array([aep_row[f'aep_{a:03d}'] for a in aep_angles])
        aep = aep / aep.max()
        ax2.plot(aep_angles, aep, color=colour, lw=1.0, ls=(0, (2, 2)) if ls == '--' else (0, (1, 1)),
                 alpha=0.85, label=f'{lab}: AEP')
    ax2 = axe.twinx()
    _twin(axe, worst, '#B03A2E', '-', f'worst d={worst.d_full:.0f}°')
    _twin(axe, best, '#1E8449', '--', f'best d={best.d_full:.0f}°')
    axe.set_xlim(0, 180); axe.set_ylim(0, 1.05)
    axe.set_xticks([0, 45, 90, 135, 180])
    axe.set_xlabel(r'Array orientation $\theta$ (°)')
    axe.set_ylabel('Normalized exposure')
    ax2.set_ylabel('Normalized AEP', color='#555555')
    ax2.tick_params(axis='y', colors='#555555', labelsize=6, width=0.4, length=1.8)
    ax2.spines['top'].set_visible(False)
    axe.set_title('Case-study farms: exposure vs AEP', fontsize=7.5)
    # 合并两轴 legend — 放右上角浅框，避免 cutoff
    h1, l1 = axe.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axe.legend(h1 + h2, l1 + l2, loc='upper right', bbox_to_anchor=(1.02, 0.99),
               fontsize=5.4, frameon=True, framealpha=0.85, edgecolor='none',
               ncol=1, handlelength=1.2, handletextpad=0.3, borderpad=0.25,
               labelspacing=0.15)
    _sty(axe); _lab(axe, 'e')

    pdf, png = savefig(fig, 'fig3_misalign')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 4 — Limited rotation captures most gain
# =====================================================================
def fig4_capture(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']

    fig = plt.figure(figsize=(W_DOUBLE, 6.0))
    gs_top = fig.add_gridspec(1, 2, left=0.06, right=0.985, top=0.94, bottom=0.53,
                              wspace=0.34, width_ratios=[1.15, 1.0])
    gs_bot = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.46, bottom=0.08,
                              wspace=0.42, width_ratios=[1.05, 1.05, 1.05])

    # (a) capture curves IQR (0..45°) + inset derivative
    axa = fig.add_subplot(gs_top[0, 0])
    thresholds_dth50 = {}
    for df, c, lab, is_on in [(on_df, C_LAND, 'Onshore', True),
                              (vp_df, C_VPTS, 'VPTS', False),
                              (ba_df, C_BAUER, 'Bauer', False)]:
        dths, mat = _capture_curves_iqr(df, on1, cur, is_on)
        q25 = np.nanpercentile(mat, 25, axis=0)
        q50 = np.nanpercentile(mat, 50, axis=0)
        q75 = np.nanpercentile(mat, 75, axis=0)
        axa.fill_between(dths, q25, q75, color=c, alpha=0.18, linewidth=0)
        axa.plot(dths, q50, color=c, lw=1.4, label=lab)
        # 记 Δθ50 median
        idx50 = np.argmax(q50 >= 50) if (q50 >= 50).any() else -1
        if idx50 > 0:
            axa.vlines(dths[idx50], 0, 50, color=c, ls=':', lw=0.7)
            axa.text(dths[idx50], -6, f'{dths[idx50]}°', color=c, ha='center',
                     fontsize=5.8)
        thresholds_dth50[lab] = dths[idx50] if idx50 > 0 else np.nan
    axa.axhline(50, color='#666666', ls=':', lw=0.6)
    axa.axhline(80, color='#666666', ls=':', lw=0.6)
    axa.text(45.5, 50, '50%', fontsize=6, color='#666666', ha='left', va='center')
    axa.text(45.5, 80, '80%', fontsize=6, color='#666666', ha='left', va='center')
    axa.axvspan(0, 20, color=C_ECO, alpha=0.08, zorder=0)
    axa.axvline(20, color=C_ECO, ls='--', lw=0.8)
    axa.text(20, 103, r'$\leq 20°$', color=C_ECO, ha='center', fontsize=6.5, fontweight='bold')
    axa.set_xlim(0, 45); axa.set_ylim(-10, 108)
    axa.set_yticks([0, 25, 50, 75, 100])
    axa.set_xlabel(r'Rotation from AEP-opt $\Delta\theta$ (°)')
    axa.set_ylabel('Share of max cut (%)')
    axa.set_title('Exposure falls steeply at small angles', fontsize=7.5)
    axa.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.4)
    _sty(axa); _lab(axa, 'a', x=-0.10, y=1.05)

    # inset: derivative (marginal gain per degree) — 置于右下 20% × 24%
    axa_in = inset_axes(axa, width='36%', height='26%', loc='lower right',
                        bbox_to_anchor=(-0.03, 0.10, 1, 1), bbox_transform=axa.transAxes,
                        borderpad=0)
    for df, c, lab, is_on in [(on_df, C_LAND, 'Onshore', True),
                              (vp_df, C_VPTS, 'VPTS', False),
                              (ba_df, C_BAUER, 'Bauer', False)]:
        dths, mat = _capture_curves_iqr(df, on1, cur, is_on)
        q50 = np.nanpercentile(mat, 50, axis=0)
        dq = np.gradient(q50, dths)
        axa_in.plot(dths, dq, color=c, lw=1.0)
    axa_in.axvline(20, color=C_ECO, ls='--', lw=0.5)
    axa_in.set_xlim(0, 45)
    axa_in.set_xlabel(r'$\Delta\theta$', fontsize=5.5, labelpad=0)
    axa_in.set_ylabel('%/°', fontsize=5.5, labelpad=1)
    axa_in.set_title('Marginal gain', fontsize=5.8, pad=1)
    axa_in.tick_params(labelsize=5.2, width=0.3, length=1.2, pad=0.5)
    axa_in.patch.set_alpha(0.85)
    for s in ('top', 'right'):
        axa_in.spines[s].set_visible(False)

    # (b) Δθ50 vs d_full — colour rel, size n_turbines
    axb = fig.add_subplot(gs_top[0, 1])
    on1_slim = on1[['farm_id', 'n_turbines']].drop_duplicates(subset='farm_id')
    on_m = on_df.merge(on1_slim, on='farm_id', how='left').dropna(subset=['d50', 'd_full'])
    on_m['n_turbines'] = on_m['n_turbines'].fillna(on_m['n_turbines'].median())
    sizes = 4 + 30 * (on_m['n_turbines'] - on_m['n_turbines'].min()) / \
        max(on_m['n_turbines'].max() - on_m['n_turbines'].min(), 1)
    sc = axb.scatter(on_m.d_full, on_m.d50, s=sizes, c=on_m.rel * 100, cmap=CMAP_RISK,
                     alpha=0.5, edgecolors='none', vmin=85, vmax=100, rasterized=True,
                     label='Onshore')
    for df, c, m, lab in [(vp_df, C_VPTS, 's', 'VPTS'), (ba_df, C_BAUER, 'D', 'Bauer')]:
        axb.scatter(df.d_full, df.d50, s=22, c=c, marker=m, edgecolors='white',
                    linewidths=0.5, label=lab, zorder=4)
    axb.plot([0, 90], [0, 90], color='#888888', lw=0.6, ls='--')
    axb.text(80, 82, '1:1', fontsize=5.8, color='#888888')
    axb.set_xlim(0, 92); axb.set_ylim(0, 92)
    axb.set_xlabel(r'Full misalignment $d_{\mathrm{full}}$ (°)')
    axb.set_ylabel(r'Rotation for half gain $\Delta\theta_{50}$ (°)')
    axb.set_title('Half-gain rotation vs. full misalignment', fontsize=7.5)
    axb.legend(loc='upper left', fontsize=5.8, frameon=False, handlelength=1.0,
               scatterpoints=1)
    _sty(axb); _lab(axb, 'b')
    cax = make_axes_locatable(axb).append_axes('right', size='3%', pad=0.05,
                                                axes_class=plt.Axes)
    cb = fig.colorbar(sc, cax=cax); cb.set_label('Rel. change (%)', fontsize=6.2)
    cb.set_ticks([85, 90, 95, 100]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    # (c) bee-swarm Δθ50 by country (top 6 countries; onshore only)
    axc = fig.add_subplot(gs_bot[0, 0])
    on_country = on_df.merge(on1[['farm_id', 'centroid_lat', 'centroid_lon']].drop_duplicates(),
                             on='farm_id', how='left', suffixes=('', '_dup'))
    if 'centroid_lat' in on_country.columns:
        on_country['country'] = on_country.apply(
            lambda r: _country_bbox(r.centroid_lat, r.centroid_lon), axis=1)
    top_c = on_country.country.value_counts().head(6).index.tolist()
    on_top = on_country[on_country.country.isin(top_c)]
    order = sorted(top_c, key=lambda k: on_top[on_top.country == k].d50.median() if not on_top[on_top.country == k].empty else 99)
    sns.violinplot(data=on_top, x='country', y='d50', ax=axc, order=order, cut=0,
                   inner=None, linewidth=0.4, color=C_LAND, saturation=0.9)
    # 加 bee-swarm 抽稀
    rng = np.random.default_rng(2)
    for i, cn in enumerate(order):
        vals = on_top[on_top.country == cn].d50.values
        if len(vals) > 350:
            vals = rng.choice(vals, 350, replace=False)
        x = i + rng.uniform(-0.16, 0.16, len(vals))
        axc.scatter(x, vals, s=2.0, c='#0B3D6E', alpha=0.35, edgecolors='none')
    for i, cn in enumerate(order):
        m = on_top[on_top.country == cn].d50.median()
        axc.hlines(m, i - 0.32, i + 0.32, color='black', lw=1.0, zorder=5)
        axc.text(i, m + 1.0, f'{m:.0f}°', ha='center', fontsize=6, fontweight='bold')
    axc.set_xlabel('')
    axc.set_ylabel(r'$\Delta\theta_{50}$ (°)')
    axc.set_ylim(0, 62)
    axc.set_title(r'$\Delta\theta_{50}$ by country (onshore)', fontsize=7.5)
    _sty(axc); _lab(axc, 'c')

    # (d) bee-swarm Δθ80 by group
    axd = fig.add_subplot(gs_bot[0, 1])
    dfd = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'd80': np.concatenate([on_df.d80.fillna(90).values, vp_df.d80.fillna(90).values,
                               ba_df.d80.fillna(90).values]),
    })
    sns.violinplot(data=dfd, x='Group', y='d80', ax=axd, order=GROUP_ORDER, cut=0,
                   inner=None, linewidth=0.4, hue='Group', palette=GROUP_COLORS,
                   saturation=0.9, legend=False)
    rng = np.random.default_rng(3)
    for i, (grp, c) in enumerate(zip(GROUP_ORDER, [C_LAND, C_VPTS, C_BAUER])):
        vals = dfd[dfd.Group == grp].d80.values
        if len(vals) > 300:
            vals = rng.choice(vals, 300, replace=False)
        x = i + rng.uniform(-0.16, 0.16, len(vals))
        axd.scatter(x, vals, s=2.0 if grp == 'Onshore' else 5, c=c, alpha=0.45,
                    edgecolors='none')
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.d80.median()
        axd.hlines(m, i - 0.32, i + 0.32, color='black', lw=1.0, zorder=5)
        axd.text(i, m + 1.5, f'{m:.0f}°', ha='center', fontsize=6.5, fontweight='bold')
    axd.set_xlabel('')
    axd.set_ylabel(r'$\Delta\theta_{80}$ (°)')
    axd.set_ylim(0, 62)
    axd.set_title(r'Rotation for 80% of max gain', fontsize=7.5)
    _sty(axd); _lab(axd, 'd')

    # (e) stacked-bar % farms hitting 50/80 within 20°/30°
    axe = fig.add_subplot(gs_bot[0, 2])
    labels = ['≥50% within 20°', '≥80% within 20°', '≥50% within 30°', '≥80% within 30°']
    metrics = ['frac20_ge50', 'frac20_ge80', 'frac30_ge50', 'frac30_ge80']
    data = []
    for df in [on_df, vp_df, ba_df]:
        row = [
            (df.frac20 * 100 >= 50).mean() * 100,
            (df.frac20 * 100 >= 80).mean() * 100,
            (df.frac30 * 100 >= 50).mean() * 100,
            (df.frac30 * 100 >= 80).mean() * 100,
        ]
        data.append(row)
    data = np.array(data)
    xs = np.arange(len(labels)); w = 0.26
    for i, (grp, c) in enumerate(zip(GROUP_ORDER, [C_LAND, C_VPTS, C_BAUER])):
        axe.bar(xs + (i - 1) * w, data[i], width=w, color=c, alpha=0.9,
                edgecolor='white', linewidth=0.5, label=grp)
        for xi, v in zip(xs + (i - 1) * w, data[i]):
            axe.text(xi, v + 1.4, f'{v:.0f}', ha='center', fontsize=5.8)
    axe.axhline(50, color='#888888', ls=':', lw=0.6)
    axe.set_xticks(xs)
    axe.set_xticklabels(labels, fontsize=5.8, rotation=15, ha='right')
    axe.set_ylabel('Share of farms (%)')
    axe.set_ylim(0, 108)
    axe.set_title('Threshold-crossing rate', fontsize=7.5)
    axe.legend(loc='upper right', fontsize=5.8, frameon=False, handlelength=1.0,
               ncol=3, columnspacing=0.6)
    _sty(axe); _lab(axe, 'e')

    pdf, png = savefig(fig, 'fig4_capture')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 5 — Energy–ecology trade-off (3 rows)
# =====================================================================
def fig5_tradeoff(on_df, vp_df, ba_df, ctx):
    fig = plt.figure(figsize=(W_DOUBLE, 9.0))

    gs_a = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.955, bottom=0.68,
                            wspace=0.32, width_ratios=[1.55, 1.0, 1.0])
    gs_b = fig.add_gridspec(1, 3, left=0.03, right=0.965, top=0.615, bottom=0.36,
                            wspace=0.10, width_ratios=[1.5, 0.9, 0.9])
    gs_c = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.29, bottom=0.05,
                            wspace=0.42, width_ratios=[0.9, 1.1, 0.9])

    # (a) LARGE Pareto scatter
    axa = fig.add_subplot(gs_a[0, 0])
    on_all = ctx['on'][['farm_id', 'budget', 'aep_cost_pct', 'risk_reduction']].copy()
    on_all['group'] = 'Onshore'
    on_all = on_all.rename(columns={'risk_reduction': 'rr'})
    to_all = ctx['to'].merge(ctx['od'][['farm_id', 'source']], on='farm_id', how='left').copy()
    to_all['group'] = to_all['source'].map({'VPTS': 'VPTS', 'Bauer_grid': 'Bauer'})
    to_all = to_all.rename(columns={'risk_reduction_pct': 'rr'})[
        ['farm_id', 'budget', 'aep_cost_pct', 'rr', 'group']]
    pa = pd.concat([on_all, to_all], ignore_index=True)
    for grp, c, m, sz, alp, order in [('Onshore', C_LAND, 'o', 3.5, 0.10, 1),
                                       ('VPTS', C_VPTS, 'o', 14, 0.55, 2),
                                       ('Bauer', C_BAUER, 's', 14, 0.55, 2)]:
        sub = pa[pa.group == grp]
        axa.scatter(sub.aep_cost_pct, sub.rr, s=sz, c=c, alpha=alp, edgecolors='none',
                    marker=m, rasterized=True, zorder=order, label=grp)
    axa.axvspan(0, 1, color=C_ECO, alpha=0.12, zorder=0)
    axa.axvline(1, color=C_ECO, ls='--', lw=0.9)
    axa.text(1, 104, r'$\leq 1\%$ AEP', color=C_ECO, ha='center', fontsize=6.8, fontweight='bold')
    # median markers per budget per group
    marker_bud = {0.005: '.', 0.01: 'o', 0.02: 's', 0.05: 'D'}
    for grp, c in [('Onshore', C_LAND), ('VPTS', C_VPTS), ('Bauer', C_BAUER)]:
        sub = pa[pa.group == grp]
        for b in [0.005, 0.01, 0.02, 0.05]:
            bx = sub[np.isclose(sub.budget, b)]
            if bx.empty:
                continue
            mx = bx.aep_cost_pct.median(); my = bx.rr.median()
            axa.scatter([mx], [my], s=55, c=c, marker=marker_bud[b],
                        edgecolors='white', linewidths=0.9, zorder=6)
            axa.annotate(f'{int(b*1000)/10}%', xy=(mx, my), xytext=(4, 4),
                         textcoords='offset points', fontsize=5.6, color=c, alpha=0.9)
    axa.set_xlim(-0.15, 5.5); axa.set_ylim(0, 110)
    axa.set_xlabel('AEP loss (%)'); axa.set_ylabel('Exposure reduction (%)')
    axa.set_title('Energy–exposure Pareto (4,246 farms × 4 budgets)', fontsize=7.5)
    axa.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.2,
               scatterpoints=1)
    _sty(axa); _lab(axa, 'a', x=-0.06, y=1.04)

    # (b) ridgeline RR by budget (onshore)
    axb = fig.add_subplot(gs_a[0, 1])
    series = [ctx['on'][ctx['on'].budget == b]['risk_reduction'].values
              for b in [0.005, 0.01, 0.02, 0.05]]
    _ridge(axb, series, ['0.5%', '1%', '2%', '5%'],
           ['#7FB3D5', '#5DADE2', '#2874A6', '#154360'],
           0, 105, 'Exposure reduction (%)  ·  Onshore')
    axb.set_title('Onshore RR by budget', fontsize=7.5)
    _lab(axb, 'b', x=-0.08, y=1.04)

    # (c) ridgeline RR by budget (offshore VPTS+Bauer combined)
    axc = fig.add_subplot(gs_a[0, 2])
    off_all = ctx['to']
    series = [off_all[off_all.budget == b]['risk_reduction_pct'].values
              for b in [0.005, 0.01, 0.02, 0.05]]
    _ridge(axc, series, ['0.5%', '1%', '2%', '5%'],
           ['#F5B7B1', '#EC7063', '#B03A2E', '#78281F'],
           0, 105, 'Exposure reduction (%)  ·  Offshore')
    axc.set_title('Offshore RR by budget', fontsize=7.5)
    _lab(axc, 'c', x=-0.08, y=1.04)

    # (d) onshore choropleth (hex-bin RR% @ 1% budget)
    on_b1 = _on_rr_at(ctx, 0.01); on_b5 = _on_rr_at(ctx, 0.05)
    off_b1 = _off_rr_at(ctx, 0.01)
    if HAS_CARTOPY:
        axd = fig.add_subplot(gs_b[0, 0], projection=ccrs.PlateCarree())
        ext = [on_b1.centroid_lon.min() - 0.8, on_b1.centroid_lon.max() + 0.8,
               on_b1.centroid_lat.min() - 0.8, on_b1.centroid_lat.max() + 0.8]
        add_basemap(axd, ext)
        hb = axd.hexbin(on_b1.centroid_lon, on_b1.centroid_lat, C=on_b1.risk_reduction,
                        cmap=CMAP_RISK, gridsize=30, mincnt=1, vmin=0, vmax=100,
                        edgecolors='none', reduce_C_function=np.median,
                        transform=ccrs.PlateCarree(), zorder=3)
        axd.set_title('Onshore RR @ 1% AEP (hex median)', fontsize=7.5, pad=2)
        _fig_lab(fig, 'd', 0.035, 0.605)
        cax = make_axes_locatable(axd).append_axes('right', size='3%', pad=0.05,
                                                    axes_class=plt.Axes)
        cb = fig.colorbar(hb, cax=cax); cb.set_label('RR (%)', fontsize=6.2)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

        # (e) VPTS offshore bubble
        axe = fig.add_subplot(gs_b[0, 1], projection=ccrs.PlateCarree())
        add_basemap(axe, [-3.5, 9.5, 49.5, 56.5])
        vp_b1 = off_b1[off_b1.source == 'VPTS']
        axe.scatter(vp_b1.centroid_lon, vp_b1.centroid_lat, c=vp_b1.risk_reduction_pct,
                    cmap=CMAP_RISK, s=52, edgecolors='white', linewidths=0.6,
                    vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=4)
        axe.set_title(f'VPTS @ 1% (n={len(vp_b1)})', fontsize=7.5, pad=2)
        _fig_lab(fig, 'e', 0.53, 0.605)

        # (f) Bauer offshore bubble
        axf = fig.add_subplot(gs_b[0, 2], projection=ccrs.PlateCarree())
        add_basemap(axf, [-3.5, 9.5, 49.5, 56.5])
        ba_b1 = off_b1[off_b1.source == 'Bauer_grid']
        axf.scatter(ba_b1.centroid_lon, ba_b1.centroid_lat, c=ba_b1.risk_reduction_pct,
                    cmap=CMAP_RISK, s=52, marker='s', edgecolors='white', linewidths=0.6,
                    vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=4)
        axf.set_title(f'Bauer @ 1% (n={len(ba_b1)})', fontsize=7.5, pad=2)
        _fig_lab(fig, 'f', 0.76, 0.605)

    # (g) median RR vs budget (three groups)
    axg = fig.add_subplot(gs_c[0, 0])
    _, rr = _median_rr_vs_budget(ctx)
    _, ae = _mean_aep_vs_budget(ctx)
    budgets = [0.005, 0.01, 0.02, 0.05]
    xp = np.arange(4)
    rr_arr = np.array([[rr[b][i] for b in budgets] for i in range(3)])
    ae_arr = np.array([[ae[b][i] for b in budgets] for i in range(3)])
    for i, (c, lab) in enumerate([(C_LAND, 'Onshore'), (C_VPTS, 'VPTS'), (C_BAUER, 'Bauer')]):
        axg.plot(xp, rr_arr[i], 'o-', color=c, lw=1.4, markersize=4,
                 markeredgecolor='white', markeredgewidth=0.5, label=lab)
        for xi, v in zip(xp, rr_arr[i]):
            axg.text(xi, v + 1.2, f'{v:.0f}', ha='center', fontsize=5.5, color=c)
    axg.set_xticks(xp); axg.set_xticklabels(['0.5%', '1%', '2%', '5%'])
    axg.set_xlabel('AEP budget'); axg.set_ylabel('Median exposure reduction (%)')
    axg.set_ylim(30, 108)
    axg.axvspan(-0.4, 1, color=C_ECO, alpha=0.08, zorder=0)
    axg.axvline(1, color=C_ECO, ls='--', lw=0.8)
    axg.text(1, 104, r'$\leq 1\%$ AEP', color=C_ECO, ha='center', fontsize=6.5, fontweight='bold')
    axg.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.2)
    axg.set_title('Median RR saturates fast', fontsize=7.5)
    _sty(axg); _lab(axg, 'g')

    # (h) saturation heatmap: rows=onshore farms (sorted by Δθ50), cols=budgets, colour=RR
    axh = fig.add_subplot(gs_c[0, 1])
    on = ctx['on']
    # 用 on_df 里 d50 排序作为 y 轴
    keys = on_df.dropna(subset=['d50']).sort_values('d50').farm_id.values
    if len(keys) > 400:
        # 每 ~10 farms 取 1（下采样）
        step = max(len(keys) // 400, 1)
        keys = keys[::step]
    piv = on.pivot_table(index='farm_id', columns='budget', values='risk_reduction',
                         aggfunc='first')
    piv = piv.reindex(keys)
    cols = [0.005, 0.01, 0.02, 0.05]
    mat = piv[cols].values
    im = axh.imshow(mat, aspect='auto', cmap=CMAP_RISK, vmin=0, vmax=100,
                    extent=(0, 4, len(keys), 0), interpolation='nearest',
                    rasterized=True)
    axh.set_xticks(np.arange(4) + 0.5); axh.set_xticklabels(['0.5%', '1%', '2%', '5%'])
    axh.set_xlabel('AEP budget')
    # 左侧 y 轴用 Δθ50 标注
    d50_sorted = on_df.set_index('farm_id').reindex(keys).d50.values
    yticks_idx = np.linspace(0, len(keys) - 1, 5).astype(int)
    axh.set_yticks(yticks_idx)
    axh.set_yticklabels([f'{d50_sorted[i]:.0f}°' for i in yticks_idx], fontsize=5.8)
    axh.set_ylabel(r'Onshore farms, sorted by $\Delta\theta_{50}$', fontsize=6.5, labelpad=1)
    axh.set_title('Saturation pattern per farm', fontsize=7.5)
    _lab(axh, 'h', x=-0.14, y=1.04)
    axh.tick_params(labelsize=5.8, width=0.4, length=1.8, pad=1)
    cax = make_axes_locatable(axh).append_axes('right', size='3%', pad=0.05, axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax); cb.set_label('RR (%)', fontsize=6.2)
    cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    # (i) mini heatmap of Table 2 (budget × group RR%) — 独立 subplot
    ax_tab = fig.add_subplot(gs_c[0, 2])
    hm = ax_tab.imshow(rr_arr, cmap=CMAP_RISK, vmin=30, vmax=100, aspect='auto')
    ax_tab.set_xticks(np.arange(4)); ax_tab.set_xticklabels(['0.5%', '1%', '2%', '5%'], fontsize=6.5)
    ax_tab.set_yticks(np.arange(3)); ax_tab.set_yticklabels(GROUP_ORDER, fontsize=6.5)
    ax_tab.set_xlabel('AEP budget')
    for i in range(3):
        for j in range(4):
            ax_tab.text(j, i, f'{rr_arr[i, j]:.0f}', ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white' if rr_arr[i, j] > 65 else '#222222')
    ax_tab.set_title('Median RR (%)  ·  budget × group', fontsize=7.5, pad=3)
    ax_tab.tick_params(width=0.4, length=1.8, pad=1)
    _lab(ax_tab, 'i', x=-0.14, y=1.04)
    cax = make_axes_locatable(ax_tab).append_axes('right', size='4%', pad=0.05, axes_class=plt.Axes)
    cb = fig.colorbar(hm, cax=cax); cb.set_label('RR (%)', fontsize=6.2)
    cb.set_ticks([30, 50, 75, 100]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    pdf, png = savefig(fig, 'fig5_tradeoff')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig S1 — Threat background (upgrade: radar rose ring + height ridge + month)
# =====================================================================
def figS1(ctx):
    # 复用 v2 的 figS1_threat（信息完整）；若原始 VPTS 数据不可用则跳过 refresh
    import shutil
    try:
        pdf, png = figS1_threat(ctx)
        for src in (pdf, png):
            dst = os.path.join(FIG6, 'figS1_threat' + os.path.splitext(src)[1])
            if src and os.path.exists(src):
                shutil.copy(src, dst)
    except Exception as e:
        print(f'  [warn] figS1 fallback: {e}')
        src = os.path.join(BASE, '..', 'outputs', 'figure', 'figS1_threat.png')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIG6, 'figS1_threat.png'))


# =====================================================================
# Fig S2 — Sensitivity space universality (hex-bin + KDE overlay + QQ)
# =====================================================================
def figS2(on_df, vp_df, ba_df, ctx):
    od = ctx['od']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']

    fig = plt.figure(figsize=(W_DOUBLE, 2.7))
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.985, top=0.85, bottom=0.18,
                          wspace=0.42, width_ratios=[1.35, 1.0, 1.0])

    if HAS_CARTOPY:
        axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ext = [on_df.centroid_lon.min() - 0.8, on_df.centroid_lon.max() + 0.8,
               on_df.centroid_lat.min() - 0.8, on_df.centroid_lat.max() + 0.8]
        add_basemap(axa, ext)
        sub = on_df.dropna(subset=['rel'])
        hb = axa.hexbin(sub.centroid_lon, sub.centroid_lat, C=sub.rel * 100,
                        cmap=CMAP_RISK, gridsize=32, mincnt=1, vmin=85, vmax=100,
                        edgecolors='none', reduce_C_function=np.median,
                        transform=ccrs.PlateCarree(), zorder=3)
        axa.scatter(vpts.centroid_lon, vpts.centroid_lat, s=22, c=C_VPTS,
                    edgecolors='white', linewidths=0.4, transform=ccrs.PlateCarree(),
                    zorder=4)
        axa.scatter(bauer.centroid_lon, bauer.centroid_lat, s=22, c=C_BAUER, marker='s',
                    edgecolors='white', linewidths=0.4, transform=ccrs.PlateCarree(),
                    zorder=4)
        for sn, (la, lo) in RADAR_LOC.items():
            axa.scatter(lo, la, marker='^', s=26, c='#C0392B', edgecolors='white',
                        linewidths=0.4, transform=ccrs.PlateCarree(), zorder=5)
        axa.set_title('Sensitivity across space (hex median)', fontsize=7.5)
        _fig_lab(fig, 'a', 0.03, 0.86)
        cb = fig.colorbar(hb, ax=axa, shrink=0.85, pad=0.02, fraction=0.045)
        cb.set_label('Rel. change (%)', fontsize=6.2)
        cb.set_ticks([85, 90, 95, 100]); cb.ax.tick_params(width=0.4, length=1.8, labelsize=5.8)

    # KDE overlay
    axb = fig.add_subplot(gs[0, 1])
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        sns.kdeplot(df.rel.values * 100, ax=axb, color=c, lw=1.2, fill=True, alpha=0.20,
                    clip=(85, 101), label=lab, common_norm=False)
    axb.set_xlim(85, 101)
    axb.set_xlabel('Relative change (%)')
    axb.set_ylabel('Density')
    axb.set_title('Rel-change distribution', fontsize=7.5)
    axb.legend(loc='upper left', fontsize=6, frameon=False, handlelength=1.2)
    _sty(axb); _lab(axb, 'b')

    # QQ plot: onshore vs offshore combined
    axc = fig.add_subplot(gs[0, 2])
    q = np.linspace(0.05, 0.95, 19)
    on_q = np.quantile(on_df.rel.values * 100, q)
    vp_q = np.quantile(vp_df.rel.values * 100, q)
    ba_q = np.quantile(ba_df.rel.values * 100, q)
    axc.plot([85, 100], [85, 100], color='#888888', lw=0.5, ls='--')
    axc.scatter(on_q, vp_q, s=15, c=C_VPTS, edgecolors='white', linewidths=0.4,
                label='VPTS vs Onshore', zorder=3)
    axc.scatter(on_q, ba_q, s=15, c=C_BAUER, marker='s', edgecolors='white',
                linewidths=0.4, label='Bauer vs Onshore', zorder=3)
    axc.set_xlim(85, 101); axc.set_ylim(85, 101)
    axc.set_xlabel('Onshore quantile (%)')
    axc.set_ylabel('Offshore quantile (%)')
    axc.set_title('Cross-source QQ', fontsize=7.5)
    axc.legend(loc='lower right', fontsize=5.8, frameon=False, handlelength=1.0,
               scatterpoints=1)
    _sty(axc); _lab(axc, 'c')

    pdf, png = savefig(fig, 'figS2_universality')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Main
# =====================================================================
def main():
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}, cartopy={HAS_CARTOPY}')

    print('Fig 1 (mechanism + region) ...'); fig1_mechanism(on_df, vp_df, ba_df, ctx)
    print('Fig 2 (directional structure) ...'); fig2_directional(on_df, vp_df, ba_df, ctx)
    print('Fig 3 (misalignment) ...'); fig3_misalign(on_df, vp_df, ba_df, ctx)
    print('Fig 4 (capture) ...'); fig4_capture(on_df, vp_df, ba_df, ctx)
    print('Fig 5 (tradeoff) ...'); fig5_tradeoff(on_df, vp_df, ba_df, ctx)
    print('Fig S2 (universality) ...'); figS2(on_df, vp_df, ba_df, ctx)
    print('Fig S1 (threat) ...'); figS1(ctx)

    print('\nDone. Outputs in figures_v6/:')
    for f in sorted(os.listdir(FIG6)):
        p = os.path.join(FIG6, f)
        print(f'  {f}  ({os.path.getsize(p)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()
