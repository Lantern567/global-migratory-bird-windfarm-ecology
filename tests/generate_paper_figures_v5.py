# -*- coding: utf-8 -*-
"""
generate_paper_figures_v5.py —— Nature Energy 版重排。

主要改动（相对 v4）：
  * 合并原 Fig1（框架示意）+ 原 Fig2（研究区/方向场/空间分布）→ 新 Fig1（3×3 双栏图）。
  * 主图 5 张：新 Fig1（合并）+ 新 Fig2（原 R1 敏感性）+ 新 Fig3（原 R2 错位）
             + 新 Fig4（原 R3 有限调整）+ 新 Fig5（原 R4 权衡 6 地图+2 折线）。
  * 版式：双栏 180 mm (7.09")、单栏 88 mm (3.46")、字号 7 pt/标题 8 pt。
  * 面板 label：黑体小写 a/b/c，左上外侧。
  * 去除文字重叠、加粗关键数字、色条统一放右侧、图内空白留足。

所有数字仍来自 figure_style.compute_metrics()，与 final_numbers.txt 一致。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN, C_ECON, C_ECO,
                          GROUP_COLORS, GROUP_ORDER, THETAS, CMAP_CONC, CMAP_RISK,
                          style_ax, panel_label, circ, on_Evec, off_Evec)

from generate_paper_figures_v2 import (add_basemap, RADAR_LOC, rep_farm, rep_curve,
                                       _pos, L_THECON, L_THMIN, L_ECON, L_EMIN,
                                       L_DTH50, L_DTH80, L_LEQ, L_THETA,
                                       HAS_CARTOPY, figS1_threat)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
FIG5 = os.path.join(BASE, '..', 'figures_v5')
os.makedirs(FIG5, exist_ok=True)

# ---------------------------------------------------------------------------
# Nature Energy 尺寸（英寸）：单栏 3.46 / 1.5 栏 4.72 / 双栏 7.09；最大高 9.72
W_SINGLE, W_ONEHALF, W_DOUBLE, H_MAX = 3.46, 4.72, 7.09, 9.72

# 局部 rcParams 覆盖（Nature Energy 偏 7pt 正文 / 8pt 标题）
plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.labelpad': 2.0,
    'axes.titlepad': 3.0,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.2,
})

# 边距与刻度
def _apply_style_all(ax):
    style_ax(ax)
    ax.tick_params(labelsize=6.5, width=0.5, length=2.2, pad=1.5)


def _label(ax, txt, x=-0.22, y=1.14):
    """Nature Energy 面板标签：黑体小写，放到 title 上方外侧，避免与 title 重叠。"""
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='left', color='black')


def savefig(fig, name):
    pdf = os.path.join(FIG5, name + '.pdf')
    png = os.path.join(FIG5, name + '.png')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=450)
    return pdf, png


# ---------------------------------------------------------------------------
# 通用取数
# ---------------------------------------------------------------------------
def _on_rr_at(ctx, budget):
    sub = ctx['on'][ctx['on'].budget == budget]
    return sub[['farm_id', 'centroid_lat', 'centroid_lon', 'risk_reduction']].copy()


def _off_rr_at(ctx, budget):
    od = ctx['od'][['farm_id', 'centroid_lat', 'centroid_lon', 'source']]
    sub = ctx['to'][ctx['to'].budget == budget][['farm_id', 'risk_reduction_pct']]
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


# =====================================================================
# Fig 1 — 合并版：框架 + 研究区 + 方向场 + 空间分布（双栏，3 行）
# =====================================================================
def fig1_framework_region(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; od = ctx['od']; grid = ctx['grid']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']
    NORTHSEA = [-3.5, 9.5, 49.5, 56.5]

    # 3 行：第 1 行框架（4 概念面板）、第 2 行研究区+方向场 2 大图、第 3 行 avoid+Δθ50 2 大图
    fig = plt.figure(figsize=(W_DOUBLE, 7.55))

    # 顶部框架带：4 列（几何 | 冲突条 | 削减 | 权衡）
    gs_top = fig.add_gridspec(1, 4, left=0.065, right=0.985, top=0.940, bottom=0.72,
                              wspace=0.50)
    # 中部：研究区 + 方向场 2 张
    gs_mid = fig.add_gridspec(1, 2, left=0.045, right=0.985, top=0.655, bottom=0.375,
                              wspace=0.08)
    # 底部：avoid + Δθ50 2 张
    gs_bot = fig.add_gridspec(1, 2, left=0.045, right=0.985, top=0.335, bottom=0.055,
                              wspace=0.08)

    # ---------------- (a) 朝向几何示意 ----------------
    axa = fig.add_subplot(gs_top[0, 0])
    axa.set_xlim(0, 10); axa.set_ylim(-1.2, 10); axa.set_aspect('equal')
    axa.axis('off')

    def _farm(x0, y0, w, h, rows_vertical):
        axa.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor='#444444', lw=0.8))
        if rows_vertical:
            xs = np.linspace(x0 + 0.9, x0 + w - 0.9, 3)
            for xi in xs:
                ys = np.linspace(y0 + 0.5, y0 + h - 0.5, 6)
                axa.plot([xi] * len(ys), ys, 'o', color=C_ECON, markersize=2.6, zorder=3)
        else:
            ys = np.linspace(y0 + 0.9, y0 + h - 0.9, 3)
            for yi in ys:
                xs = np.linspace(x0 + 0.5, x0 + w - 0.5, 6)
                axa.plot(xs, [yi] * len(xs), 'o', color=C_ECON, markersize=2.6, zorder=3)

    _farm(0.4, 0.6, 3.4, 8.6, rows_vertical=True)
    axa.annotate('', xy=(2.1, 9.7), xytext=(2.1, 0.3),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=1.8, mutation_scale=14))
    _farm(6.2, 0.6, 3.4, 8.6, rows_vertical=False)
    axa.annotate('', xy=(7.9, 9.7), xytext=(7.9, 0.3),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=1.8, mutation_scale=14))
    axa.text(2.1, 9.4, 'migration', fontsize=6, color=C_SPRING, ha='center', va='top')
    axa.text(2.1, -0.5, 'parallel', fontsize=6.5, ha='center', va='top', fontweight='bold')
    axa.text(2.1, -1.05, 'low exposure', fontsize=6, ha='center', va='top', color='#555555')
    axa.text(7.9, -0.5, 'perpendicular', fontsize=6.5, ha='center', va='top', fontweight='bold')
    axa.text(7.9, -1.05, 'high exposure', fontsize=6, ha='center', va='top', color='#555555')
    axa.set_title('Orientation sets geometry', fontsize=7.5)
    _label(axa, 'a', x=-0.02, y=1.10)

    # ---------------- (b) 冲突（两类最优错位 ≈ 48°） ----------------
    axb = fig.add_subplot(gs_top[0, 1])
    axb.axis('off')
    axb.set_xlim(0, 180); axb.set_ylim(-1.2, 10)
    y_axis, y_labels, y_arrow_t, y_names = 4.2, 3.0, 6.2, 7.6
    axb.plot([0, 180], [y_axis, y_axis], color='#444444', lw=1.0)
    for x in [0, 45, 90, 135, 180]:
        axb.plot([x, x], [y_axis - 0.25, y_axis + 0.25], color='#444444', lw=0.6)
        axb.text(x, y_labels, f'{x}°', fontsize=6, ha='center', color='#555555')
    te, tm = 45.0, 93.0
    axb.annotate('', xy=(tm, y_axis), xytext=(te, y_axis),
                 arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0, mutation_scale=10))
    axb.scatter([te], [y_axis], s=60, marker='x', c=C_ECON, zorder=5, linewidths=1.4)
    axb.scatter([tm], [y_axis], s=44, marker='o', c=C_ECO, zorder=5, edgecolors='white', linewidths=0.6)
    axb.text((te + tm) / 2, y_arrow_t, 'misalignment ≈ 48°', ha='center', fontsize=6.5, color='#333333')
    # 两个标签用连线指出，避免文本挤在一起
    axb.annotate('AEP-opt', xy=(te, y_axis + 0.4), xytext=(10, y_names),
                 color=C_ECON, fontsize=6.5, ha='left', fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color=C_ECON, lw=0.6))
    axb.annotate('eco-opt', xy=(tm, y_axis + 0.4), xytext=(130, y_names),
                 color=C_ECO, fontsize=6.5, ha='left', fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color=C_ECO, lw=0.6))
    axb.set_title('Energy and ecology diverge', fontsize=7.5)
    _label(axb, 'b', x=-0.02, y=1.10)

    # ---------------- (c) 削减：快速下降后饱和 ----------------
    axc = fig.add_subplot(gs_top[0, 2])
    dth = np.linspace(0, 90, 400)
    red = 1 - np.exp(-dth / 12.0)
    axc.plot(dth, red, color='#222222', lw=1.6)
    axc.fill_between(dth, 0, red, color='#222222', alpha=0.06)
    axc.axvspan(0, 20, color=C_ECO, alpha=0.10, zorder=0)
    axc.axvline(20, color=C_ECO, ls='--', lw=0.9)
    axc.text(20, 1.05, r'$\leq 20°$', color=C_ECO, fontsize=6.5, ha='center', fontweight='bold')
    axc.set_xlabel(r'Rotation from AEP-opt $\Delta\theta$ (°)', fontsize=6.5)
    axc.set_ylabel('Exposure reduction', fontsize=6.5)
    axc.set_xlim(0, 90); axc.set_ylim(0, 1.12)
    axc.set_yticks([0, 0.5, 1.0])
    axc.set_title('Small angle, most gain', fontsize=7.5)
    _apply_style_all(axc)
    _label(axc, 'c', x=-0.24, y=1.14)

    # ---------------- (d) 权衡 frontier（不对称）+ 结论 ----------------
    axd = fig.add_subplot(gs_top[0, 3])
    aep = np.linspace(0, 5, 400)
    gain = 1 - np.exp(-aep / 0.45)
    axd.plot(aep, gain, color='#222222', lw=1.6)
    axd.fill_between(aep, 0, gain, color='#222222', alpha=0.06)
    axd.axvspan(0, 1, color=C_ECO, alpha=0.10, zorder=0)
    axd.axvline(1, color=C_ECO, ls='--', lw=0.9)
    axd.text(1, 1.05, r'$\leq 1\%$ AEP', color=C_ECO, fontsize=6.5, ha='center', fontweight='bold')
    axd.annotate('~97% exposure cut\nonshore', xy=(1, 1 - np.exp(-1 / 0.45)),
                 xytext=(2.5, 0.38), fontsize=6.2, color='#333333',
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=0.7))
    axd.set_xlabel('AEP loss (%)', fontsize=6.5)
    axd.set_ylabel('Exposure reduction', fontsize=6.5)
    axd.set_xlim(0, 5); axd.set_ylim(0, 1.12)
    axd.set_yticks([0, 0.5, 1.0])
    axd.set_title('Asymmetric frontier', fontsize=7.5)
    _apply_style_all(axd)
    _label(axd, 'd', x=-0.24, y=1.14)

    # ---------------- (e) 研究区总览 ----------------
    if HAS_CARTOPY:
        axe = fig.add_subplot(gs_mid[0, 0], projection=ccrs.PlateCarree())
        add_basemap(axe, [-12, 20, 39, 59])
        gx = [grid.lon.min(), grid.lon.max(), grid.lon.max(), grid.lon.min(), grid.lon.min()]
        gy = [grid.lat.min(), grid.lat.min(), grid.lat.max(), grid.lat.max(), grid.lat.min()]
        axe.plot(gx, gy, color='#333333', lw=0.9, ls='--', transform=ccrs.PlateCarree(), zorder=2)
        axe.scatter(on1.centroid_lon, on1.centroid_lat, s=0.5, c=C_LAND, alpha=0.28,
                    edgecolors='none', transform=ccrs.PlateCarree(), zorder=3)
        axe.scatter(vpts.centroid_lon, vpts.centroid_lat, s=14, c=C_VPTS, edgecolors='white',
                    linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
        axe.scatter(bauer.centroid_lon, bauer.centroid_lat, s=14, c=C_BAUER, marker='s',
                    edgecolors='white', linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
        for sn, (la, lo) in RADAR_LOC.items():
            axe.scatter(lo, la, marker='^', s=26, c='#C0392B', edgecolors='white',
                        linewidths=0.35, transform=ccrs.PlateCarree(), zorder=5)
        # 北海放大框
        axe.plot([NORTHSEA[0], NORTHSEA[1], NORTHSEA[1], NORTHSEA[0], NORTHSEA[0]],
                 [NORTHSEA[2], NORTHSEA[2], NORTHSEA[3], NORTHSEA[3], NORTHSEA[2]],
                 color='#C0392B', lw=0.7, ls='-', transform=ccrs.PlateCarree(), zorder=6)
        axe.set_title('Study region  (n$_{on}$=4,191  n$_{VPTS}$=29  n$_{Bauer}$=26)', fontsize=7.5)
        _label(axe, 'e', x=-0.02, y=1.10)
        # 图例（右下角内嵌）
        leg = [
            Line2D([], [], marker='o', ls='', color=C_LAND, markersize=3.5, label='Onshore'),
            Line2D([], [], marker='o', ls='', color=C_VPTS, markersize=4, label='Offshore VPTS'),
            Line2D([], [], marker='s', ls='', color=C_BAUER, markersize=4, label='Offshore Bauer'),
            Line2D([], [], marker='^', ls='', color='#C0392B', markersize=4.5, label='Radar station'),
        ]
        axe.legend(handles=leg, loc='lower left', fontsize=6, frameon=True, framealpha=0.85,
                   edgecolor='none', handletextpad=0.3, borderpad=0.3, labelspacing=0.25)

    # ---------------- (f) 春季方向场 ----------------
    if HAS_CARTOPY:
        axf = fig.add_subplot(gs_mid[0, 1], projection=ccrs.PlateCarree())
        gsub = grid[(grid.row % 2 == 0) & (grid.col % 2 == 0)]
        extent = [grid.lon.min() - 0.5, grid.lon.max() + 0.5, grid.lat.min() - 0.5, grid.lat.max() + 0.5]
        add_basemap(axf, extent)
        u = np.sin(np.radians(gsub.spring_dir.values)); v = np.cos(np.radians(gsub.spring_dir.values))
        q = axf.quiver(gsub.lon.values, gsub.lat.values, u, v, gsub.spring_conc.values,
                       cmap=CMAP_CONC, scale=42, width=0.0028, headwidth=2.6, headlength=3.0,
                       transform=ccrs.PlateCarree(), zorder=3)
        q.set_clim(0.5, 0.9)
        axf.set_title('Spring migration direction field', fontsize=7.5)
        _label(axf, 'f', x=-0.02, y=1.10)
        cax = make_axes_locatable(axf).append_axes('right', size='3.5%', pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(q, cax=cax); cb.set_label('Concentration', fontsize=6.5)
        cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9]); cb.ax.tick_params(width=0.5, length=2.0, labelsize=6)

    # ---------------- (g) 可避免暴露空间分布 ----------------
    if HAS_CARTOPY:
        axg = fig.add_subplot(gs_bot[0, 0], projection=ccrs.PlateCarree())
        add_basemap(axg, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axg.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.avoid * 100,
                         cmap=CMAP_RISK, s=1.4, alpha=0.65, edgecolors='none',
                         vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=3)
        # 海上叠加大点
        off_all = pd.concat([vp_df, ba_df])
        offm = _off_metric(od, off_all, 'avoid')
        axg.scatter(offm.centroid_lon, offm.centroid_lat, c=offm.avoid * 100,
                    cmap=CMAP_RISK, s=26, edgecolors='white', linewidths=0.4,
                    vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=4)
        axg.set_title('Avoidable exposure', fontsize=7.5)
        _label(axg, 'g', x=-0.02, y=1.10)
        cax = make_axes_locatable(axg).append_axes('right', size='3.5%', pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label('Avoidable (%)', fontsize=6.5)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.5, length=2.0, labelsize=6)

    # ---------------- (h) Δθ50 空间分布 ----------------
    if HAS_CARTOPY:
        axh = fig.add_subplot(gs_bot[0, 1], projection=ccrs.PlateCarree())
        add_basemap(axh, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axh.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.d50.fillna(90),
                         cmap='viridis', s=1.4, alpha=0.65, edgecolors='none',
                         vmin=0, vmax=45, transform=ccrs.PlateCarree(), zorder=3)
        offm = _off_metric(od, pd.concat([vp_df, ba_df]), 'd50')
        axh.scatter(offm.centroid_lon, offm.centroid_lat, c=offm.d50.fillna(90),
                    cmap='viridis', s=26, edgecolors='white', linewidths=0.4,
                    vmin=0, vmax=45, transform=ccrs.PlateCarree(), zorder=4)
        axh.set_title('Rotation to 50% of max gain ($\\Delta\\theta_{50}$)', fontsize=7.5)
        _label(axh, 'h', x=-0.02, y=1.10)
        cax = make_axes_locatable(axh).append_axes('right', size='3.5%', pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label(r'$\Delta\theta_{50}$ (°)', fontsize=6.5)
        cb.set_ticks([0, 15, 30, 45]); cb.ax.tick_params(width=0.5, length=2.0, labelsize=6)

    pdf, png = savefig(fig, 'fig1_framework_region')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 2 — R1 敏感性（双栏，2×3）
# =====================================================================
def fig2_sensitivity(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']; grid = ctx['grid']
    gdir = grid.dropna(subset=['spring_dir', 'autumn_dir']).copy()

    fig = plt.figure(figsize=(W_DOUBLE, 4.5))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.90, bottom=0.09,
                          hspace=0.75, wspace=0.45)

    # (a) 玫瑰
    axa = fig.add_subplot(gs[0, 0], projection='polar')
    bins = np.arange(0, 361, 15)
    sp_hist, _ = np.histogram(gdir.spring_dir.values, bins=bins)
    au_hist, _ = np.histogram(gdir.autumn_dir.values, bins=bins)
    tcent = np.radians((bins[:-1] + bins[1:]) / 2)
    axa.set_theta_zero_location('N'); axa.set_theta_direction(-1)
    axa.bar(tcent, sp_hist, width=np.radians(15), color=C_SPRING, alpha=0.7,
            edgecolor='white', linewidth=0.3, zorder=2)
    axa.bar(tcent, au_hist, width=np.radians(15), color=C_AUTUMN, alpha=0.65,
            edgecolor='white', linewidth=0.3, zorder=2)
    axa.set_yticks([])
    axa.tick_params(labelsize=6, pad=0.5)
    axa.set_title('Migration direction (rose)', fontsize=8, pad=8)
    axa.legend([Line2D([], [], color=C_SPRING, lw=4, alpha=0.7),
                Line2D([], [], color=C_AUTUMN, lw=4, alpha=0.65)],
               ['Spring', 'Autumn'], loc='upper right', bbox_to_anchor=(1.16, 1.10),
               fontsize=6.5, frameon=False, handlelength=1.2)
    _label(axa, 'a', x=-0.14, y=1.14)

    # (b) 代表 E(θ)
    axb = fig.add_subplot(gs[0, 1])
    for df, c, lab in [(on_df, C_LAND, 'Onshore'),
                       (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        _, E, _ = rep_curve(df, on1, cur)
        axb.plot(THETAS, E, color=c, lw=1.4, alpha=0.9, label=lab)
    axb.set_xlabel(r'Array orientation $\theta$ (°)')
    axb.set_ylabel('Normalized exposure')
    axb.set_xlim(0, 180); axb.set_ylim(-0.02, 1.08)
    axb.set_xticks([0, 45, 90, 135, 180])
    axb.set_title('Exposure vs. orientation (rep. farm)', fontsize=8)
    axb.legend(loc='upper right', fontsize=6.2, frameon=False, ncol=1,
               handlelength=1.5, borderpad=0.2)
    _apply_style_all(axb); _label(axb, 'b')

    # (c) rel 中位分组柱
    axc = fig.add_subplot(gs[0, 2])
    x = np.arange(3); w = 0.55
    rel = [on_df.rel.median() * 100, vp_df.rel.median() * 100, ba_df.rel.median() * 100]
    bars = axc.bar(x, rel, w, color=[C_LAND, C_VPTS, C_BAUER], alpha=0.9,
                   edgecolor='white', linewidth=0.6)
    for xi, v in zip(x, rel):
        axc.text(xi, v + 1.5, f'{v:.1f}%', ha='center', fontsize=7, fontweight='bold')
    axc.set_xticks(x); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel('Median rel. change (%)'); axc.set_ylim(0, 110)
    axc.set_title('Sensitivity amplitude', fontsize=8)
    _apply_style_all(axc); _label(axc, 'c')

    # (d) 春季集中度
    axd = fig.add_subplot(gs[1, 0])
    sc = grid.spring_conc.dropna()
    axd.hist(sc.values, bins=25, color=C_SPRING, alpha=0.75, edgecolor='white', linewidth=0.4)
    axd.axvline(sc.median(), color='#111111', ls='--', lw=1.1)
    axd.text(sc.median() + 0.005, axd.get_ylim()[1] * 0.92,
             f'median\n{sc.median():.2f}', fontsize=6.5, color='#222222', va='top')
    axd.set_xlabel('Concentration'); axd.set_ylabel('n grid cells')
    axd.set_title('Spring directional concentration', fontsize=8)
    _apply_style_all(axd); _label(axd, 'd')

    # (e) 秋季集中度
    axe = fig.add_subplot(gs[1, 1])
    ac = grid.autumn_conc.dropna()
    axe.hist(ac.values, bins=25, color=C_AUTUMN, alpha=0.75, edgecolor='white', linewidth=0.4)
    axe.axvline(ac.median(), color='#111111', ls='--', lw=1.1)
    axe.text(ac.median() + 0.005, axe.get_ylim()[1] * 0.92,
             f'median\n{ac.median():.2f}', fontsize=6.5, color='#222222', va='top')
    axe.set_xlabel('Concentration'); axe.set_ylabel('n grid cells')
    axe.set_title('Autumn directional concentration', fontsize=8)
    _apply_style_all(axe); _label(axe, 'e')

    # (f) 普遍性 CDF
    axf = fig.add_subplot(gs[1, 2])
    thr = np.arange(90, 101, 1)
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        vals = [(df.rel.values * 100 >= t).mean() * 100 for t in thr]
        axf.plot(thr, vals, 'o-', color=c, lw=1.4, markersize=3.2, label=lab)
    axf.axhline(100, color='#999999', ls=':', lw=0.7)
    axf.set_xlabel('Sensitivity threshold (%)')
    axf.set_ylabel('Share of farms above (%)')
    axf.set_xlim(90, 100); axf.set_ylim(0, 106)
    axf.set_xticks([90, 92, 94, 96, 98, 100])
    axf.set_title('Universality of high sensitivity', fontsize=8)
    axf.legend(loc='lower left', fontsize=6.2, frameon=False, handlelength=1.5)
    _apply_style_all(axf); _label(axf, 'f')

    pdf, png = savefig(fig, 'fig2_sensitivity')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 3 — R2 misalignment（双栏，2×3）
# =====================================================================
def fig3_misalignment(on_df, vp_df, ba_df, ctx):
    fig = plt.figure(figsize=(W_DOUBLE, 4.5))
    gs = fig.add_gridspec(2, 3, left=0.065, right=0.985, top=0.90, bottom=0.09,
                          hspace=0.70, wspace=0.45)

    # (a) avoid 小提琴
    axa = fig.add_subplot(gs[0, 0])
    dfa = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Avoidable (%)': np.concatenate([on_df.avoid.values * 100, vp_df.avoid.values * 100,
                                         ba_df.avoid.values * 100]),
    })
    sns.violinplot(data=dfa, x='Group', y='Avoidable (%)', ax=axa,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.5, saturation=0.9, legend=False, cut=0)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.avoid.median() * 100
        axa.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.2, zorder=5)
        axa.text(i, m + 2.5, f'{m:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    axa.set_ylim(0, 115)
    axa.set_xlabel('')
    axa.set_ylabel('Avoidable exposure (%)')
    axa.set_title('Avoidable by re-orientation', fontsize=8)
    _apply_style_all(axa); _label(axa, 'a')

    # (b) misalignment 小提琴
    axb = fig.add_subplot(gs[0, 1])
    dff = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Misalignment (°)': np.concatenate([on_df.d_full.values, vp_df.d_full.values,
                                            ba_df.d_full.values]),
    })
    sns.violinplot(data=dff, x='Group', y='Misalignment (°)', ax=axb,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.5, saturation=0.9, legend=False, cut=0)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.d_full.median()
        axb.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.2, zorder=5)
        axb.text(i, m + 2.5, f'{m:.0f}°', ha='center', va='bottom', fontsize=7, fontweight='bold')
    axb.set_ylim(0, 95)
    axb.set_xlabel('')
    axb.set_ylabel('Misalignment (°)')
    axb.set_title('AEP-opt vs eco-opt misalignment', fontsize=8)
    _apply_style_all(axb); _label(axb, 'b')

    # (c) θ_min vs θ_econ 散点：陆上 4,191 点先用低 alpha 底、海上再叠加
    axc = fig.add_subplot(gs[0, 2])
    axc.plot([0, 180], [0, 180], color='#999999', ls='--', lw=1.0, zorder=1)
    axc.text(140, 158, 'aligned', fontsize=6.5, color='#777777', ha='left', va='bottom')
    # 陆上用极低 alpha 的小散点做密度感（4,191 场址会集中在几列 θ_econ 上）
    axc.scatter(on_df.theta_econ + np.random.uniform(-1.5, 1.5, len(on_df)),
                on_df.th_min + np.random.uniform(-1.5, 1.5, len(on_df)),
                s=3, c=C_LAND, alpha=0.15, edgecolors='none', rasterized=True, label='Onshore')
    axc.scatter(vp_df.theta_econ, vp_df.th_min, s=16, c=C_VPTS, alpha=0.85,
                edgecolors='white', linewidths=0.4, label='VPTS')
    axc.scatter(ba_df.theta_econ, ba_df.th_min, s=16, c=C_BAUER, alpha=0.85,
                marker='s', edgecolors='white', linewidths=0.4, label='Bauer')
    axc.set_xlabel(r'$\theta_{\mathrm{econ}}$ (AEP-opt, °)')
    axc.set_ylabel(r'$\theta_{\mathrm{min}}$ (min-exposure, °)')
    axc.set_xlim(0, 180); axc.set_ylim(0, 180)
    axc.set_xticks([0, 45, 90, 135, 180]); axc.set_yticks([0, 45, 90, 135, 180])
    axc.set_aspect('equal')
    axc.set_title('AEP-opt vs eco-opt orientation', fontsize=8)
    axc.legend(loc='lower right', fontsize=6, frameon=False, handletextpad=0.3,
               borderpad=0.2, labelspacing=0.2)
    _apply_style_all(axc); _label(axc, 'c')

    # (d) E(θ_econ)/E_min 比值 log 柱
    axd = fig.add_subplot(gs[1, 0])
    ratio = [(df.Ee / df.Emin.replace(0, np.nan)).median() for df in [on_df, vp_df, ba_df]]
    x = np.arange(3)
    axd.bar(x, ratio, width=0.55, color=[C_LAND, C_VPTS, C_BAUER], alpha=0.9,
            edgecolor='white', linewidth=0.6)
    axd.set_yscale('log')
    axd.set_ylim(1, 3000)
    for xi, v in zip(x, ratio):
        axd.text(xi, v * 1.7, f'{v:.0f}×', ha='center', fontsize=7.5, fontweight='bold')
    axd.set_xticks(x); axd.set_xticklabels(GROUP_ORDER)
    axd.set_ylabel(r'$E(\theta_{\mathrm{econ}})\,/\,E_{\mathrm{min}}$ (log)')
    axd.set_title('AEP-opt exposure over minimum', fontsize=8)
    _apply_style_all(axd); _label(axd, 'd')

    # (e) 中位方向对比（dumbbell）
    axe = fig.add_subplot(gs[1, 1])
    for i, df in enumerate([on_df, vp_df, ba_df]):
        te = df.theta_econ.median(); tm = df.th_min.median()
        y = 2 - i
        axe.plot([te, tm], [y, y], color='#999999', lw=1.2, zorder=2)
        axe.scatter([te], [y], s=55, marker='x', c=C_ECON, zorder=3, linewidths=1.4)
        axe.scatter([tm], [y], s=44, marker='o', c=C_ECO, zorder=3,
                    edgecolors='white', linewidths=0.5)
    axe.set_yticks([2, 1, 0]); axe.set_yticklabels(GROUP_ORDER)
    axe.set_xlim(0, 180); axe.set_xticks([0, 45, 90, 135, 180])
    axe.set_xlabel('Orientation (°)')
    axe.set_title('Median AEP-opt vs eco-opt', fontsize=8)
    axe.legend([Line2D([], [], marker='x', ls='', color=C_ECON, markersize=7, mew=1.4),
                Line2D([], [], marker='o', ls='', color=C_ECO, markersize=6,
                       markeredgecolor='white', markeredgewidth=0.5)],
               ['AEP-opt', 'eco-opt'], loc='upper right', fontsize=6.2,
               frameon=False, handletextpad=0.3, borderpad=0.2)
    _apply_style_all(axe); _label(axe, 'e')

    # (f) misalignment 密度直方图
    axf = fig.add_subplot(gs[1, 2])
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        axf.hist(df.d_full.values, bins=18, range=(0, 90), color=c, alpha=0.5,
                 density=True, label=lab, edgecolor='white', linewidth=0.3)
    axf.set_xlabel('Misalignment (°)'); axf.set_ylabel('Density')
    axf.set_title('Distribution of misalignment', fontsize=8)
    axf.legend(loc='upper right', fontsize=6.2, frameon=False, handlelength=1.2)
    _apply_style_all(axf); _label(axf, 'f')

    pdf, png = savefig(fig, 'fig3_misalignment')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 4 — R3 capture（双栏，2×3）
# =====================================================================
def fig4_capture(on_df, vp_df, ba_df, ctx):
    groups = [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'), (ba_df, C_BAUER, 'Bauer')]

    fig = plt.figure(figsize=(W_DOUBLE, 4.5))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.88, bottom=0.09,
                          hspace=0.75, wspace=0.45)

    # (a) 捕获曲线
    axa = fig.add_subplot(gs[0, 0])
    pts = [0, 5, 10, 20, 30]
    for df, c, lab in groups:
        vals = [0] + [df[f'frac{d}'].median() * 100 for d in (5, 10, 20, 30)]
        axa.plot(pts, vals, 'o-', color=c, lw=1.5, markersize=3.6,
                 markeredgecolor='white', markeredgewidth=0.5, label=lab)
    axa.axhline(50, color='#999999', ls=':', lw=0.8)
    axa.text(1.5, 53, 'half of max gain', fontsize=6.2, color='#555555')
    axa.axvspan(0, 20, color=C_ECO, alpha=0.10, zorder=0)
    axa.axvline(20, color=C_ECO, ls='--', lw=0.9)
    axa.text(20, 102, r'$\leq 20°$', color=C_ECO, fontsize=7, ha='center', fontweight='bold')
    axa.set_xlabel(r'Rotation $\Delta\theta$ (°)')
    axa.set_ylabel('Share of max cut (%)')
    axa.set_xlim(0, 32); axa.set_ylim(0, 108)
    axa.set_xticks([0, 10, 20, 30])
    axa.set_title('Exposure falls steeply at small angles', fontsize=8)
    axa.legend(loc='lower right', fontsize=6.2, frameon=False, handlelength=1.5)
    _apply_style_all(axa); _label(axa, 'a')

    # (b) Δθ50/Δθ80/d_full 分组柱（三色）
    axb = fig.add_subplot(gs[0, 1])
    x = np.arange(3); w = 0.26
    series = [(r'$\Delta\theta_{50}$', 'd50', '#A9CCE3'),
              (r'$\Delta\theta_{80}$', 'd80', '#2E86C1'),
              (r'$d_{\mathrm{full}}$', 'd_full', '#1B4F72')]
    for off, (lab, col, color) in zip([-w, 0, w], series):
        vals = [on_df[col].median(), vp_df[col].median(), ba_df[col].median()]
        axb.bar(x + off, vals, width=w, color=color, edgecolor='white',
                linewidth=0.5, label=lab, zorder=3)
        for xi, v in zip(x + off, vals):
            axb.text(xi, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=6.2,
                     color='#222222')
    axb.set_xticks(x); axb.set_xticklabels(GROUP_ORDER)
    axb.set_ylabel('Rotation needed (°)')
    axb.set_ylim(0, 62)
    axb.set_title('Half the gain needs one-third the rotation', fontsize=8)
    axb.legend(loc='upper left', fontsize=6.2, frameon=False, ncol=3,
               handlelength=1.0, columnspacing=0.9, borderpad=0.2)
    _apply_style_all(axb); _label(axb, 'b')

    # (c) 占比：≤20° / ≤30° 捕获 >50%
    axc = fig.add_subplot(gs[0, 2])
    x = np.arange(3); w = 0.38
    thr = [(r'$\leq 20°$', 'frac20', '#AEB6BF'),
           (r'$\leq 30°$', 'frac30', '#5D6D7E')]
    for off, (lab, col, color) in zip([-w / 2, w / 2], thr):
        vals = [(df[col] * 100 > 50).mean() * 100 for df, _, _ in groups]
        axc.bar(x + off, vals, width=w, color=color, edgecolor='white',
                linewidth=0.5, label=lab, zorder=3)
        for xi, v in zip(x + off, vals):
            axc.text(xi, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=6.5,
                     color='#222222')
    axc.axhline(50, color='#999999', ls=':', lw=0.8)
    axc.set_xticks(x); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel('Share of farms (%)')
    axc.set_ylim(0, 105)
    axc.set_title('Most farms >50% within 20°', fontsize=8)
    axc.legend(loc='upper right', fontsize=6.2, frameon=False, ncol=2,
               handlelength=1.0, columnspacing=0.8, borderpad=0.2)
    _apply_style_all(axc); _label(axc, 'c')

    # (d) Δθ50 小提琴
    axd = fig.add_subplot(gs[1, 0])
    dfd = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        r'$\Delta\theta_{50}$ (°)': np.concatenate([on_df.d50.values, vp_df.d50.values, ba_df.d50.values]),
    })
    sns.violinplot(data=dfd, x='Group', y=r'$\Delta\theta_{50}$ (°)', ax=axd,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.5, saturation=0.9, legend=False, cut=0)
    for i, (df, _, _) in enumerate(groups):
        m = df.d50.median()
        axd.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.2, zorder=5)
        axd.text(i, m + 2, f'{m:.0f}°', ha='center', va='bottom', fontsize=7, fontweight='bold')
    axd.set_xlabel('')
    axd.set_ylim(0, 62)
    axd.set_title('Rotation for 50% of max gain', fontsize=8)
    _apply_style_all(axd); _label(axd, 'd')

    # (e) Δθ80 小提琴
    axe = fig.add_subplot(gs[1, 1])
    dfe = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        r'$\Delta\theta_{80}$ (°)': np.concatenate([on_df.d80.values, vp_df.d80.values, ba_df.d80.values]),
    })
    sns.violinplot(data=dfe, x='Group', y=r'$\Delta\theta_{80}$ (°)', ax=axe,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.5, saturation=0.9, legend=False, cut=0)
    for i, (df, _, _) in enumerate(groups):
        m = df.d80.median()
        axe.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.2, zorder=5)
        axe.text(i, m + 2, f'{m:.0f}°', ha='center', va='bottom', fontsize=7, fontweight='bold')
    axe.set_xlabel('')
    axe.set_ylim(0, 62)
    axe.set_title('Rotation for 80% of max gain', fontsize=8)
    _apply_style_all(axe); _label(axe, 'e')

    # (f) ≤30° 内捕获 >80% 场址占比
    axf = fig.add_subplot(gs[1, 2])
    x = np.arange(3)
    vals = [(df.frac30 * 100 > 80).mean() * 100 for df, _, _ in groups]
    axf.bar(x, vals, width=0.55, color=[C_LAND, C_VPTS, C_BAUER], alpha=0.9,
            edgecolor='white', linewidth=0.6)
    for xi, v in zip(x, vals):
        axf.text(xi, v + 1.5, f'{v:.0f}%', ha='center', fontsize=7.5, fontweight='bold')
    axf.axhline(50, color='#999999', ls=':', lw=0.8)
    axf.set_xticks(x); axf.set_xticklabels(GROUP_ORDER)
    axf.set_ylabel('Share of farms (%)')
    axf.set_ylim(0, 100)
    axf.set_title(r'>80% within $\leq 30°$', fontsize=8)
    _apply_style_all(axf); _label(axf, 'f')

    pdf, png = savefig(fig, 'fig4_capture')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 5 — R4 trade-off（双栏，6 地图 + 2 折线）
# =====================================================================
def fig5_tradeoff(on_df, vp_df, ba_df, ctx):
    BUDGETS = [0.005, 0.01, 0.02, 0.05]
    BLABELS = ['0.5', '1', '2', '5']
    XP = np.arange(len(BUDGETS))

    on_b1 = _on_rr_at(ctx, 0.01); on_b5 = _on_rr_at(ctx, 0.05)
    off_b1 = _off_rr_at(ctx, 0.01); off_b5 = _off_rr_at(ctx, 0.05)

    def _split(m):
        return m[m.source == 'VPTS'], m[m.source == 'Bauer_grid']

    vp_b1, ba_b1 = _split(off_b1); vp_b5, ba_b5 = _split(off_b5)

    map_rows = [
        ('Onshore', '1% AEP', on_b1), ('Onshore', '5% AEP', on_b5),
        ('VPTS', '1% AEP', vp_b1), ('VPTS', '5% AEP', vp_b5),
        ('Bauer', '1% AEP', ba_b1), ('Bauer', '5% AEP', ba_b5),
    ]

    fig = plt.figure(figsize=(W_DOUBLE, 5.2))
    gsm = fig.add_gridspec(2, 3, left=0.045, right=0.86, top=0.94, bottom=0.44,
                           hspace=0.32, wspace=0.14)
    gsl = fig.add_gridspec(1, 2, left=0.09, right=0.92, top=0.33, bottom=0.09, wspace=0.32)

    axes = []
    if HAS_CARTOPY:
        for k, (grp, bud, m) in enumerate(map_rows):
            ax = fig.add_subplot(gsm[k // 3, k % 3], projection=ccrs.PlateCarree())
            axes.append(ax)
            col = 'risk_reduction' if grp == 'Onshore' else 'risk_reduction_pct'
            s = 1.4 if grp == 'Onshore' else 42
            if grp == 'Onshore':
                add_basemap(ax, [m.centroid_lon.min() - 0.5, m.centroid_lon.max() + 0.5,
                                 m.centroid_lat.min() - 0.5, m.centroid_lat.max() + 0.5])
                ax.scatter(m.centroid_lon, m.centroid_lat, c=m[col], cmap=CMAP_RISK,
                           s=s, alpha=0.7, edgecolors='none', vmin=0, vmax=100,
                           transform=ccrs.PlateCarree(), zorder=3)
            else:
                add_basemap(ax, [-3.5, 9.5, 49.5, 56.5])
                ax.scatter(m.centroid_lon, m.centroid_lat, c=m[col], cmap=CMAP_RISK,
                           s=s, edgecolors='white', linewidths=0.4, vmin=0, vmax=100,
                           transform=ccrs.PlateCarree(), zorder=4)
            ax.set_title(f'{grp} — {bud} budget', fontsize=7.5, pad=2)
            _label(ax, chr(ord('a') + k), x=-0.02, y=1.10)

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(0, 100), cmap=CMAP_RISK); sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
        cb.set_label('Exposure reduction (%)', fontsize=7)
        cb.set_ticks([0, 25, 50, 75, 100])
        cb.ax.tick_params(width=0.5, length=2.0, labelsize=6.5)

    # (g) 中位 RR vs 预算
    _, rr_b = _median_rr_vs_budget(ctx)
    _, ae_b = _mean_aep_vs_budget(ctx)
    rr_vals = np.array([[rr_b[b][i] for b in BUDGETS] for i in range(3)])
    ae_vals = np.array([[ae_b[b][i] for b in BUDGETS] for i in range(3)])

    axg = fig.add_subplot(gsl[0, 0])
    for i, (c, lab) in enumerate([(C_LAND, 'Onshore'), (C_VPTS, 'VPTS'), (C_BAUER, 'Bauer')]):
        axg.plot(XP, rr_vals[i], 'o-', color=c, lw=1.5, markersize=3.8,
                 markeredgecolor='white', markeredgewidth=0.5, label=lab)
    axg.set_xticks(XP); axg.set_xticklabels(BLABELS)
    axg.set_xlabel('AEP budget (%)'); axg.set_ylabel('Median exposure reduction (%)')
    axg.set_ylim(30, 108)
    axg.axvspan(-0.5, 1, color=C_ECO, alpha=0.08, zorder=0)
    axg.axvline(1, color=C_ECO, ls='--', lw=0.9)
    axg.text(1, 105, r'$\leq 1\%$ AEP', color=C_ECO, fontsize=7, ha='center', fontweight='bold')
    axg.legend(loc='lower right', fontsize=6.5, frameon=False, handlelength=1.5)
    _apply_style_all(axg); _label(axg, 'g')

    axh = fig.add_subplot(gsl[0, 1])
    for i, c in enumerate([C_LAND, C_VPTS, C_BAUER]):
        axh.plot(XP, ae_vals[i], 's--', color=c, lw=1.1, markersize=3.4,
                 markeredgecolor='white', markeredgewidth=0.4, alpha=0.9)
    axh.set_xticks(XP); axh.set_xticklabels(BLABELS)
    axh.set_xlabel('AEP budget (%)'); axh.set_ylabel('Mean AEP cost (%)')
    axh.set_ylim(0, 1.1)
    axh.axvspan(-0.5, 1, color=C_ECO, alpha=0.08, zorder=0)
    axh.axvline(1, color=C_ECO, ls='--', lw=0.9)
    _apply_style_all(axh); _label(axh, 'h')

    fig.text(0.5, 0.395, 'Most gain at negligible energy cost', fontsize=8.5,
             ha='center', va='center', fontweight='bold')

    pdf, png = savefig(fig, 'fig5_tradeoff')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig S2 — 敏感性空间普遍性（原有，重排为 Nature Energy 双栏）
# =====================================================================
def figS2_universality(on_df, vp_df, ba_df, ctx):
    od = ctx['od']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']

    fig = plt.figure(figsize=(W_DOUBLE, 2.4))
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.985, top=0.85, bottom=0.18,
                          wspace=0.55, width_ratios=[1.35, 1.0, 1.0])

    if HAS_CARTOPY:
        axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        add_basemap(axa, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axa.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.rel * 100,
                         cmap=CMAP_RISK, s=1.4, alpha=0.65, edgecolors='none',
                         vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=3)
        axa.scatter(vpts.centroid_lon, vpts.centroid_lat, s=22, c=C_VPTS, marker='s',
                    edgecolors='white', linewidths=0.4, transform=ccrs.PlateCarree(), zorder=4)
        axa.scatter(bauer.centroid_lon, bauer.centroid_lat, s=22, c=C_BAUER, marker='s',
                    edgecolors='white', linewidths=0.4, transform=ccrs.PlateCarree(), zorder=4)
        for sn, (la, lo) in RADAR_LOC.items():
            axa.scatter(lo, la, marker='^', s=30, c='#C0392B', edgecolors='white',
                        linewidths=0.4, transform=ccrs.PlateCarree(), zorder=5)
        axa.set_title('Sensitivity across space', fontsize=7.5)
        _label(axa, 'a', x=-0.02, y=1.10)
        cb = fig.colorbar(sc, ax=axa, shrink=0.85, pad=0.02, fraction=0.045)
        cb.set_label('Rel. change (%)', fontsize=6.5)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.5, length=2.0, labelsize=6.5)

    axb = fig.add_subplot(gs[0, 1])
    bp = axb.boxplot([on_df.rel.values * 100, vp_df.rel.values * 100, ba_df.rel.values * 100],
                     positions=[0, 1, 2], widths=0.5, patch_artist=True,
                     medianprops=dict(color='black', lw=1.1),
                     whiskerprops=dict(color='#666666', lw=0.7),
                     capprops=dict(color='#666666', lw=0.7),
                     flierprops=dict(marker='o', markersize=1.8, markeredgewidth=0, color='#999999'))
    for patch, c in zip(bp['boxes'], [C_LAND, C_VPTS, C_BAUER]):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor('white'); patch.set_linewidth(0.5)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.rel.median() * 100
        axb.text(i, m + 0.4, f'{m:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    axb.set_xticks([0, 1, 2]); axb.set_xticklabels(GROUP_ORDER)
    axb.set_ylim(85, 102)
    axb.set_ylabel('Rel. change (%)')
    axb.set_title('Sensitivity, three groups', fontsize=7.5)
    _apply_style_all(axb); _label(axb, 'b')

    axc = fig.add_subplot(gs[0, 2])
    thr = np.arange(90, 101, 1)
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        vals = [(df.rel.values * 100 >= t).mean() * 100 for t in thr]
        axc.plot(thr, vals, 'o-', color=c, lw=1.3, markersize=3, label=lab,
                 markeredgecolor='white', markeredgewidth=0.4)
    axc.axhline(100, color='#999999', ls=':', lw=0.7)
    axc.set_xlabel('Rel. change threshold (%)')
    axc.set_ylabel('Share of farms above (%)')
    axc.set_xlim(90, 100); axc.set_ylim(0, 105)
    axc.set_xticks([90, 92, 94, 96, 98, 100])
    axc.set_title('Universality of high sensitivity', fontsize=7.5)
    axc.legend(loc='lower left', fontsize=6.2, frameon=False, handlelength=1.5)
    _apply_style_all(axc); _label(axc, 'c')

    pdf, png = savefig(fig, 'figS2_universality')
    plt.close(fig)
    return pdf, png


def _figS1_wrapper(ctx):
    """S1 用 v2 布局；若原始 VPTS 数据缺失，则回落到已发布的 outputs/figure/figS1_threat.png。"""
    import shutil
    try:
        from generate_paper_figures_v2 import figS1_threat as _s1
        pdf, png = _s1(ctx)
        for src in (pdf, png):
            dst = os.path.join(FIG5, 'figS1_threat' + os.path.splitext(src)[1])
            shutil.copy(src, dst)
    except FileNotFoundError:
        # 回落到已发布的 PNG
        src = os.path.join(BASE, '..', 'outputs', 'figure', 'figS1_threat.png')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIG5, 'figS1_threat.png'))
            print('  (fallback: copied existing figS1_threat.png)')
    return None


# ---------------------------------------------------------------------------
def main():
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}, cartopy={HAS_CARTOPY}')

    print('Fig 1 (framework + region) ...'); fig1_framework_region(on_df, vp_df, ba_df, ctx)
    print('Fig 2 (sensitivity) ...'); fig2_sensitivity(on_df, vp_df, ba_df, ctx)
    print('Fig 3 (misalignment) ...'); fig3_misalignment(on_df, vp_df, ba_df, ctx)
    print('Fig 4 (capture) ...'); fig4_capture(on_df, vp_df, ba_df, ctx)
    print('Fig 5 (trade-off) ...'); fig5_tradeoff(on_df, vp_df, ba_df, ctx)
    print('Fig S2 (universality) ...'); figS2_universality(on_df, vp_df, ba_df, ctx)
    print('Fig S1 (threat, v2 layout copied) ...'); _figS1_wrapper(ctx)

    print('\nDone. Outputs in figures_v5/:')
    for f in sorted(os.listdir(FIG5)):
        p = os.path.join(FIG5, f)
        print(f'  {f}  ({os.path.getsize(p)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()
