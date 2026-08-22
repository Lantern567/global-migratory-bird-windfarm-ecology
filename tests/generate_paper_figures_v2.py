# -*- coding: utf-8 -*-
"""
generate_paper_figures_v2.py —— 论文制图（v2 重构版，含用户逐图审查修订）。

结构（学长要求：4 张主图 = 每个 R 一张，各 6 panel；+ 1 张 overview 框架图 + 1 张威胁背景 supporting
       + 1 张研究区地图 fig0_study_area，加强空间可视化）：
  fig0_study_area 研究区地图（欧洲总览 + 海上放大 + 陆上场址 n_turbines）
  fig0_overview   图形摘要式框架图（每阶段 = 真实数据缩略图 + 标题数字 + 结论）
  fig1_R1         迁徙方向使碰撞暴露对风场方向高度敏感
  fig2_R2         能量最优方向保留大量可避免暴露
  fig3_R3         有限方向改变捕获大部分可削减暴露
  fig4_R4         大部分暴露削减以极小能源代价实现
  figS1_threat    候鸟威胁数量/密度/通量/高度/时相（背景，非 R1–R4 结果）

每张主图 6 panel，按「前提→机制→实证→量化→普遍性→空间」递进排列（见 results_framework.md）。
所有数字来自 figure_style.compute_metrics()，与 final_numbers.txt 一致。
图内文字英文；mathtext 一律 raw string；色盲安全 Okabe-Ito 色板；矢量(PDF)+PNG 双导出。
PDF 体积：底图栅格化(zorder<2) + quiver 抽稀。

全局修订（用户审查意见）：
  1. 面板标签 16pt 加粗纯黑、左上角偏移（figure_style.panel_label）。
  2. 颜色全局固定：春季=#E69F00 橙、秋季=#56B4E9 天蓝、陆上/VPTS/Bauer 三组固定。
  3. 图例不遮挡：柱/小提琴/折线图例放顶部外侧或数据空白角，散点图例加白底。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN, C_ECON, C_ECO,
                          GROUP_COLORS, GROUP_ORDER, THETAS, CMAP_CONC, CMAP_RISK,
                          style_ax, panel_label, savefig, circ, on_Evec, off_Evec)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

LAND_F = '#F2F0E6'
OCEAN_F = '#D6E4F0'
VPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'radar_vpts')

# mathtext 标签（raw string，避免反斜杠转义问题；no-latex 下用 mathtext 字体渲染 ≤/Δ/θ）
L_THECON = r'$\theta_{\mathrm{econ}}$'
L_THMIN  = r'$\theta_{\mathrm{min}}$'
L_ECON   = r'$E_{\mathrm{econ}}$'
L_EMIN   = r'$E_{\mathrm{min}}$'
L_DTH50  = r'$\Delta\theta_{50}$'
L_DTH80  = r'$\Delta\theta_{80}$'
L_LEQ    = r'$\leq$'
L_THETA  = r'$\theta$'

# 雷达站坐标（lat, lon）——全部来自本仓库已有脚本，勿臆造
RADAR_LOC = {
    'nlhrw': (52.95, 4.75),   # Den Helder, NL
    'bejab': (51.18, 3.07),   # Jabbeke, BE
    'deess': (51.40, 6.97),   # Essen, DE
    'frabb': (50.13, 1.83),   # Abbeville, FR
    'behel': (51.05, 5.42),   # Helchteren, BE  (visualize_maps_fig1_4.py)
    'nldhl': (51.84, 5.15),   # Herwijnen, NL   (visualize_maps_fig1_4.py)
}
HEIGHT_STATIONS = {'nlhrw', 'bejab', 'deess', 'frabb'}  # 高度/月相所用 VPTS 站（同旧稿）


def add_basemap(ax, extent):
    """加底图并栅格化（zorder<2），降低矢量 PDF 体积；数据(zorder>=2)保矢量。"""
    ax.add_feature(cfeature.LAND, facecolor=LAND_F, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_F, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#999999', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#BBBBBB', zorder=1)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_rasterization_zorder(2)


def rep_farm(df):
    idx = (df.avoid - df.avoid.median()).abs().idxmin()
    return df.loc[idx]


def rep_curve(df, on1, cur):
    """返回代表场的归一化 E(θ)（0–1）与 farm_id。"""
    r = rep_farm(df)
    fid = r.farm_id
    if fid in set(on1.farm_id):
        row = on1[on1.farm_id == fid].iloc[0]
        E = on_Evec(row.spring_dir, row.autumn_dir, row.spring_conc, row.autumn_conc)
    else:
        E = off_Evec(fid, cur)
    E = E / max(E.max(), 1e-12)
    return fid, E, r


def _pos(v):
    """正数保留，非正/NaN 置 NaN（log 轴不画缺失条）。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if (v == v and v > 0) else np.nan


def _vpts_height_and_month():
    """读原始 VPTS：高度分层占比 + 月相（夜迁 20-06、dens>10、春 3-5 月 / 秋 8-11 月）。
    站点过滤与旧稿 compute_height_profile 一致（4 站）。"""
    from collections import defaultdict
    bins = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200)]

    def _f(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float('nan')

    height = {'spring': defaultdict(float), 'autumn': defaultdict(float)}
    month = {'spring': defaultdict(float), 'autumn': defaultdict(float)}
    for fname in sorted(os.listdir(VPTS_DIR)):
        if not fname.endswith('.txt'):
            continue
        if fname.split('_')[0] not in HEIGHT_STATIONS:
            continue
        with open(os.path.join(VPTS_DIR, fname), encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                p = line.split()
                if len(p) < 18:
                    continue
                try:
                    hght = int(p[2]); dens = _f(p[12]); mm = int(p[0][4:6])
                except (ValueError, IndexError):
                    continue
                if dens != dens or dens < 10:
                    continue
                hour = int(p[1][:2])
                if 6 < hour < 20:
                    continue
                if 3 <= mm <= 5:
                    season = 'spring'
                elif 8 <= mm <= 11:
                    season = 'autumn'
                else:
                    continue
                height[season][hght] += dens
                month[season][mm] += dens

    hfrac = {}
    for season in ('spring', 'autumn'):
        fracs = []
        for lo, hi in bins:
            fracs.append(sum(height[season][h] for h in height[season] if lo <= h < hi))
        fracs.append(sum(height[season][h] for h in height[season] if h >= 1200))
        tot = sum(fracs)
        hfrac[season] = [f / tot * 100 if tot > 0 else 0 for f in fracs]
    return hfrac, dict(month)


# =====================================================================
# Fig 0 — 研究区地图（空间总览：欧洲陆上 + 海上 + 雷达 + 格网）
# =====================================================================
def fig0_study_area(ctx):
    on1 = ctx['on1']; od = ctx['od']; grid = ctx['grid']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']

    if not HAS_CARTOPY:
        print('  [skip] fig0_study_area (no cartopy)')
        return None, None

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.10,
                          left=0.06, right=0.94, top=0.86, bottom=0.08)

    # (a) 研究区总览（跨上排两列）
    axa = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    add_basemap(axa, [-12, 20, 39, 59])
    gx = [grid.lon.min(), grid.lon.max(), grid.lon.max(), grid.lon.min(), grid.lon.min()]
    gy = [grid.lat.min(), grid.lat.min(), grid.lat.max(), grid.lat.max(), grid.lat.min()]
    axa.plot(gx, gy, color='#333333', lw=1.1, ls='--', transform=ccrs.PlateCarree(), zorder=2)
    axa.scatter(on1.centroid_lon, on1.centroid_lat, s=0.7, c=C_LAND, alpha=0.35,
                edgecolors='none', transform=ccrs.PlateCarree(), zorder=3)
    axa.scatter(vpts.centroid_lon, vpts.centroid_lat, s=16, c=C_VPTS, edgecolors='white',
                linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
    axa.scatter(bauer.centroid_lon, bauer.centroid_lat, s=16, c=C_BAUER, edgecolors='white',
                linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
    for sn, (la, lo) in RADAR_LOC.items():
        axa.scatter(lo, la, marker='^', s=55, c='#C0392B', edgecolors='white', linewidths=0.6,
                    transform=ccrs.PlateCarree(), zorder=5)
    axa.set_title('Study region: Western Europe onshore + North Sea offshore', fontsize=9)
    panel_label(axa, 'a')

    # (b) 海上放大：55 场（VPTS/Bauer）+ 雷达站
    axb = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    add_basemap(axb, [-3.5, 15, 42, 57])
    axb.scatter(vpts.centroid_lon, vpts.centroid_lat, s=30, c=C_VPTS, edgecolors='white',
                linewidths=0.4, transform=ccrs.PlateCarree(), zorder=4, label=f'VPTS (n={len(vpts)})')
    axb.scatter(bauer.centroid_lon, bauer.centroid_lat, s=30, c=C_BAUER, edgecolors='white',
                linewidths=0.4, transform=ccrs.PlateCarree(), zorder=4, label=f'Bauer (n={len(bauer)})')
    for sn, (la, lo) in RADAR_LOC.items():
        axb.scatter(lo, la, marker='^', s=60, c='#C0392B', edgecolors='white', linewidths=0.6,
                    transform=ccrs.PlateCarree(), zorder=5)
        axb.text(lo, la + 0.28, sn, fontsize=6.5, ha='center', va='bottom', color='#C0392B',
                 transform=ccrs.PlateCarree(), zorder=6,
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'))
    axb.legend(loc='upper right', fontsize=6.5, frameon=True, framealpha=1.0,
               edgecolor='#CCCCCC')
    axb.set_title('Offshore farms & radar stations', fontsize=9)
    panel_label(axb, 'b')

    # (c) 陆上场址：n_turbines（色标）
    axc = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    add_basemap(axc, [on1.centroid_lon.min() - 1, on1.centroid_lon.max() + 1,
                      on1.centroid_lat.min() - 1, on1.centroid_lat.max() + 1])
    nt = on1.n_turbines.fillna(0).clip(lower=0)
    sc = axc.scatter(on1.centroid_lon, on1.centroid_lat, c=nt, cmap='cividis', s=2.5,
                     alpha=0.65, edgecolors='none', transform=ccrs.PlateCarree(), zorder=3)
    axc.set_title('Onshore farms — number of turbines', fontsize=9)
    panel_label(axc, 'c')
    cax = make_axes_locatable(axc).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('Turbines per farm', fontsize=7); cb.ax.tick_params(width=0.6, length=2.5, labelsize=6.5)

    # 图级图例（一次，避免逐 panel 遮挡）
    handles = [
        Line2D([], [], marker='o', ls='', color=C_LAND, markersize=5, label=f'Onshore (n={len(on1)})'),
        Line2D([], [], marker='o', ls='', color=C_VPTS, markersize=5, label=f'Offshore VPTS (n={len(vpts)})'),
        Line2D([], [], marker='o', ls='', color=C_BAUER, markersize=5, label=f'Offshore Bauer (n={len(bauer)})'),
        Line2D([], [], marker='^', ls='', color='#C0392B', markersize=6, label='Radar station (VPTS)'),
        Line2D([], [], marker='', ls='--', color='#333333', lw=1.1, label='Bauer direction grid'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.985),
               ncol=5, fontsize=6.5, frameon=False)

    pdf, png = savefig(fig, 'fig0_study_area')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 0 — Overview（图形摘要式：每阶段真实数据缩略图 + 数字 + 结论）
# =====================================================================
def fig0_overview(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']

    fig = plt.figure(figsize=(12.5, 4.3))
    fig.text(0.5, 0.965,
             'Wind-farm orientation as a low-cost lever against migratory-bird collision exposure',
             ha='center', va='center', fontsize=13, fontweight='bold', color='black')
    fig.text(0.5, 0.905,
             'Mechanism  →  Conflict  →  Mitigation  →  Trade-off    (values = medians across farms)',
             ha='center', va='center', fontsize=8, color='#555555')

    # 缩略图数据（真实）
    th = np.linspace(0, 180, 200)
    r2fid, r2E, r2 = rep_curve(on_df, on1, cur)
    r3_pts = [0, 5, 10, 20, 30]
    r3_vals = [0] + [on_df[f'frac{d}'].median() * 100 for d in (5, 10, 20, 30)]
    budgets = [0.005, 0.01, 0.02, 0.05]
    on = ctx['on']
    r4_vals = [on[on.budget == b]['risk_reduction'].median() for b in budgets]

    stages = [
        ('R1  Mechanism', 'Exposure is highly\norientation-sensitive', '93.4–99.9%', C_SPRING,
         'curve', (th, np.sin(np.radians(th - 51.5)) ** 2, None)),
        ('R2  Conflict', 'AEP-opt leaves most\nexposure avoidable', '90.0–99.8%', C_LAND,
         'r2', (THETAS, r2E, r2)),
        ('R3  Mitigation', 'A small change captures\nmost of the gain', '14–18°', C_VPTS,
         'r3', (r3_pts, r3_vals, None)),
        ('R4  Trade-off', '1% AEP captures most\ngain, then saturates', '74.5–98.0%', C_BAUER,
         'r4', (budgets, r4_vals, None)),
    ]

    gs = fig.add_gridspec(1, 4, left=0.045, right=0.955, top=0.86, bottom=0.32, wspace=0.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax, (tag, claim, num, c, kind, data) in zip(axes, stages):
        # 标题（彩色 tag）
        ax.set_title(tag, loc='left', fontsize=8.5, fontweight='bold', color=c, pad=3)
        # 标题数字（右上，紧贴子图上方）
        ax.text(0.99, 1.02, num, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, fontweight='bold', color=c)
        # 缩略图
        ax.tick_params(labelsize=6.5, width=0.5, length=1.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if kind == 'curve':
            ax.plot(data[0], data[1], color='#333333', lw=1.3)
            ax.set_xlim(0, 180); ax.set_ylim(0, 1.05)
            ax.set_xticks([0, 90, 180]); ax.set_yticks([0, 0.5, 1])
        elif kind == 'r2':
            ax.plot(data[0], data[1], color=C_LAND, lw=1.3)
            ax.axvline(data[2].theta_econ, color=C_ECON, ls='--', lw=0.9)
            ax.axvline(data[2].th_min, color=C_ECO, ls='-', lw=1.1)
            ax.set_xlim(0, 180); ax.set_ylim(0, 1.05)
            ax.set_xticks([0, 90, 180]); ax.set_yticks([0, 0.5, 1])
        elif kind == 'r3':
            ax.plot(data[0], data[1], 'o-', color=C_VPTS, lw=1.3, markersize=2.8)
            ax.set_xlim(0, 32); ax.set_ylim(0, 100)
            ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 50, 100])
        elif kind == 'r4':
            ax.plot(data[0], data[1], 'o-', color=C_BAUER, lw=1.3, markersize=2.8)
            ax.set_xlim(0, 0.055); ax.set_ylim(40, 105)
            ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
            ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
            ax.set_yticks([50, 75, 100])
        # 结论（xlabel 两行）
        ax.set_xlabel(claim, fontsize=6.8, color='#333333')

    # 阶段间箭头
    pos = [ax.get_position() for ax in axes]
    ymid = (pos[0].y0 + pos[0].y1) / 2
    for i in range(3):
        fig.add_artist(FancyArrowPatch((pos[i].x1 + 0.006, ymid), (pos[i + 1].x0 - 0.006, ymid),
                                       arrowstyle='-|>', mutation_scale=12,
                                       linewidth=0.9, color='#999999',
                                       transform=fig.transFigure))

    fig.text(0.5, 0.15,
             'Re-orienting arrays within a 1% AEP budget converts most avoidable exposure into ecological gain.',
             ha='center', va='center', fontsize=8.5, style='italic', color='#222222')

    pdf, png = savefig(fig, 'fig0_overview')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 1 — R1（前提/机制/实证/量化/普遍性，6 panel）
# =====================================================================
def fig1_R1(on_df, vp_df, ba_df, ctx):
    grid = ctx['grid'].dropna(subset=['spring_dir', 'autumn_dir']).copy()
    on1 = ctx['on1']; cur = ctx['cur']

    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (a) 玫瑰图：方向结构（春秋近反向）
    axa = fig.add_subplot(gs[0, 0], projection='polar')
    bins = np.arange(0, 361, 15)
    sp_hist, _ = np.histogram(grid.spring_dir.values, bins=bins)
    au_hist, _ = np.histogram(grid.autumn_dir.values, bins=bins)
    tcent = np.radians((bins[:-1] + bins[1:]) / 2)
    width = np.radians(15)
    axa.set_theta_zero_location('N'); axa.set_theta_direction('clockwise')
    axa.bar(tcent, sp_hist, width=width, color=C_SPRING, alpha=0.6, zorder=2)
    axa.bar(tcent, au_hist, width=width, color=C_AUTUMN, alpha=0.6, zorder=2)
    axa.set_ylim(0, 1.0); axa.set_yticks([])
    axa.set_title('Directional structure (near-antiparallel)', fontsize=8, pad=8)
    axa.legend([plt.Line2D([], [], color=C_SPRING, lw=5, alpha=0.6),
                plt.Line2D([], [], color=C_AUTUMN, lw=5, alpha=0.6)],
               ['Spring', 'Autumn'], loc='upper right', bbox_to_anchor=(1.28, 1.14),
               fontsize=6.5, frameon=False)
    panel_label(axa, 'a')

    # (b) 春季方向场地图（抽稀 quiver）
    axb = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        gsub = grid[(grid.row % 2 == 0) & (grid.col % 2 == 0)]
        extent = [grid.lon.min() - 0.5, grid.lon.max() + 0.5, grid.lat.min() - 0.5, grid.lat.max() + 0.5]
        add_basemap(axb, extent)
        u = np.sin(np.radians(gsub.spring_dir.values)); v = np.cos(np.radians(gsub.spring_dir.values))
        sc = axb.quiver(gsub.lon.values, gsub.lat.values, u, v, gsub.spring_conc.values,
                        cmap=CMAP_CONC, scale=42, width=0.0032, headwidth=2.8, headlength=3.2,
                        transform=ccrs.PlateCarree(), zorder=3)
        sc.set_clim(0.5, 0.9)
        axb.set_title('Spring migration direction (grid)', fontsize=8)
        panel_label(axb, 'b')
        cax = make_axes_locatable(axb).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('Concentration', fontsize=7); cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9])
        cb.ax.tick_params(width=0.6, length=2.5, labelsize=6.5)

    # (c) 机制：sin²(θ−φ) 几何
    axc = fig.add_subplot(gs[0, 2])
    phi = 51.5
    thc = np.linspace(0, 180, 400)
    Ec = np.sin(np.radians(thc - phi)) ** 2
    axc.plot(thc, Ec, color='#333333', lw=1.6)
    axc.axvline(phi, color=C_VPTS, ls='--', lw=1.0)
    axc.axvline((phi + 90) % 180, color=C_SPRING, ls='--', lw=1.0)
    axc.annotate('parallel = low', xy=(phi, 0.02), xytext=(phi, 0.30),
                 fontsize=7, color=C_VPTS, ha='center',
                 arrowprops=dict(arrowstyle='->', color=C_VPTS, lw=0.8))
    axc.annotate('perpendicular = high', xy=((phi + 90) % 180, 1.0),
                 xytext=((phi + 90) % 180, 1.14), fontsize=7, color=C_SPRING, ha='center',
                 arrowprops=dict(arrowstyle='->', color=C_SPRING, lw=0.8))
    axc.set_xlabel('Array orientation ' + L_THETA + ' (°)')
    axc.set_ylabel(r'Exposure $\propto \sin^2(\theta-\phi)$')
    axc.set_xlim(0, 180); axc.set_ylim(0, 1.22)
    axc.set_title('Why orientation matters (geometry)', fontsize=8)
    panel_label(axc, 'c'); style_ax(axc)

    # (d) 实证：代表场 E(θ)
    axd = fig.add_subplot(gs[1, 0])
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'), (ba_df, C_BAUER, 'Bauer')]:
        fid, E, r = rep_curve(df, on1, cur)
        axd.plot(THETAS, E, color=c, lw=1.3, label=lab)
    axd.set_xlabel('Array orientation ' + L_THETA + ' (°)'); axd.set_ylabel('Normalized exposure')
    axd.set_xlim(0, 180); axd.set_ylim(0, 1.03)
    axd.set_title('E(' + L_THETA + ') under real orientation', fontsize=8)
    axd.legend(loc='lower center', fontsize=6, frameon=False, ncol=3)
    panel_label(axd, 'd'); style_ax(axd)

    # (e) 量化：相对变化小提琴
    axe = fig.add_subplot(gs[1, 1])
    rels = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Relative change (%)': np.concatenate([on_df.rel.values * 100,
                                               vp_df.rel.values * 100, ba_df.rel.values * 100]),
    })
    sns.violinplot(data=rels, x='Group', y='Relative change (%)', ax=axe,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.6, saturation=0.9, legend=False)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.rel.median() * 100
        axe.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.3, zorder=5)
        axe.text(i, 103, f'{m:.1f}%', ha='center', va='top', fontsize=7, zorder=6)
    axe.set_ylim(0, 108)
    axe.set_title('Exposure sensitivity magnitude', fontsize=8)
    panel_label(axe, 'e'); style_ax(axe)

    # (f) 普遍性：高敏感比例
    axf = fig.add_subplot(gs[1, 2])
    x = np.arange(3); w = 0.36
    for thr, off, lab in [(0.5, -w / 2, '>50% change'), (0.8, w / 2, '>80% change')]:
        vals = [(df.rel > thr).mean() * 100 for df in (on_df, vp_df, ba_df)]
        axf.bar(x + off, vals, w, label=lab, color=(C_ECON if thr == 0.5 else C_ECO))
        for xi, v in zip(x, vals):
            axf.text(xi + off, v + 1.5, f'{v:.0f}', ha='center', fontsize=6.5)
    axf.set_xticks(x); axf.set_xticklabels(GROUP_ORDER)
    axf.set_ylabel('Share of farms (%)'); axf.set_ylim(0, 108)
    axf.set_title('Universality of high sensitivity', fontsize=8)
    axf.legend(loc='lower center', fontsize=6, frameon=False, ncol=2)
    panel_label(axf, 'f'); style_ax(axf)

    pdf, png = savefig(fig, 'fig1_R1')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 2 — R2
# =====================================================================
def fig2_R2(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (a) 冲突概念
    axa = fig.add_subplot(gs[0, 0])
    r = rep_farm(on_df)
    row = on1[on1.farm_id == r.farm_id].iloc[0]
    E = on_Evec(row.spring_dir, row.autumn_dir, row.spring_conc, row.autumn_conc)
    E = E / E.max()
    axa.plot(THETAS, E, color=C_LAND, lw=1.4)
    te = r.theta_econ; tm = r.th_min
    axa.axvline(te, color=C_ECON, ls='--', lw=1.0, label=L_THECON + ' (AEP-opt)')
    axa.axvline(tm, color=C_ECO, ls='-', lw=1.2, label=L_THMIN + ' (eco-opt)')
    axa.annotate('', xy=(te, E[int(round(te)) % 180]), xytext=(tm, E[int(round(tm)) % 180]),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
    axa.text((te + tm) / 2, 1.05, f'{circ(te, tm):.0f}°', ha='center', fontsize=7.5)
    axa.set_xlabel('Array orientation ' + L_THETA + ' (°)'); axa.set_ylabel('Normalized exposure')
    axa.set_xlim(0, 180); axa.set_ylim(0, 1.12)
    axa.legend(loc='upper right', fontsize=6, frameon=False)
    axa.set_title('AEP-opt lands on high exposure', fontsize=8)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 错位角分布
    axb = fig.add_subplot(gs[0, 1])
    dff = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Misalignment (°)': np.concatenate([on_df.d_full.values, vp_df.d_full.values, ba_df.d_full.values]),
    })
    sns.violinplot(data=dff, x='Group', y='Misalignment (°)', ax=axb,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.6, saturation=0.9, legend=False)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.d_full.median()
        axb.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.3, zorder=5)
        axb.text(i, 72, f'{m:.0f}°', ha='center', va='top', fontsize=7)
    axb.set_ylim(0, 78)
    axb.set_title(L_THECON + ' vs ' + L_THMIN + ' misalignment', fontsize=8)
    panel_label(axb, 'b'); style_ax(axb)

    # (c) 暴露差距比值（log 箱线，留顶部空白）
    axc = fig.add_subplot(gs[0, 2])
    ratios = [on_df.Ee.values / on_df.Emin.values, vp_df.Ee.values / vp_df.Emin.values,
              ba_df.Ee.values / ba_df.Emin.values]
    meds = [np.median(x) for x in ratios]

    def _whisker_top(r):
        q1, q3 = np.percentile(r, 25), np.percentile(r, 75)
        return min(r.max(), q3 + 1.5 * (q3 - q1))

    tops = [_whisker_top(x) for x in ratios]
    bp = axc.boxplot(ratios, positions=[1, 2, 3], widths=0.5, showfliers=False,
                     patch_artist=True, medianprops=dict(color='black', lw=1.2))
    for patch, c in zip(bp['boxes'], [C_LAND, C_VPTS, C_BAUER]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    axc.set_yscale('log')
    axc.set_ylim(0.5, max(tops) * 4)
    axc.set_xticks([1, 2, 3]); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel(L_ECON + ' / ' + L_EMIN + '  (log)')
    for i, (m, t) in enumerate(zip(meds, tops)):
        axc.text(i + 1, t * 1.8, f'{m:.0f}×', ha='center', fontsize=7.5, fontweight='bold')
    axc.set_title('AEP-opt exposure vs attainable min', fontsize=8)
    panel_label(axc, 'c'); style_ax(axc)

    # (d) 可削减比例
    axd = fig.add_subplot(gs[1, 0])
    av = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Avoidable exposure (%)': np.concatenate([on_df.avoid.values * 100,
                                                  vp_df.avoid.values * 100, ba_df.avoid.values * 100]),
    })
    sns.violinplot(data=av, x='Group', y='Avoidable exposure (%)', ax=axd,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.6, saturation=0.9, legend=False)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.avoid.median() * 100
        axd.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.3, zorder=5)
        axd.text(i, 104, f'{m:.1f}%', ha='center', va='top', fontsize=7)
    axd.set_ylim(0, 110)
    axd.set_title('Share avoidable by rotation', fontsize=8)
    panel_label(axd, 'd'); style_ax(axd)

    # (e) 空间分布
    axe = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axe, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axe.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.avoid * 100,
                         cmap=CMAP_RISK, s=1.6, alpha=0.6, edgecolors='none',
                         vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=3)
        axe.set_title('Avoidable exposure — spatial extent', fontsize=8)
        panel_label(axe, 'e')
        cax = make_axes_locatable(axe).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label('Avoidable\n(%)', fontsize=7)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=6.5)

    # (f) log-log 散点（陆上全样本）
    axf = fig.add_subplot(gs[1, 2])
    axf.scatter(on_df.Emin.values, on_df.Ee.values, s=2, c=C_LAND, alpha=0.15, edgecolors='none')
    lim = [max(on_df.Emin.min(), 1e-4), on_df.Ee.max() * 1.5]
    axf.plot(lim, lim, color='black', ls='--', lw=1.1, label='1:1 line')
    axf.set_xscale('log'); axf.set_yscale('log')
    axf.set_xlabel(L_EMIN + ' (attainable)'); axf.set_ylabel(L_ECON + ' (AEP-optimal)')
    axf.set_xlim(lim); axf.set_ylim(lim)
    axf.set_title('Systematic gap (median 602×)', fontsize=8)
    axf.legend(loc='lower right', fontsize=6.5, frameon=True, framealpha=1.0,
               edgecolor='#CCCCCC')
    panel_label(axf, 'f'); style_ax(axf)

    pdf, png = savefig(fig, 'fig2_R2')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 3 — R3（机制→量化→部分≪完整→分布→普遍性→空间）
# =====================================================================
def fig3_R3(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; cur = ctx['cur']
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (a) 机制：E(θ) + Δθ 标注
    axa = fig.add_subplot(gs[0, 0])
    r = rep_farm(on_df)
    row = on1[on1.farm_id == r.farm_id].iloc[0]
    E = on_Evec(row.spring_dir, row.autumn_dir, row.spring_conc, row.autumn_conc)
    E = E / E.max()
    te = r.theta_econ
    axa.plot(THETAS, E, color=C_LAND, lw=1.4)
    Ee = E[int(round(te)) % 180]
    axa.scatter([te], [Ee], s=30, c=C_ECON, marker='x', zorder=5, label=L_THECON)
    for dth, c in [(5, C_SPRING), (10, C_ECO), (20, C_VPTS), (30, C_BAUER)]:
        th = (te + dth) % 180
        axa.scatter([th], [E[int(round(th)) % 180]], s=22, c=c, zorder=5, label=f'{dth}°')
    axa.set_xlabel('Array orientation ' + L_THETA + ' (°)'); axa.set_ylabel('Normalized exposure')
    axa.set_xlim(0, 180); axa.set_ylim(0, 1.05)
    axa.set_title('Most reduction in the first tens of degrees', fontsize=8)
    axa.legend(loc='lower center', fontsize=5.5, frameon=False, ncol=5)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 捕获曲线（聚合中位）
    axb = fig.add_subplot(gs[0, 1])
    pts = [0, 5, 10, 20, 30]
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'), (ba_df, C_BAUER, 'Bauer')]:
        vals = [0] + [df[f'frac{d}'].median() * 100 for d in (5, 10, 20, 30)]
        axb.plot(pts, vals, 'o-', color=c, lw=1.4, markersize=3.5, label=lab)
    axb.axhline(50, color='#999999', ls=':', lw=0.9)
    axb.set_xlabel(r'$\Delta\theta$ (°)'); axb.set_ylabel('Fraction of max gain (%)')
    axb.set_xlim(0, 32); axb.set_ylim(0, 100)
    axb.set_title('Non-linear early capture', fontsize=8)
    axb.legend(loc='lower right', fontsize=6.5, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    # (c) Δθ50 / Δθ80 / 完整
    axc = fig.add_subplot(gs[0, 2])
    x = np.arange(3); w = 0.26
    d50 = [on_df.d50.median(), vp_df.d50.median(), ba_df.d50.median()]
    d80 = [on_df.d80.median(), vp_df.d80.median(), ba_df.d80.median()]
    dfl = [on_df.d_full.median(), vp_df.d_full.median(), ba_df.d_full.median()]
    axc.bar(x - w, d50, w, color=C_VPTS, label=L_DTH50)
    axc.bar(x, d80, w, color=C_ECO, label=L_DTH80)
    axc.bar(x + w, dfl, w, color=C_ECON, label='full (' + L_THMIN + ')')
    for xi, v in zip(x, d50):
        axc.text(xi - w, v + 1, f'{v:.0f}', ha='center', fontsize=8, fontweight='bold')
    for xi, v in zip(x, d80):
        axc.text(xi, v + 1, f'{v:.0f}', ha='center', fontsize=8, fontweight='bold')
    for xi, v in zip(x, dfl):
        axc.text(xi + w, v + 1, f'{v:.0f}', ha='center', fontsize=8, fontweight='bold')
    axc.set_xticks(x); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel('Orientation change (°)'); axc.set_ylim(0, 55)
    axc.set_title('Partial change ' + r'$\ll$' + ' full re-orientation', fontsize=8)
    axc.legend(loc='upper center', fontsize=6, frameon=False, ncol=3)
    panel_label(axc, 'c'); style_ax(axc)

    # (d) Δθ50 分布（陆上）
    axd = fig.add_subplot(gs[1, 0])
    axd.hist(on_df.d50.dropna(), bins=np.arange(0, 61, 3), color=C_LAND, alpha=0.85,
             edgecolor='white', lw=0.3)
    axd.axvline(on_df.d50.median(), color='black', ls='--', lw=1.0)
    axd.set_xlabel(L_DTH50 + ' (°)'); axd.set_ylabel('Number of farms')
    axd.tick_params(labelsize=8)
    axd.xaxis.label.set_size(9); axd.yaxis.label.set_size(9)
    axd.set_title(f'Median {L_DTH50} = {on_df.d50.median():.0f}°', fontsize=8)
    panel_label(axd, 'd'); style_ax(axd)

    # (e) ≤20°/≤30° 内 >50%
    axe = fig.add_subplot(gs[1, 1])
    x = np.arange(3); w = 0.36
    for thresh, off, lab in [(20, -w / 2, L_LEQ + '20°'), (30, w / 2, L_LEQ + '30°')]:
        vals = [(df[f'frac{thresh}'] > 0.5).mean() * 100 for df in (on_df, vp_df, ba_df)]
        axe.bar(x + off, vals, w, label=lab, color=(C_LAND if thresh == 20 else C_ECO))
        for xi, v in zip(x, vals):
            axe.text(xi + off, v + 1.5, f'{v:.0f}', ha='center', fontsize=6.5)
    axe.set_xticks(x); axe.set_xticklabels(GROUP_ORDER)
    axe.set_ylabel('Share of farms (%)'); axe.set_ylim(0, 88)
    axe.set_title('Early gains are widespread (>50% cut)', fontsize=8)
    axe.legend(loc='lower center', fontsize=6, frameon=False, ncol=2)
    panel_label(axe, 'e'); style_ax(axe)

    # (f) Δθ50 空间分布
    axf = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axf, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axf.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.d50.fillna(90),
                         cmap='viridis', s=1.6, alpha=0.6, edgecolors='none',
                         vmin=0, vmax=45, transform=ccrs.PlateCarree(), zorder=3)
        axf.set_title(L_DTH50 + ' — spatial extent', fontsize=8)
        panel_label(axf, 'f')
        cax = make_axes_locatable(axf).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label(L_DTH50 + ' (°)', fontsize=7)
        cb.set_ticks([0, 15, 30, 45]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=6.5)

    pdf, png = savefig(fig, 'fig3_R3')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig 4 — R4
# =====================================================================
def fig4_R4(on_df, vp_df, ba_df, ctx):
    on = ctx['on']; to = ctx['to']; vpts_ids = ctx['vpts_ids']; bauer_ids = ctx['bauer_ids']
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (a) 陆上交换散点
    axa = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    axa.scatter(on_df.aep.values + rng.normal(0, 0.006, len(on_df)),
                on_df.rr.values + rng.normal(0, 0.7, len(on_df)),
                s=2, c=C_LAND, alpha=0.12, edgecolors='none')
    axa.axvline(0.5, color='#999999', ls='--', lw=0.9)
    axa.text(0.97, 0.10, f'median RR = {on_df.rr.median():.0f}%\nmean AEP cost = {on_df.aep.mean():.2f}%',
             transform=axa.transAxes, ha='right', va='bottom', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none'))
    axa.set_xlabel('AEP cost (%)'); axa.set_ylabel('Exposure reduction (%)')
    axa.set_xlim(0, 1.0); axa.set_ylim(-5, 105)
    axa.set_title('Onshore trade-off at 1% AEP', fontsize=8)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 海上交换散点（图例加白底）
    axb = fig.add_subplot(gs[0, 1])
    axb.scatter(vp_df.aep, vp_df.rr, marker='o', s=26, c=C_VPTS, edgecolors='white', linewidths=0.4,
                label=f'VPTS (n={len(vp_df)})')
    axb.scatter(ba_df.aep, ba_df.rr, marker='o', s=26, c=C_BAUER, edgecolors='white', linewidths=0.4,
                label=f'Bauer (n={len(ba_df)})')
    axb.axvline(0.5, color='#999999', ls='--', lw=0.9)
    axb.set_xlabel('AEP cost (%)'); axb.set_xlim(0, 1.0); axb.set_ylim(-5, 105)
    axb.set_title('Offshore trade-off at 1% AEP', fontsize=8)
    axb.legend(loc='lower right', fontsize=6.5, frameon=True, framealpha=1.0,
               edgecolor='#CCCCCC')
    panel_label(axb, 'b'); style_ax(axb)

    # (c) 中位 RR vs 预算
    axc = fig.add_subplot(gs[0, 2])
    budgets = [0.005, 0.01, 0.02, 0.05]; blabels = ['0.5', '1', '2', '5']; xp = np.arange(4)

    def _agg(df_all, ids, col):
        out = []
        for b in budgets:
            sub = df_all[df_all.budget == b] if ids is None else df_all[(df_all.budget == b) & df_all.farm_id.isin(ids)]
            out.append(sub[col].median())
        return out

    axc.plot(xp, _agg(on, None, 'risk_reduction'), 'o-', color=C_LAND, lw=1.4, markersize=4, label='Onshore')
    axc.plot(xp, _agg(to, vpts_ids, 'risk_reduction_pct'), 's-', color=C_VPTS, lw=1.4, markersize=4, label='VPTS')
    axc.plot(xp, _agg(to, bauer_ids, 'risk_reduction_pct'), '^-', color=C_BAUER, lw=1.4, markersize=4, label='Bauer')
    axc.set_xticks(xp); axc.set_xticklabels(blabels)
    axc.set_xlabel('AEP budget (%)'); axc.set_ylabel('Median exposure reduction (%)')
    axc.set_ylim(40, 105)
    axc.set_title('Most gain unlocked at ' + L_LEQ + '1% AEP', fontsize=8)
    axc.legend(loc='lower right', fontsize=6.5, frameon=False)
    panel_label(axc, 'c'); style_ax(axc)

    # (d) 平均代价 vs 预算
    axd = fig.add_subplot(gs[1, 0])

    def _agga(df_all, ids, col):
        out = []
        for b in budgets:
            sub = df_all[df_all.budget == b] if ids is None else df_all[(df_all.budget == b) & df_all.farm_id.isin(ids)]
            out.append(sub[col].mean())
        return out

    axd.plot(xp, _agga(on, None, 'aep_cost_pct'), 'o-', color=C_LAND, lw=1.4, markersize=4, label='Onshore')
    axd.plot(xp, _agga(to, vpts_ids, 'aep_cost_pct'), 's-', color=C_VPTS, lw=1.4, markersize=4, label='VPTS')
    axd.plot(xp, _agga(to, bauer_ids, 'aep_cost_pct'), '^-', color=C_BAUER, lw=1.4, markersize=4, label='Bauer')
    axd.set_xticks(xp); axd.set_xticklabels(blabels)
    axd.set_xlabel('AEP budget (%)'); axd.set_ylabel('Mean AEP cost (%)')
    axd.set_ylim(0, 1.3)
    axd.set_title('Actual cost stays far below budget', fontsize=8)
    axd.legend(loc='lower right', fontsize=6.5, frameon=False)
    panel_label(axd, 'd'); style_ax(axd)

    # (e) 1% 捕获比例
    axe = fig.add_subplot(gs[1, 1])
    cap = [(on_df.rr / 100 / on_df.avoid).median() * 100,
           (vp_df.rr / 100 / vp_df.avoid).median() * 100,
           (ba_df.rr / 100 / ba_df.avoid).median() * 100]
    axe.bar([1, 2, 3], cap, color=[C_LAND, C_VPTS, C_BAUER], width=0.55, alpha=0.9)
    for i, v in enumerate(cap):
        axe.text(i + 1, v + 2, f'{v:.1f}%', ha='center', fontsize=7.5, fontweight='bold')
    axe.set_xticks([1, 2, 3]); axe.set_xticklabels(GROUP_ORDER)
    axe.set_ylabel('Max avoidable captured at 1% AEP (%)'); axe.set_ylim(0, 105)
    axe.set_title('1% AEP captures most of the gain', fontsize=8)
    panel_label(axe, 'e'); style_ax(axe)

    # (f) 2%→5% 饱和
    axf = fig.add_subplot(gs[1, 2])
    sat = fs.r43_saturation(ctx)
    vals = [sat['Onshore'], sat['VPTS'], sat['Bauer']]
    axf.bar([1, 2, 3], vals, color=[C_LAND, C_VPTS, C_BAUER], width=0.55, alpha=0.9)
    for i, v in enumerate(vals):
        axf.text(i + 1, v + 2.5, f'{v:.0f}%', ha='center', fontsize=7.5)
    axf.set_xticks([1, 2, 3]); axf.set_xticklabels(GROUP_ORDER)
    axf.set_ylabel('Farms with no new gain 2%→5% (%)'); axf.set_ylim(0, 115)
    axf.set_title('Ecological gain saturates early', fontsize=8)
    panel_label(axf, 'f'); style_ax(axf)

    pdf, png = savefig(fig, 'fig4_R4')
    plt.close(fig)
    return pdf, png


# =====================================================================
# Fig S1 — 威胁背景（数量/密度/通量/高度/时相）
# =====================================================================
def figS1_threat(ctx):
    radar = ctx['radar'].copy()
    stations = ['deess', 'behel', 'bejab', 'frabb', 'nlhrw', 'nldhl']

    def _d(season, col):
        return {r['station']: r[col] for _, r in radar[radar.season == season].iterrows()}

    sp_d = _d('spring', 'avg_density'); au_d = _d('autumn', 'avg_density')
    sp_f = _d('spring', 'total_flux'); au_f = _d('autumn', 'total_flux')
    sp_m = _d('spring', 'max_density'); au_m = _d('autumn', 'max_density')
    x = np.arange(len(stations)); w = 0.36

    hfrac, month = _vpts_height_and_month()

    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30, top=0.90)

    # (a) 站点地图（站名加白色半透明 bbox）
    axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axa, [0.5, 8.0, 49.0, 54.5])
        for sn, (la, lo) in RADAR_LOC.items():
            axa.scatter(lo, la, marker='^', s=42, c='#C0392B', edgecolors='white', linewidths=0.6,
                        transform=ccrs.PlateCarree(), zorder=5)
            axa.text(lo, la + 0.20, sn, fontsize=6.5, ha='center', va='bottom', color='#C0392B',
                     transform=ccrs.PlateCarree(), zorder=6,
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'))
        axa.set_title('Radar stations (VPTS)', fontsize=8)
        panel_label(axa, 'a')

    # (b) 平均密度（X 轴水平）
    axb = fig.add_subplot(gs[0, 1])
    axb.bar(x - w / 2, [_pos(sp_d.get(s)) for s in stations], w, color=C_SPRING, alpha=0.9)
    axb.bar(x + w / 2, [_pos(au_d.get(s)) for s in stations], w, color=C_AUTUMN, alpha=0.9)
    axb.set_yscale('log'); axb.set_ylim(3, 300)
    axb.set_xticks(x); axb.set_xticklabels(stations, rotation=0, fontsize=6)
    axb.set_ylabel('Avg density (birds km$^{-3}$)')
    axb.set_title('Migration intensity by station', fontsize=8)
    panel_label(axb, 'b'); style_ax(axb)

    # (c) 总通量（X 轴水平）
    axc = fig.add_subplot(gs[0, 2])
    axc.bar(x - w / 2, [_pos(sp_f.get(s)) for s in stations], w, color=C_SPRING, alpha=0.9)
    axc.bar(x + w / 2, [_pos(au_f.get(s)) for s in stations], w, color=C_AUTUMN, alpha=0.9)
    axc.set_yscale('log'); axc.set_ylim(1e2, 1e6)
    axc.set_xticks(x); axc.set_xticklabels(stations, rotation=0, fontsize=6)
    axc.set_ylabel('Total flux (relative)')
    axc.set_title('Integrated passage', fontsize=8)
    panel_label(axc, 'c'); style_ax(axc)

    # (d) 高度层分布（转子层入图例）
    axd = fig.add_subplot(gs[1, 0])
    bin_labels = ['0–200', '200–400', '400–600', '600–800', '800–1000', '1000–1200', '>1200']
    bx = np.arange(len(bin_labels)); bw = 0.36
    axd.bar(bx - bw / 2, hfrac['spring'], bw, color=C_SPRING, alpha=0.9)
    axd.bar(bx + bw / 2, hfrac['autumn'], bw, color=C_AUTUMN, alpha=0.9)
    axd.axvspan(-0.5, 0.5, color=C_VPTS, alpha=0.15, zorder=0)
    axd.set_xticks(bx); axd.set_xticklabels(bin_labels, rotation=45, ha='right')
    axd.set_ylabel('Density share (%)'); axd.set_xlabel('Flight height (m)')
    axd.set_ylim(0, max(hfrac['spring'] + hfrac['autumn']) * 1.18)
    axd.set_title('Flight-height distribution', fontsize=8)
    panel_label(axd, 'd'); style_ax(axd)

    # (e) 最大密度
    axe = fig.add_subplot(gs[1, 1])
    axe.bar(x - w / 2, [_pos(sp_m.get(s)) for s in stations], w, color=C_SPRING, alpha=0.9)
    axe.bar(x + w / 2, [_pos(au_m.get(s)) for s in stations], w, color=C_AUTUMN, alpha=0.9)
    axe.set_yscale('log'); axe.set_ylim(3, 3000)
    axe.set_xticks(x); axe.set_xticklabels(stations, rotation=0, fontsize=6)
    axe.set_ylabel('Max density (birds km$^{-3}$)')
    axe.set_title('Peak intensity', fontsize=8)
    panel_label(axe, 'e'); style_ax(axe)

    # (f) 月相
    axf = fig.add_subplot(gs[1, 2])
    months = list(range(1, 13))
    sp_mo = [month['spring'].get(m, 0) for m in months]
    au_mo = [month['autumn'].get(m, 0) for m in months]
    axf.bar([m - 0.18 for m in months], sp_mo, 0.36, color=C_SPRING, alpha=0.9)
    axf.bar([m + 0.18 for m in months], au_mo, 0.36, color=C_AUTUMN, alpha=0.9)
    axf.set_xticks(months)
    axf.set_xlabel('Month'); axf.set_ylabel('Total flux (relative)')
    axf.set_title('Passage timing (sampled Apr–May / Sep–Oct)', fontsize=8)
    panel_label(axf, 'f'); style_ax(axf)

    # 图级图例（一次，避免逐 panel 遮挡；含转子层说明）
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_SPRING, alpha=0.9, edgecolor='none'),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_AUTUMN, alpha=0.9, edgecolor='none'),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_VPTS, alpha=0.15, edgecolor='none'),
    ]
    fig.legend(handles, ['Spring', 'Autumn', 'Rotor layer (0–200 m)'],
               loc='upper center', bbox_to_anchor=(0.5, 0.985), ncol=3,
               fontsize=7, frameon=False)

    pdf, png = savefig(fig, 'figS1_threat')
    plt.close(fig)
    return pdf, png


def main():
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}, cartopy={HAS_CARTOPY}')

    print('Fig 0 study area ...'); fig0_study_area(ctx)
    print('Fig 0 overview ...'); fig0_overview(on_df, vp_df, ba_df, ctx)
    print('Fig 1 R1 ...'); fig1_R1(on_df, vp_df, ba_df, ctx)
    print('Fig 2 R2 ...'); fig2_R2(on_df, vp_df, ba_df, ctx)
    print('Fig 3 R3 ...'); fig3_R3(on_df, vp_df, ba_df, ctx)
    print('Fig 4 R4 ...'); fig4_R4(on_df, vp_df, ba_df, ctx)
    print('Fig S1 threat ...'); figS1_threat(ctx)

    print('\nDone. Outputs in figures_v2/:')
    for f in sorted(os.listdir(fs.FIG)):
        p = os.path.join(fs.FIG, f)
        print(f'  {f}  ({os.path.getsize(p)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()