# -*- coding: utf-8 -*-
"""
generate_paper_figures_v3.py —— 照抄参考稿《Unlocking Solar Poverty Alleviation…》配图版式。

参考稿 5 张主图版式（据「子刊图片讲解」）：
  图1  1×3 图形摘要：Predicament | 3D/机制示意图 | Solution + 底部 3 段加粗问题
  图2  2×3，共 4 子图：地图 | 分组柱 | 对比小提琴 | 堆积柱
  图3  混合 3 子图：极坐标玫瑰 | 散点矩阵(带边缘直方图) | 宽小提琴
  图4  1×4，4 幅地图，每幅带放大 inset + 分组柱
  图5  上半 6 幅地图(统一色条) + 下半 2 幅折线(红框标注)

我们的 5 张主图对应：
  fig1_framework  研究框架图（Predicament | 机制 E(θ) | Solution + 底部三问）
  fig2_R1         方向敏感性（rel 地图 | rel+avoid 分组柱 | 陆上vs海上小提琴 | 分级堆积柱）
  fig3_R2         错位（玫瑰 | 3×3 散点矩阵 | d_full 宽小提琴）
  fig4_spatial    空间总览（研究区 | 方向场 | 可削减 | Δθ50，各带北海放大 + 三组柱）
  fig5_R34        交换前沿（3组×2预算 6 幅 RR 地图 + 边际代价曲线）
  figS1_threat    威胁背景（沿用 v2，不变）

所有数字来自 figure_style.compute_metrics()，与 final_numbers.txt 一致。
图内文字英文；mathtext 一律 raw string；色盲安全 Okabe-Ito 色板；矢量(PDF)+PNG 双导出。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN, C_ECON, C_ECO,
                          GROUP_COLORS, GROUP_ORDER, THETAS, CMAP_CONC, CMAP_RISK,
                          style_ax, panel_label, circ, on_Evec, off_Evec)

# 复用 v2 的底图/雷达/代表场/高度月相/标签常量（同源，不重复造）
from generate_paper_figures_v2 import (add_basemap, RADAR_LOC, HEIGHT_STATIONS,
                                       rep_farm, rep_curve, _pos, _vpts_height_and_month,
                                       L_THECON, L_THMIN, L_ECON, L_EMIN, L_DTH50, L_DTH80,
                                       L_LEQ, L_THETA, HAS_CARTOPY, figS1_threat)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
FIG3 = os.path.join(BASE, '..', 'figures_v3')
os.makedirs(FIG3, exist_ok=True)


def savefig3(fig, name):
    pdf = os.path.join(FIG3, name + '.pdf')
    png = os.path.join(FIG3, name + '.png')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=300)
    return pdf, png


# ---------------------------------------------------------------------------
# 通用取数辅助
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
    """三组中位 RR（%）随预算变化。"""
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
# 图1 — 研究框架图（1×3 图形摘要，无 a/b/c）
# =====================================================================
def fig1_framework(on_df, vp_df, ba_df, ctx):
    """研究框架（图形摘要）：三幅极简示意——朝向几何 | E(θ) 机制 | 小代价大收益。"""
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(14.0, 4.7))
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.97, top=0.80, bottom=0.10, wspace=0.24)

    # ---- (a) 朝向几何：平行 vs 垂直（俯视示意）----
    axa = fig.add_subplot(gs[0, 0])
    axa.set_xlim(0, 10); axa.set_ylim(-0.6, 10); axa.set_aspect('equal')
    axa.axis('off')

    def _farm(x0, y0, w, h, rows_vertical):
        axa.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor='#444444', lw=1.0))
        if rows_vertical:
            xs = np.linspace(x0 + 0.9, x0 + w - 0.9, 3)
            for xi in xs:
                ys = np.linspace(y0 + 0.5, y0 + h - 0.5, 6)
                axa.plot([xi] * len(ys), ys, 'o', color=C_ECON, markersize=3.6, zorder=3)
        else:
            ys = np.linspace(y0 + 0.9, y0 + h - 0.9, 3)
            for yi in ys:
                xs = np.linspace(x0 + 0.5, x0 + w - 0.5, 6)
                axa.plot(xs, [yi] * len(xs), 'o', color=C_ECON, markersize=3.6, zorder=3)

    # 左：行与迁徙方向平行（候鸟沿行间空道通过）
    _farm(0.4, 0.6, 3.0, 8.6, rows_vertical=True)
    axa.annotate('', xy=(1.9, 9.7), xytext=(1.9, 0.3),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=2.2, mutation_scale=18))
    # 右：行与迁徙方向垂直（候鸟反复穿越旋翼面）
    _farm(6.6, 0.6, 3.0, 8.6, rows_vertical=False)
    axa.annotate('', xy=(8.1, 9.7), xytext=(8.1, 0.3),
                 arrowprops=dict(arrowstyle='-|>', color=C_SPRING, lw=2.2, mutation_scale=18))

    axa.text(1.9, 9.95, 'migration', fontsize=8, color=C_SPRING, ha='center', va='bottom')
    axa.text(1.9, -0.35, 'parallel → low exposure', fontsize=8, ha='center', va='top', fontweight='bold')
    axa.text(8.1, -0.35, 'perpendicular → high exposure', fontsize=8, ha='center', va='top', fontweight='bold')
    panel_label(axa, 'a')

    # ---- (b) E(θ) 机制（概念曲线，不标具体数值）----
    axb = fig.add_subplot(gs[0, 1])
    th = np.linspace(0, 180, 400)
    E = np.sin(np.radians(th)) ** 2
    E = E / E.max()
    axb.plot(th, E, color='#333333', lw=1.8)
    te, tm = 45.0, 0.0          # AEP-opt（高暴露） / eco-opt（低暴露）
    Ete, Etm = np.sin(np.radians(te)) ** 2, 0.0
    axb.axvline(te, color=C_ECON, ls='--', lw=1.1)
    axb.axvline(tm, color=C_ECO, ls='-', lw=1.3)
    axb.scatter([te], [Ete], s=46, c=C_ECON, marker='x', zorder=5)
    axb.scatter([tm], [Etm], s=46, c=C_ECO, marker='o', zorder=5)
    axb.annotate('', xy=(te, 0.74), xytext=(tm, 0.74),
                 arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
    axb.text((te + tm) / 2, 0.78, 'misalignment', ha='center', fontsize=7.5, color='#333333')
    axb.annotate('AEP-opt', xy=(te, Ete), xytext=(te + 30, 0.98), fontsize=7.5,
                 color=C_ECON, ha='center', arrowprops=dict(arrowstyle='->', color=C_ECON, lw=0.9))
    axb.annotate('eco-opt', xy=(tm, Etm), xytext=(tm + 26, 0.26), fontsize=7.5,
                 color=C_ECO, ha='center', arrowprops=dict(arrowstyle='->', color=C_ECO, lw=0.9))
    axb.set_xlabel('Array orientation ' + L_THETA + ' (°)', fontsize=8)
    axb.set_ylabel('Exposure', fontsize=8)
    axb.set_xlim(0, 180); axb.set_ylim(0, 1.14)
    axb.set_yticks([0, 0.5, 1.0])
    axb.tick_params(labelsize=7)
    axb.set_title('Exposure peaks where rows cross migration', fontsize=9)
    style_ax(axb)
    panel_label(axb, 'b')

    # ---- (c) 小代价大收益（单一对比）----
    axc = fig.add_subplot(gs[0, 2])
    axc.axis('off')
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    axc.text(0.5, 0.70, '~1% AEP loss', fontsize=14, ha='center', va='center',
             fontweight='bold', color=C_ECON)
    axc.text(0.5, 0.44, '→', fontsize=26, ha='center', va='center', color='#555555')
    axc.text(0.5, 0.16, '~97% exposure cut', fontsize=14, ha='center', va='center',
             fontweight='bold', color=C_ECO)
    panel_label(axc, 'c')

    # 因果箭头
    fig.canvas.draw()
    b0 = axa.get_position(); b1 = axb.get_position(); b2 = axc.get_position()
    y0 = b0.y0 + 0.5 * b0.height
    for x0, x1 in [(b0.x1, b1.x0), (b1.x1, b2.x0)]:
        fig.add_artist(FancyArrowPatch((x0 + 0.010, y0), (x1 - 0.010, y0),
                                       transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=18, color='#555555', lw=1.4, zorder=6))

    pdf, png = savefig3(fig, 'fig1_framework')
    plt.close(fig)
    return pdf, png


# =====================================================================
# 图2 — R1 方向敏感性（2×3 共 4 子图）
# =====================================================================
def fig2_R1(on_df, vp_df, ba_df, ctx):
    """R1 主图（图3）：方向结构 + 代表曲线 + 敏感性量级，三面板回答「暴露是否敏感」。"""
    on1 = ctx['on1']; cur = ctx['cur']; grid = ctx['grid']
    gdir = grid.dropna(subset=['spring_dir', 'autumn_dir']).copy()

    fig = plt.figure(figsize=(11.6, 3.6))
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.97, top=0.80, bottom=0.16, wspace=0.30)

    # (a) 玫瑰：迁徙方向结构（R1.1）
    axa = fig.add_subplot(gs[0, 0], projection='polar')
    bins = np.arange(0, 361, 15)
    sp_hist, _ = np.histogram(gdir.spring_dir.values, bins=bins)
    au_hist, _ = np.histogram(gdir.autumn_dir.values, bins=bins)
    tcent = np.radians((bins[:-1] + bins[1:]) / 2)
    axa.set_theta_zero_location('N'); axa.set_theta_direction('clockwise')
    axa.bar(tcent, sp_hist, width=np.radians(15), color=C_SPRING, alpha=0.6, zorder=2)
    axa.bar(tcent, au_hist, width=np.radians(15), color=C_AUTUMN, alpha=0.6, zorder=2)
    axa.set_ylim(0, 1.0); axa.set_yticks([])
    axa.tick_params(labelsize=7, pad=0.5)
    axa.set_title('Directional structure of migration', fontsize=9, pad=10)
    axa.legend([Line2D([], [], color=C_SPRING, lw=5, alpha=0.6),
                Line2D([], [], color=C_AUTUMN, lw=5, alpha=0.6)],
               ['Spring', 'Autumn'], loc='upper right', bbox_to_anchor=(1.28, 1.12),
               fontsize=7.5, frameon=False)
    panel_label(axa, 'a')

    # (b) E(θ) 代表曲线（R1.2：平行→垂直暴露单调上升，三组）
    axb = fig.add_subplot(gs[0, 1])
    for df, c in [(on_df, C_LAND), (vp_df, C_VPTS), (ba_df, C_BAUER)]:
        _, E, _ = rep_curve(df, on1, cur)
        axb.plot(THETAS, E, color=c, lw=1.2, alpha=0.9)
    axb.set_xlabel('Array orientation ' + L_THETA + ' (°)', fontsize=8)
    axb.set_ylabel('Normalized exposure', fontsize=8)
    axb.set_xlim(0, 180); axb.set_ylim(0, 1.05)
    axb.tick_params(labelsize=7)
    axb.set_title('Exposure vs orientation (rep. farm)', fontsize=9)
    from matplotlib.lines import Line2D as _L2
    axb.legend([_L2([], [], color=c, lw=1.2) for c in (C_LAND, C_VPTS, C_BAUER)],
               ['Onshore', 'VPTS', 'Bauer'], loc='upper center', fontsize=7, frameon=False, ncol=3)
    panel_label(axb, 'b'); style_ax(axb)

    # (c) 分组柱：三组 rel 中位（R1.2）
    axc = fig.add_subplot(gs[0, 2])
    x = np.arange(3); w = 0.5
    rel = [on_df.rel.median() * 100, vp_df.rel.median() * 100, ba_df.rel.median() * 100]
    axc.bar(x, rel, w, color=[C_LAND, C_VPTS, C_BAUER], alpha=0.85)
    for xi, v in zip(x, rel):
        axc.text(xi, v + 1.5, f'{v:.1f}', ha='center', fontsize=8)
    axc.set_xticks(x); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel('Median rel. change (%)'); axc.set_ylim(0, 108)
    axc.set_title('Sensitivity magnitude (median)', fontsize=9)
    panel_label(axc, 'c'); style_ax(axc)

    pdf, png = savefig3(fig, 'fig2_R1')
    plt.close(fig)
    return pdf, png


def figS2_R1(on_df, vp_df, ba_df, ctx):
    """R1 补充图（图S2）：敏感性在空间上的普遍性（空间分布 + 分布箱线 + 累积占比）。"""
    od = ctx['od']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']

    fig = plt.figure(figsize=(13.0, 3.9))
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.98, top=0.80, bottom=0.14, wspace=0.26)

    # (a) 地图：陆上 rel 着色 + 海上 55 + 雷达（单格）
    axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axa, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axa.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.rel * 100,
                         cmap=CMAP_RISK, s=1.4, alpha=0.6, edgecolors='none',
                         vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=3)
        axa.scatter(vpts.centroid_lon, vpts.centroid_lat, s=24, c=C_VPTS, marker='s',
                    edgecolors='white', linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)
        axa.scatter(bauer.centroid_lon, bauer.centroid_lat, s=24, c=C_BAUER, marker='s',
                    edgecolors='white', linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)
        for sn, (la, lo) in RADAR_LOC.items():
            axa.scatter(lo, la, marker='^', s=36, c='#C0392B', edgecolors='white',
                        linewidths=0.5, transform=ccrs.PlateCarree(), zorder=5)
        axa.set_title('Orientation sensitivity (rel. change, %)', fontsize=9)
        panel_label(axa, 'a')
        cax = make_axes_locatable(axa).append_axes('right', size='4%', pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label('Rel. change (%)', fontsize=7)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=7.5)

    # (b) 箱线图：三组 rel 分布（纵轴放大到有效区间）
    axb = fig.add_subplot(gs[0, 1])
    bp = axb.boxplot([on_df.rel.values * 100, vp_df.rel.values * 100, ba_df.rel.values * 100],
                     positions=[0, 1, 2], widths=0.5, patch_artist=True,
                     medianprops=dict(color='black', lw=1.2),
                     whiskerprops=dict(color='#666666', lw=0.8),
                     capprops=dict(color='#666666', lw=0.8),
                     flierprops=dict(marker='o', markersize=2, markeredgewidth=0, color='#999999'))
    for patch, c in zip(bp['boxes'], [C_LAND, C_VPTS, C_BAUER]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.rel.median() * 100
        axb.text(i, m + 0.4, f'{m:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    axb.set_xticks([0, 1, 2]); axb.set_xticklabels(GROUP_ORDER)
    axb.set_ylim(85, 102)
    axb.set_ylabel('Rel. change (%)')
    axb.set_title('Sensitivity: three groups', fontsize=9)
    panel_label(axb, 'b'); style_ax(axb)

    # (c) 普遍性：超过给定 rel 阈值的场址占比（CDF 式）
    axc = fig.add_subplot(gs[0, 2])
    thr = np.arange(90, 101, 1)
    for df, c, lab in [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'),
                       (ba_df, C_BAUER, 'Bauer')]:
        vals = [(df.rel.values * 100 >= t).mean() * 100 for t in thr]
        axc.plot(thr, vals, 'o-', color=c, lw=1.3, markersize=3, label=lab)
    axc.axhline(100, color='#999999', ls=':', lw=0.8)
    axc.set_xlabel('Rel. change threshold (%)')
    axc.set_ylabel('Share of farms above (%)')
    axc.set_xlim(90, 100); axc.set_ylim(0, 105)
    axc.set_xticks([90, 92, 94, 96, 98, 100])
    axc.set_title('Universality of high sensitivity', fontsize=9)
    axc.legend(loc='lower left', fontsize=7, frameon=False)
    panel_label(axc, 'c'); style_ax(axc)

    pdf, png = savefig3(fig, 'figS2_R1')
    plt.close(fig)
    return pdf, png


# =====================================================================
# 图3 — R2 错位（玫瑰 | 3×3 散点矩阵 | 宽小提琴）
# =====================================================================
def fig3_R2(on_df, vp_df, ba_df, ctx):
    # R2-only：d_full（R2.1）、avoid（R2.3）；散点矩阵 θ_econ/θ_min/d_full（R2.1）
    fig = plt.figure(figsize=(9.4, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], hspace=0.34, wspace=0.32)

    # (a) 左上：可削减暴露比例 avoid（R2.3）
    axa = fig.add_subplot(gs[0, 0])
    dfa = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Avoidable (%)': np.concatenate([on_df.avoid.values * 100, vp_df.avoid.values * 100,
                                         ba_df.avoid.values * 100]),
    })
    sns.violinplot(data=dfa, x='Group', y='Avoidable (%)', ax=axa,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.6, saturation=0.9, legend=False)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.avoid.median() * 100
        axa.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.3, zorder=5)
        axa.text(i, m + 2, f'{m:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    axa.set_ylim(0, 112)
    axa.set_ylabel('Avoidable exposure (%)')
    axa.set_title('Share avoidable by re-orientation', fontsize=9)
    panel_label(axa, 'a'); style_ax(axa)

    # (c) 左下：d_full 宽小提琴（R2.1），中位黑线 + 均值蓝线
    axc = fig.add_subplot(gs[1, 0])
    dff = pd.DataFrame({
        'Group': ['Onshore'] * len(on_df) + ['VPTS'] * len(vp_df) + ['Bauer'] * len(ba_df),
        'Misalignment (°)': np.concatenate([on_df.d_full.values, vp_df.d_full.values,
                                            ba_df.d_full.values]),
    })
    sns.violinplot(data=dff, x='Group', y='Misalignment (°)', ax=axc,
                   order=GROUP_ORDER, hue='Group', palette=GROUP_COLORS,
                   inner=None, linewidth=0.6, saturation=0.9, legend=False)
    for i, df in enumerate([on_df, vp_df, ba_df]):
        m = df.d_full.median()
        mu = df.d_full.mean()
        axc.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.3, zorder=5)
        axc.hlines(mu, i - 0.28, i + 0.28, color='#1B7AC6', lw=1.3, ls='--', zorder=5)
        axc.text(i, 88, f'{m:.0f}°', ha='center', va='top', fontsize=8, fontweight='bold')
        axc.text(i, 79, f'μ={mu:.0f}°', ha='center', va='top', fontsize=7.5, color='#1B7AC6')
    axc.set_ylim(0, 95)
    axc.set_ylabel('Misalignment (°)')
    axc.set_title(L_THECON + ' vs ' + L_THMIN + ' misalignment', fontsize=9)
    from matplotlib.lines import Line2D as _L
    axc.legend([_L([], [], color='black', lw=1.3), _L([], [], color='#1B7AC6', lw=1.3, ls='--')],
               ['median', 'mean'], loc='upper left', fontsize=7.5, frameon=False)
    panel_label(axc, 'c'); style_ax(axc)

    # (b) 右列：θ_min vs θ_econ 散点 + 恒等线（错位的联合证据）
    axb = fig.add_subplot(gs[0:2, 1])
    axb.plot([0, 180], [0, 180], color='#999999', ls='--', lw=1.2, zorder=1)
    axb.text(150, 138, 'aligned', fontsize=7.5, color='#777777', ha='center', va='bottom')
    for df, c in [(on_df, C_LAND), (vp_df, C_VPTS), (ba_df, C_BAUER)]:
        axb.scatter(df.theta_econ, df.th_min, s=7, c=c, alpha=0.45,
                    edgecolors='none', rasterized=True)
    axb.set_xlabel(r'$	heta_{\mathrm{econ}}$  (AEP-opt, °)', fontsize=8)
    axb.set_ylabel(r'$	heta_{\mathrm{min}}$  (min-exposure, °)', fontsize=8)
    axb.set_xlim(0, 180); axb.set_ylim(0, 180)
    axb.set_xticks([0, 45, 90, 135, 180]); axb.set_yticks([0, 45, 90, 135, 180])
    axb.set_aspect('equal')
    axb.set_title('Energy-opt vs eco-opt orientation', fontsize=9)
    axb.legend(GROUP_ORDER, loc='upper left', fontsize=7.5, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    pdf, png = savefig3(fig, 'fig3_R2')
    plt.close(fig)
    return pdf, png


# =====================================================================
# 图4 — 空间总览（1×4 地图，各带北海放大 + 三组柱）
# =====================================================================
def _overlay_inset(fig, ax, extent, show):
    """在 ax 左下角叠加一个小地图 inset，返回 inset 轴。"""
    pos = ax.get_position()
    axins = fig.add_axes([pos.x0 + 0.015, pos.y0 + 0.095, 0.42 * pos.width, 0.42 * pos.height],
                         projection=ccrs.PlateCarree())
    add_basemap(axins, extent)
    show(axins)
    for s in axins.spines.values():
        s.set_edgecolor('white'); s.set_linewidth(1.4)
    return axins


def _draw_bar(ax, vals, labels, colors, ylabel, ymax, log=False):
    """在常规轴上画三组柱状图（图4 第二排）。"""
    xx = np.arange(len(vals))
    ax.bar(xx, vals, color=colors, width=0.6, alpha=0.9)
    if log:
        ax.set_yscale('log'); ax.set_ylim(1, 10000)
        ax.set_yticks([10, 100, 1000, 10000])
        for xi, v in zip(xx, vals):
            ax.text(xi, v * 1.5, f'{v:.0f}', ha='center', fontsize=7.5)
    else:
        for xi, v in zip(xx, vals):
            ax.text(xi, v + ymax * 0.03, f'{v:.0f}', ha='center', fontsize=7.5)
        ax.set_ylim(0, ymax)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return ax


def fig4_spatial(on_df, vp_df, ba_df, ctx):
    on1 = ctx['on1']; od = ctx['od']; grid = ctx['grid']
    vpts = od[od.source == 'VPTS']; bauer = od[od.source == 'Bauer_grid']
    NORTHSEA = [-3.5, 9.5, 49.5, 56.5]

    fig = plt.figure(figsize=(16.0, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.0], hspace=0.22, wspace=0.16,
                          left=0.03, right=0.97, top=0.91, bottom=0.06)

    def _inset_offshore(axins):
        axins.scatter(vpts.centroid_lon, vpts.centroid_lat, s=22, c=C_VPTS, marker='s',
                      edgecolors='white', linewidths=0.4, transform=ccrs.PlateCarree(), zorder=4)
        axins.scatter(bauer.centroid_lon, bauer.centroid_lat, s=22, c=C_BAUER, marker='s',
                      edgecolors='white', linewidths=0.4, transform=ccrs.PlateCarree(), zorder=4)
        for sn, (la, lo) in RADAR_LOC.items():
            axins.scatter(lo, la, marker='^', s=30, c='#C0392B', edgecolors='white',
                          linewidths=0.4, transform=ccrs.PlateCarree(), zorder=5)

    # --- 第一排 (a) 研究区总览 ---
    axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axa, [-12, 20, 39, 59])
        gx = [grid.lon.min(), grid.lon.max(), grid.lon.max(), grid.lon.min(), grid.lon.min()]
        gy = [grid.lat.min(), grid.lat.min(), grid.lat.max(), grid.lat.max(), grid.lat.min()]
        axa.plot(gx, gy, color='#333333', lw=1.0, ls='--', transform=ccrs.PlateCarree(), zorder=2)
        axa.scatter(on1.centroid_lon, on1.centroid_lat, s=0.6, c=C_LAND, alpha=0.3,
                    edgecolors='none', transform=ccrs.PlateCarree(), zorder=3)
        axa.scatter(vpts.centroid_lon, vpts.centroid_lat, s=14, c=C_VPTS, edgecolors='white',
                    linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
        axa.scatter(bauer.centroid_lon, bauer.centroid_lat, s=14, c=C_BAUER, edgecolors='white',
                    linewidths=0.3, transform=ccrs.PlateCarree(), zorder=4)
        axa.set_title('Study region', fontsize=9)
        panel_label(axa, 'a')
        _overlay_inset(fig, axa, NORTHSEA, _inset_offshore)

    # --- 第一排 (b) 方向场（Bauer 春季 quiver） ---
    axb = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        gsub = grid[(grid.row % 2 == 0) & (grid.col % 2 == 0)]
        extent = [grid.lon.min() - 0.5, grid.lon.max() + 0.5, grid.lat.min() - 0.5, grid.lat.max() + 0.5]
        add_basemap(axb, extent)
        u = np.sin(np.radians(gsub.spring_dir.values)); v = np.cos(np.radians(gsub.spring_dir.values))
        sc = axb.quiver(gsub.lon.values, gsub.lat.values, u, v, gsub.spring_conc.values,
                        cmap=CMAP_CONC, scale=44, width=0.0032, headwidth=2.8, headlength=3.2,
                        transform=ccrs.PlateCarree(), zorder=3)
        sc.set_clim(0.5, 0.9)
        axb.set_title('Spring migration direction field', fontsize=9)
        panel_label(axb, 'b')
        cax = make_axes_locatable(axb).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label('Concentration', fontsize=7)
        cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=7.5)

        def _inset_field(axins):
            gsub2 = grid[(grid.row % 2 == 0) & (grid.col % 2 == 0) &
                         (grid.lon >= NORTHSEA[0]) & (grid.lon <= NORTHSEA[1]) &
                         (grid.lat >= NORTHSEA[2]) & (grid.lat <= NORTHSEA[3])]
            u2 = np.sin(np.radians(gsub2.spring_dir.values)); v2 = np.cos(np.radians(gsub2.spring_dir.values))
            q = axins.quiver(gsub2.lon.values, gsub2.lat.values, u2, v2, gsub2.spring_conc.values,
                             cmap=CMAP_CONC, scale=22, width=0.004, headwidth=2.2, headlength=2.6,
                             transform=ccrs.PlateCarree(), zorder=3)
            q.set_clim(0.5, 0.9)
        _overlay_inset(fig, axb, NORTHSEA, _inset_field)

    # --- 第一排 (c) 可削减暴露（avoid） ---
    axc = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axc, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axc.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.avoid * 100,
                         cmap=CMAP_RISK, s=1.6, alpha=0.6, edgecolors='none',
                         vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=3)
        axc.set_title('Avoidable exposure (%)', fontsize=9)
        panel_label(axc, 'c')
        cax = make_axes_locatable(axc).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label('Avoidable (%)', fontsize=7)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=7.5)

        off_all = pd.concat([vp_df, ba_df])
        offm = _off_metric(od, off_all, 'avoid')

        def _inset_avoid(axins):
            sc2 = axins.scatter(offm.centroid_lon, offm.centroid_lat, c=offm.avoid * 100,
                                cmap=CMAP_RISK, s=40, edgecolors='white', linewidths=0.4,
                                vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=4)
            return sc2
        _overlay_inset(fig, axc, NORTHSEA, _inset_avoid)

    # --- 第一排 (d) Δθ50 ---
    axd = fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
    if HAS_CARTOPY:
        add_basemap(axd, [on_df.centroid_lon.min() - 0.5, on_df.centroid_lon.max() + 0.5,
                          on_df.centroid_lat.min() - 0.5, on_df.centroid_lat.max() + 0.5])
        sc = axd.scatter(on_df.centroid_lon, on_df.centroid_lat, c=on_df.d50.fillna(90),
                         cmap='viridis', s=1.6, alpha=0.6, edgecolors='none',
                         vmin=0, vmax=45, transform=ccrs.PlateCarree(), zorder=3)
        axd.set_title(L_DTH50 + ' (°)', fontsize=9)
        panel_label(axd, 'd')
        cax = make_axes_locatable(axd).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
        cb = fig.colorbar(sc, cax=cax); cb.set_label(L_DTH50 + ' (°)', fontsize=7)
        cb.set_ticks([0, 15, 30, 45]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=7.5)

        offm = _off_metric(od, pd.concat([vp_df, ba_df]), 'd50')

        def _inset_d50(axins):
            axins.scatter(offm.centroid_lon, offm.centroid_lat, c=offm.d50.fillna(90),
                          cmap='viridis', s=40, edgecolors='white', linewidths=0.4,
                          vmin=0, vmax=45, transform=ccrs.PlateCarree(), zorder=4)
        _overlay_inset(fig, axd, NORTHSEA, _inset_d50)

    # --- 第二排：三组柱状图（与上方地图一一对应） ---
    sp_ax = grid.spring_dir.median() % 180
    au_ax = grid.autumn_dir.median() % 180

    axbar_a = fig.add_subplot(gs[1, 0])
    _draw_bar(axbar_a, [len(on1), len(vpts), len(bauer)], ['On', 'VP', 'Ba'],
              [C_LAND, C_VPTS, C_BAUER], 'n farms', 4500, log=True)

    axbar_b = fig.add_subplot(gs[1, 1])
    _draw_bar(axbar_b, [sp_ax, au_ax], ['Spr', 'Aut'],
              [C_SPRING, C_AUTUMN], 'axis (°)', 90)

    axbar_c = fig.add_subplot(gs[1, 2])
    _draw_bar(axbar_c, [on_df.avoid.median() * 100, vp_df.avoid.median() * 100,
                        ba_df.avoid.median() * 100],
              ['On', 'VP', 'Ba'], [C_LAND, C_VPTS, C_BAUER], 'avoid (%)', 110)

    axbar_d = fig.add_subplot(gs[1, 3])
    _draw_bar(axbar_d, [on_df.d50.median(), vp_df.d50.median(), ba_df.d50.median()],
              ['On', 'VP', 'Ba'], [C_LAND, C_VPTS, C_BAUER], L_DTH50 + ' (°)', 55)

    # 图级图例（一次）
    handles = [
        Line2D([], [], marker='o', ls='', color=C_LAND, markersize=5, label=f'Onshore (n={len(on1)})'),
        Line2D([], [], marker='s', ls='', color=C_VPTS, markersize=5, label=f'Offshore VPTS (n={len(vpts)})'),
        Line2D([], [], marker='s', ls='', color=C_BAUER, markersize=5, label=f'Offshore Bauer (n={len(bauer)})'),
        Line2D([], [], marker='^', ls='', color='#C0392B', markersize=6, label='Radar station'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.985),
               ncol=4, fontsize=7.5, frameon=False)

    pdf, png = savefig3(fig, 'fig4_spatial')
    plt.close(fig)
    return pdf, png


# =====================================================================
# 图5 — R3+R4（6 幅 RR 地图 + 捕获曲线 + 边际代价曲线）
# =====================================================================
def fig5_R34(on_df, vp_df, ba_df, ctx):
    BUDGETS = [0.005, 0.01, 0.02, 0.05]
    BLABELS = ['0.5', '1', '2', '5']
    XP = np.arange(len(BUDGETS))

    on_b1 = _on_rr_at(ctx, 0.01); on_b5 = _on_rr_at(ctx, 0.05)
    off_b1 = _off_rr_at(ctx, 0.01); off_b5 = _off_rr_at(ctx, 0.05)

    def _split(m):
        return m[m.source == 'VPTS'], m[m.source == 'Bauer_grid']

    vp_b1, ba_b1 = _split(off_b1)
    vp_b5, ba_b5 = _split(off_b5)

    map_rows = [
        ('Onshore', '1% AEP', on_b1), ('Onshore', '5% AEP', on_b5),
        ('VPTS', '1% AEP', vp_b1), ('VPTS', '5% AEP', vp_b5),
        ('Bauer', '1% AEP', ba_b1), ('Bauer', '5% AEP', ba_b5),
    ]

    fig = plt.figure(figsize=(13.5, 9.0))
    gsm = fig.add_gridspec(2, 3, left=0.05, right=0.86, top=0.95, bottom=0.42,
                           hspace=0.16, wspace=0.12)
    gsl = fig.add_gridspec(1, 2, left=0.10, right=0.90, top=0.34, bottom=0.08, wspace=0.28)

    axes = []
    scs = []
    for k, (grp, bud, m) in enumerate(map_rows):
        ax = fig.add_subplot(gsm[k // 3, k % 3], projection=ccrs.PlateCarree()) if HAS_CARTOPY else None
        axes.append(ax)
        if not HAS_CARTOPY:
            continue
        col = 'risk_reduction' if grp == 'Onshore' else 'risk_reduction_pct'
        s = 1.6 if grp == 'Onshore' else 46
        if grp == 'Onshore':
            add_basemap(ax, [m.centroid_lon.min() - 0.5, m.centroid_lon.max() + 0.5,
                             m.centroid_lat.min() - 0.5, m.centroid_lat.max() + 0.5])
            sc = ax.scatter(m.centroid_lon, m.centroid_lat, c=m[col], cmap=CMAP_RISK,
                            s=s, alpha=0.6, edgecolors='none', vmin=0, vmax=100,
                            transform=ccrs.PlateCarree(), zorder=3)
        else:
            add_basemap(ax, [-3.5, 9.5, 49.5, 56.5])
            sc = ax.scatter(m.centroid_lon, m.centroid_lat, c=m[col], cmap=CMAP_RISK,
                            s=s, edgecolors='white', linewidths=0.4, vmin=0, vmax=100,
                            transform=ccrs.PlateCarree(), zorder=4)
        scs.append(sc)
        ax.set_title(f'{grp} — {bud} budget', fontsize=8.5)
        panel_label(ax, chr(ord('a') + k))

    # 统一色条（右侧）
    if HAS_CARTOPY:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(0, 100), cmap=CMAP_RISK)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, shrink=0.9, pad=0.03)
        cb.set_label('Exposure reduction (%)', fontsize=8)
        cb.set_ticks([0, 25, 50, 75, 100]); cb.ax.tick_params(width=0.6, length=2.5, labelsize=7)

    # (g, h) 边际代价：拆为两个单轴面板（dataviz 禁双轴），红框 ≤1%
    #   (g) 中位暴露降 vs AEP 预算；(h) 平均 AEP 代价 vs AEP 预算
    _, rr_b = _median_rr_vs_budget(ctx)
    _, ae_b = _mean_aep_vs_budget(ctx)
    rr_vals = np.array([[rr_b[b][i] for b in BUDGETS] for i in range(3)])
    ae_vals = np.array([[ae_b[b][i] for b in BUDGETS] for i in range(3)])

    axg = fig.add_subplot(gsl[0, 0])
    for i, (c, lab) in enumerate([(C_LAND, 'Onshore'), (C_VPTS, 'VPTS'), (C_BAUER, 'Bauer')]):
        axg.plot(XP, rr_vals[i], 'o-', color=c, lw=1.4, markersize=4, label=lab)
    axg.set_xticks(XP); axg.set_xticklabels(BLABELS)
    axg.set_xlabel('AEP budget (%)'); axg.set_ylabel('Median exposure reduction (%)')
    axg.set_ylim(30, 108)
    axg.axvspan(-0.5, 1, color=C_ECO, alpha=0.06, zorder=0)
    axg.axvline(1, color=C_ECO, ls='--', lw=1.0)
    axg.text(1, 106, r'$\leq 1\%$ AEP', color=C_ECO, fontsize=8, ha='center', fontweight='bold')
    axg.legend(loc='lower right', fontsize=7.5, frameon=False)
    panel_label(axg, 'g'); style_ax(axg)

    axh = fig.add_subplot(gsl[0, 1])
    for i, c in enumerate([C_LAND, C_VPTS, C_BAUER]):
        axh.plot(XP, ae_vals[i], 's--', color=c, lw=1.0, markersize=3.5, alpha=0.75)
    axh.set_xticks(XP); axh.set_xticklabels(BLABELS)
    axh.set_xlabel('AEP budget (%)'); axh.set_ylabel('Mean AEP cost (%)')
    axh.set_ylim(0, 1.1)
    axh.axvspan(-0.5, 1, color=C_ECO, alpha=0.06, zorder=0)
    axh.axvline(1, color=C_ECO, ls='--', lw=1.0)
    panel_label(axh, 'h'); style_ax(axh)

    fig.text(0.5, 0.38, 'Most gain at negligible energy cost', fontsize=9,
             ha='center', va='center', fontweight='bold')

    pdf, png = savefig3(fig, 'fig5_R34')
    plt.close(fig)
    return pdf, png


# =====================================================================
# 图5 — R3 有限方向改变（1×3：捕获曲线 | Δθ50/80 vs d_full | >50% 占比）
# =====================================================================
def fig5_R3(on_df, vp_df, ba_df, ctx):
    """R3 专用图：有限方向改变即可捕获大部分可削减暴露。

    (a) 捕获曲线：5/10/20/30° 逐级调整占最大可削减暴露的比例（中位）。
    (b) 分组柱：Δθ50、Δθ80 与完整调整 d_full 对比（中位）。
    (c) 分组柱：≤20°/≤30° 内捕获 >50% 最大可削减暴露的场址占比。
    """
    groups = [(on_df, C_LAND, 'Onshore'), (vp_df, C_VPTS, 'VPTS'), (ba_df, C_BAUER, 'Bauer')]

    fig = plt.figure(figsize=(13.5, 3.9))
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.80, bottom=0.16, wspace=0.30)

    # ---- (a) 捕获曲线（R3.1）----
    axa = fig.add_subplot(gs[0, 0])
    pts = [0, 5, 10, 20, 30]
    for df, c, lab in groups:
        vals = [0] + [df[f'frac{d}'].median() * 100 for d in (5, 10, 20, 30)]
        axa.plot(pts, vals, 'o-', color=c, lw=1.5, markersize=3.8, label=lab)
    axa.axhline(50, color='#999999', ls=':', lw=0.9)
    axa.text(0.8, 53, 'half of max gain', fontsize=7.5, color='#555555')
    axa.axvspan(0, 20, color=C_ECO, alpha=0.05, zorder=0)
    axa.axvline(20, color=C_ECO, ls='--', lw=1.0)
    axa.text(20, 98, r'$\leq 20°$', color=C_ECO, fontsize=8, ha='center', fontweight='bold')
    axa.set_xlabel(r'Rotation $\Delta\theta$ (°)')
    axa.set_ylabel('Share of max exposure cut (%)')
    axa.set_xlim(0, 32); axa.set_ylim(0, 104)
    axa.set_xticks([0, 10, 20, 30])
    axa.set_title('Exposure falls steeply at small angles', fontsize=9)
    axa.legend(loc='lower right', fontsize=7.5, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    # ---- (b) Δθ50 / Δθ80 vs d_full（R3.2）----
    axb = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    w = 0.26
    series = [(r'$\Delta\theta_{50}$', 'd50', '#85C1E9'),
              (r'$\Delta\theta_{80}$', 'd80', '#2E86C1'),
              (r'$d_{\mathrm{full}}$', 'd_full', '#1B4F72')]
    for off, (lab, col, color) in zip([-w, 0, w], series):
        vals = [on_df[col].median(), vp_df[col].median(), ba_df[col].median()]
        axb.bar(x + off, vals, width=w, color=color, edgecolor='none', label=lab, zorder=3)
        for xi, v in zip(x + off, vals):
            axb.text(xi, v + 1.3, f'{v:.0f}', ha='center', va='bottom', fontsize=7, color='#333333')
    axb.set_xticks(x); axb.set_xticklabels(GROUP_ORDER)
    axb.set_ylabel('Rotation needed (°)')
    axb.set_ylim(0, 60)
    axb.set_title('Half the gain needs far less than full rotation', fontsize=9)
    axb.legend(loc='upper left', fontsize=7.5, frameon=False, ncol=3)
    panel_label(axb, 'b'); style_ax(axb)

    # ---- (c) ≤20°/≤30° 内捕获 >50% 的场址占比（R3.3）----
    axc = fig.add_subplot(gs[0, 2])
    x = np.arange(3)
    w = 0.38
    thr = [(r'$\leq 20°$', 'frac20', '#AEB6BF'),
           (r'$\leq 30°$', 'frac30', '#5D6D7E')]
    for off, (lab, col, color) in zip([-w / 2, w / 2], thr):
        vals = [(df[col] * 100 > 50).mean() * 100 for df, _, _ in groups]
        axc.bar(x + off, vals, width=w, color=color, edgecolor='none', label=lab, zorder=3)
        for xi, v in zip(x + off, vals):
            axc.text(xi, v + 1.2, f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#333333')
    axc.axhline(50, color='#999999', ls=':', lw=0.9)
    axc.set_xticks(x); axc.set_xticklabels(GROUP_ORDER)
    axc.set_ylabel('Share of farms (%)')
    axc.set_ylim(0, 100)
    axc.set_title('Most farms reach >50% within 20°', fontsize=9)
    axc.legend(loc='upper left', fontsize=7.5, frameon=False)
    panel_label(axc, 'c'); style_ax(axc)

    pdf, png = savefig3(fig, 'fig5_R3')
    plt.close(fig)
    return pdf, png


def main():
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}, cartopy={HAS_CARTOPY}')

    print('Fig 1 framework ...'); fig1_framework(on_df, vp_df, ba_df, ctx)
    print('Fig 2 R1 ...'); fig2_R1(on_df, vp_df, ba_df, ctx)
    print('Fig S2 R1 (suppl) ...'); figS2_R1(on_df, vp_df, ba_df, ctx)
    print('Fig 3 R2 ...'); fig3_R2(on_df, vp_df, ba_df, ctx)
    print('Fig 4 spatial ...'); fig4_spatial(on_df, vp_df, ba_df, ctx)
    print('Fig 5 R34 ...'); fig5_R34(on_df, vp_df, ba_df, ctx)
    print('Fig 5 R3 ...'); fig5_R3(on_df, vp_df, ba_df, ctx)
    print('Fig S1 threat (unchanged, v2) ...'); figS1_threat(ctx)

    print('\nDone. Outputs in figures_v3/:')
    for f in sorted(os.listdir(FIG3)):
        p = os.path.join(FIG3, f)
        print(f'  {f}  ({os.path.getsize(p)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()