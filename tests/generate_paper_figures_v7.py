# -*- coding: utf-8 -*-
"""generate_paper_figures_v7.py —— v7 主图重制（Fig 2–5 + 附图）。

相对 v6 的三项改动（2026-08-28 用户要求）：

1. **色系统一**：分类量一律 Okabe-Ito 组色；与某一组绑定的连续场用该组色的
   单色渐变；跨组物理量统一 viridis；阈值线一律生态朱红。v6 里 cividis /
   magma / viridis / 蓝红 KDE 四套色标混用的情况全部取消。
2. **子图对应子论点**：每张图按 R*.1 / R*.2 / R*.3 分成三条横带，带首写明
   该带回答的论断，子图与论断一一对应。
3. **压缩留白**：坐标轴按数据实际范围收紧；原先大片空白处补入信息子图。

所有数字经 figure_style.compute_metrics() 现算，与 final_numbers.txt 同源。

用法：python tests/generate_paper_figures_v7.py [2|3|4|5|all]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN,
                          C_ECON, C_ECO, GROUP_COLORS, GROUP_ORDER, THETAS,
                          circ, on_Evec, off_Evec)
import figure_style_v7 as v7
from figure_style_v7 import (W_DOUBLE, GROUP_LABEL, C_TXT, C_MUTED, C_GRID,
                             FS_PANEL, FS_TITLE, FS_TICK, FS_ANNOT,
                             band_header, plabel, flabel, sty, tight_limits,
                             raincloud, rule_line, cbar, seq_cmap, _shift,
                             CMAP_ONSHORE, GROUP_CMAP, CMAP_PHYS)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

from generate_paper_figures_v2 import add_basemap, RADAR_LOC

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(BASE, '..', 'figures_v7'))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    pdf = os.path.join(OUT, name + '.pdf')
    png = os.path.join(OUT, name + '.png')
    fig.savefig(pdf, facecolor='white')
    fig.savefig(png, facecolor='white', dpi=400)
    print(f'  -> {name}.png / .pdf')
    return pdf, png


def _groups(on_df, vp_df, ba_df):
    return {'Onshore': on_df, 'VPTS': vp_df, 'Bauer': ba_df}


# =====================================================================
# Fig 2 —— R1：方向结构稳定 => 暴露对朝向高度敏感
# =====================================================================
def fig2_sensitivity(on_df, vp_df, ba_df, ctx):
    G = _groups(on_df, vp_df, ba_df)
    grid = ctx['grid']
    on1 = ctx['on1']
    cur = ctx['cur']
    od = ctx['od']

    # 第三行的 panel g 是等纵横比地图，实测 1.59 in；整行若仍按 2.49 in 分配，
    # 多出的 0.90 in 会被 panel h 拉高。这里把整行压到地图高度并缩短画布。
    fig = plt.figure(figsize=(W_DOUBLE, 6.705))

    # ---- 条带 1（R1.1）：迁徙有稳定主轴 -------------------------------
    band_header(fig, 0.982, 'R1.1',
                'Migration keeps a stable axis — the two seasons reverse as vectors '
                '(183° apart) but differ by only 3.4° as exposure axes')
    gs1 = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.9433, bottom=0.7348,
                           wspace=0.42, width_ratios=[0.86, 1.0, 1.12])

    # (a) 方向玫瑰：春 / 秋
    axa = fig.add_subplot(gs1[0, 0], projection='polar')
    axa.set_theta_zero_location('N'); axa.set_theta_direction(-1)
    bins = np.arange(0, 361, 10)
    for col, c, lab in [('spring_dir', C_SPRING, 'Spring'),
                        ('autumn_dir', C_AUTUMN, 'Autumn')]:
        h, _ = np.histogram(grid[col] % 360, bins=bins)
        h = h / h.max()
        axa.bar(np.radians(bins[:-1] + 5), h, width=np.radians(10), bottom=0.0,
                color=c, alpha=0.62, edgecolor=c, linewidth=0.3, zorder=3)
        m = np.degrees(np.arctan2(np.median(np.sin(np.radians(grid[col]))),
                                  np.median(np.cos(np.radians(grid[col]))))) % 360
        axa.annotate('', xy=(np.radians(m), 1.0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='-|>', color=_shift(c, -0.28),
                                     lw=1.5, mutation_scale=9), zorder=6)
        axa.text(np.radians(m), 1.17, f'{m:.0f}°', fontsize=FS_ANNOT, fontweight='bold',
                 color=_shift(c, -0.3), ha='center', va='center')
    axa.set_rlim(0, 1.05); axa.set_rticks([])
    axa.set_xticks(np.radians([0, 90, 180, 270]))
    axa.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=FS_ANNOT)
    axa.tick_params(pad=-2)
    axa.grid(color=C_GRID, lw=0.4)
    axa.set_title('Flight direction (2,025 cells)', fontsize=FS_TITLE, pad=4)
    axa.legend(handles=[Line2D([], [], color=C_SPRING, lw=3, label='Spring'),
                        Line2D([], [], color=C_AUTUMN, lw=3, label='Autumn')],
               loc='lower right', bbox_to_anchor=(1.14, -0.02), ncol=1,
               fontsize=FS_ANNOT, handlelength=1.0, labelspacing=0.25)
    flabel(fig, 'a', 0.008, 0.9613)

    # (b) 方向集中度分布：春 vs 秋
    axb = fig.add_subplot(gs1[0, 1])
    for col, c, lab in [('spring_conc', C_SPRING, 'Spring'),
                        ('autumn_conc', C_AUTUMN, 'Autumn')]:
        v = grid[col].dropna().values
        axb.hist(v, bins=np.linspace(0.4, 1.0, 46), color=c, alpha=0.55,
                 edgecolor=_shift(c, -0.2), linewidth=0.35, zorder=3,
                 label=f'{lab}  median {np.median(v):.2f}')
        axb.axvline(np.median(v), color=_shift(c, -0.32), lw=1.1, ls='-', zorder=5)
    axb.set_xlabel('Directional concentration')
    axb.set_ylabel('Grid cells')
    axb.set_xlim(0.4, 1.0)
    axb.legend(loc='upper left', fontsize=FS_ANNOT, handlelength=1.0)
    sty(axb, grid_axis='y')
    plabel(axb, 'b', x=-0.26)

    # (c) 轴向折叠：方向 mod 180 后两季几乎重合
    axc = fig.add_subplot(gs1[0, 2])
    ax_s = grid.spring_dir.values % 180
    ax_a = grid.autumn_dir.values % 180
    for v, c, lab in [(ax_s, C_SPRING, 'Spring axis'), (ax_a, C_AUTUMN, 'Autumn axis')]:
        h, e = np.histogram(v, bins=np.arange(0, 181, 5))
        axc.step(e[:-1] + 2.5, h / h.sum() * 100, where='mid', color=c, lw=1.3,
                 zorder=4, label=lab)
        axc.fill_between(e[:-1] + 2.5, h / h.sum() * 100, step='mid',
                         color=c, alpha=0.22, zorder=3)
    ms = np.median(ax_s); ma = np.median(ax_a)
    for m, c in [(ms, C_SPRING), (ma, C_AUTUMN)]:
        axc.axvline(m, color=_shift(c, -0.3), lw=1.0, ls='--', zorder=5)
    top = axc.get_ylim()[1]
    axc.set_ylim(0, top * 1.26)
    yA = top * 1.09
    axc.annotate('', xy=(ma, yA), xytext=(ms, yA),
                 arrowprops=dict(arrowstyle='<->', color=C_ECO, lw=1.0))
    axc.text((ms + ma) / 2, yA * 1.05, f'Δ axis = {circ(ms, ma):.1f}°',
             fontsize=FS_ANNOT, fontweight='bold', color=C_ECO, ha='center', va='bottom')
    axc.set_xlabel('Direction folded to the 180° axis (°)')
    axc.set_ylabel('Grid cells (%)')
    axc.set_xlim(25, 105); axc.set_xticks([30, 45, 60, 75, 90, 105])
    axc.legend(loc='upper right', fontsize=FS_ANNOT, handlelength=1.1)
    sty(axc, grid_axis='y')
    plabel(axc, 'c', x=-0.20)

    # ---- 条带 2（R1.2）：该主轴令暴露对朝向高度敏感 --------------------
    band_header(fig, 0.700, 'R1.2',
                'That axis makes exposure highly sensitive to orientation — rotation '
                'alone spans 93–100% of each farm’s exposure range')
    gs2 = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.6555, bottom=0.3823,
                           wspace=0.42, width_ratios=[1.25, 0.78, 1.0])

    # (d) 暴露曲线族：以各场自身最小值为原点对齐，检验是否服从 sin²
    axd = fig.add_subplot(gs2[0, 0])
    off_x = np.arange(-90, 91)
    rng = np.random.default_rng(3)
    for g, df in G.items():
        c = GROUP_COLORS[g]
        ids = df.farm_id.values
        pick = rng.choice(ids, min(45, len(ids)), replace=False)
        curves = []
        for fid in pick:
            if g == 'Onshore':
                r = on1[on1.farm_id == fid]
                if r.empty:
                    continue
                r = r.iloc[0]
                E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
            else:
                E = off_Evec(fid, cur)
            rngE = E.max() - E.min()
            if rngE < 1e-12:
                continue
            En = (E - E.min()) / rngE
            k = int(np.argmin(E))
            curves.append(np.array([En[(k + d) % 180] for d in off_x]))
        if not curves:
            continue
        M = np.vstack(curves)
        axd.plot(off_x, np.median(M, axis=0), color=c, lw=1.5, zorder=5,
                 label=GROUP_LABEL[g])
        axd.fill_between(off_x, np.percentile(M, 10, axis=0),
                         np.percentile(M, 90, axis=0), color=c, alpha=0.16,
                         lw=0, zorder=3)
    axd.plot(off_x, np.sin(np.radians(off_x)) ** 2, color=C_ECON, lw=0.9, ls='--',
             zorder=6, label=r'$\sin^{2}$ geometry')
    axd.set_xlabel('Rotation away from each farm’s own exposure minimum (°)')
    axd.set_ylabel('Normalised exposure')
    axd.set_xlim(-90, 90); axd.set_ylim(-0.03, 1.03)
    axd.set_xticks([-90, -45, 0, 45, 90])
    axd.legend(loc='lower center', fontsize=FS_ANNOT, ncol=2, handlelength=1.2,
               columnspacing=1.0)
    sty(axd, grid_axis='y')
    plabel(axd, 'd', x=-0.14)

    # (e) 敏感性幅度 rel 的雨云图（轴按数据收紧）
    axe = fig.add_subplot(gs2[0, 1])
    rel = {g: df.rel.values * 100 for g, df in G.items()}
    raincloud(axe, rel, annotate={g: np.median(v) for g, v in rel.items()},
              annot_fmt='{:.1f}%')
    tight_limits(axe, np.concatenate(list(rel.values())), pad=0.10, hi=100.8)
    axe.set_ylabel('Exposure range reachable\nby rotation alone (%)')
    axe.annotate('every farm > 80%', xy=(0.5, 0.035), xycoords='axes fraction',
                 fontsize=FS_ANNOT, color=C_MUTED, ha='center')
    sty(axe, grid_axis='y')
    plabel(axe, 'e', x=-0.42)

    # (f) 机制：方向越集中，敏感性越高
    axf = fig.add_subplot(gs2[0, 2])
    mm = on_df.merge(on1[['farm_id', 'spring_conc']], on='farm_id', how='left').dropna(
        subset=['spring_conc'])
    axf.scatter(mm.spring_conc, mm.rel * 100, s=1.2, color=C_LAND, alpha=0.20,
                edgecolors='none', rasterized=True, zorder=3)
    qs = np.linspace(mm.spring_conc.min(), mm.spring_conc.max(), 13)
    xm, ym, lo, hi = [], [], [], []
    for i in range(len(qs) - 1):
        s = mm[(mm.spring_conc >= qs[i]) & (mm.spring_conc < qs[i + 1])]
        if len(s) < 12:
            continue
        xm.append(s.spring_conc.mean()); ym.append(np.median(s.rel) * 100)
        lo.append(np.percentile(s.rel, 25) * 100); hi.append(np.percentile(s.rel, 75) * 100)
    axf.fill_between(xm, lo, hi, color=C_LAND, alpha=0.22, lw=0, zorder=4)
    axf.plot(xm, ym, color=_shift(C_LAND, -0.25), lw=1.5, zorder=5)
    axf.set_xlabel('Directional concentration at the farm')
    axf.set_ylabel('Exposure range reachable (%)')
    tight_limits(axf, mm.rel.values * 100, pad=0.10, hi=100.4)
    axf.set_title('Onshore, binned median ± IQR', fontsize=FS_TITLE, color=C_MUTED, pad=3)
    sty(axf, grid_axis='y')
    plabel(axf, 'f', x=-0.24)

    # ---- 条带 3（R1.3）：跨源、跨空间普遍成立 --------------------------
    band_header(fig, 0.392, 'R1.3',
                'The sensitivity is universal — three independent data sources and the '
                'whole study region land in the same 93–100% band')
    gs3 = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.3029, bottom=0.0658,
                           wspace=0.46, width_ratios=[1.24, 1.0])

    # (g) rel 的空间分布
    if HAS_CARTOPY:
        axg = fig.add_subplot(gs3[0, 0], projection=ccrs.PlateCarree())
        add_basemap(axg, [-10.5, 19.5, 40.5, 57.5])
        s = on_df.dropna(subset=['rel'])
        hb = axg.hexbin(s.centroid_lon, s.centroid_lat, C=s.rel * 100,
                        cmap=CMAP_ONSHORE, gridsize=30, mincnt=1, vmin=96.5, vmax=100,
                        reduce_C_function=np.median, edgecolors='none',
                        transform=ccrs.PlateCarree(), zorder=3)
        off = od[['farm_id', 'source', 'centroid_lon', 'centroid_lat']].merge(
            pd.concat([vp_df.assign(grp='VPTS'), ba_df.assign(grp='Bauer')])
            [['farm_id', 'rel', 'grp']], on='farm_id', how='inner')
        for grp, mk in [('VPTS', 'o'), ('Bauer', 's')]:
            sub = off[off.grp == grp]
            axg.scatter(sub.centroid_lon, sub.centroid_lat, s=16, marker=mk,
                        c=GROUP_COLORS[grp], edgecolors='white', linewidths=0.45,
                        transform=ccrs.PlateCarree(), zorder=5)
        axg.set_title('Across space: onshore hex median, offshore farms overlaid',
                      fontsize=FS_TITLE, pad=3)
        cbar(fig, hb, axg, 'Range reachable (%)', ticks=[97, 98, 99, 100], size='3%')
        flabel(fig, 'g', 0.008, 0.3169)

    # (h) 三源森林图：中位 + IQR + 5–95%
    axh = fig.add_subplot(gs3[0, 1])
    for i, g in enumerate(GROUP_ORDER):
        v = G[g].rel.values * 100
        c = GROUP_COLORS[g]
        p5, q1, med, q3, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        axh.plot([p5, p95], [i, i], color=_shift(c, 0.55), lw=1.4,
                 solid_capstyle='round', zorder=3)
        axh.plot([q1, q3], [i, i], color=c, lw=5.0, solid_capstyle='butt', zorder=4)
        axh.scatter([med], [i], s=34, color='white', edgecolors=_shift(c, -0.35),
                    linewidths=1.2, zorder=6)
        axh.annotate(f'{med:.1f}%', xy=(med, i), xytext=(0, 8),
                     textcoords='offset points', fontsize=FS_ANNOT, fontweight='bold',
                     color=_shift(c, -0.3), ha='center')
        axh.annotate(f'n = {len(v):,}', xy=(0.995, i), xycoords=('axes fraction', 'data'),
                     xytext=(0, -9), textcoords='offset points', fontsize=FS_ANNOT,
                     color=C_MUTED, ha='right', va='center')
    axh.axvspan(93, 100, color=C_ECO, alpha=0.07, lw=0, zorder=0)
    axh.set_yticks(range(3))
    axh.set_yticklabels([GROUP_LABEL[g] for g in GROUP_ORDER])
    axh.set_ylim(2.6, -0.6)
    axh.set_xlabel('Exposure range reachable by rotation alone (%)')
    axh.set_xlim(91.2, 101)
    axh.set_title('Bars: IQR   ·   whiskers: 5–95%', fontsize=FS_TITLE, color=C_MUTED, pad=3)
    sty(axh, grid_axis='x')
    plabel(axh, 'h', x=-0.30)

    return save(fig, 'fig2_sensitivity')



# =====================================================================
# Fig 3 —— R2：能源最优与生态最优朝向系统性错位
# =====================================================================
def fig3_misalignment(on_df, vp_df, ba_df, ctx):
    G = _groups(on_df, vp_df, ba_df)
    on1 = ctx['on1']
    on_aep = ctx['on_aep']
    od = ctx['od']

    # 第三行的 panel i 是等纵横比地图：原来列宽不足使它只有 1.07 in、仅占行高
    # 的 39%，g/h 被拉高。这里加宽地图列并把整行压到地图高度。
    fig = plt.figure(figsize=(W_DOUBLE, 6.799))

    # ---- 条带 1（R2.1）：两个最优朝向系统性错开 ------------------------
    band_header(fig, 0.982, 'R2.1',
                'The two optima are set by independent physics and land apart — median '
                'misalignment 48° onshore, 40° VPTS, 45° Bauer')
    gs1 = fig.add_gridspec(1, 3, left=0.062, right=0.945, top=0.9422, bottom=0.7142,
                           wspace=0.52, width_ratios=[1.16, 0.62, 1.02])

    allf = pd.concat([df.assign(grp=g) for g, df in G.items()], ignore_index=True)

    # (a) 背靠背直方图：AEP 最优由风资源决定（散布全角域），生态最优被迁徙轴钉住
    axa = fig.add_subplot(gs1[0, 0])
    bins = np.arange(0, 181, 7.5)
    he, _ = np.histogram(allf.theta_econ % 180, bins=bins)
    hm, _ = np.histogram(allf.th_min % 180, bins=bins)
    he = he / he.sum() * 100
    hm = hm / hm.sum() * 100
    ctr = bins[:-1] + 3.75
    axa.bar(ctr, he, width=7.0, color=_shift(C_ECON, 0.45), edgecolor=C_ECON,
            linewidth=0.35, zorder=3, label='AEP-optimal θ')
    axa.bar(ctr, -hm, width=7.0, color=_shift(C_ECO, 0.45), edgecolor=C_ECO,
            linewidth=0.35, zorder=3, label='Exposure-optimal θ')
    axa.axhline(0, color='#999999', lw=0.6, zorder=4)
    axa.set_xlim(0, 180); axa.set_xticks([0, 45, 90, 135, 180])
    yl = max(he.max(), hm.max()) * 1.42
    axa.set_ylim(-yl, yl)
    tk = [-20, -10, 0, 10, 20]
    axa.set_yticks(tk)
    axa.set_yticklabels([str(abs(t)) for t in tk])
    axa.set_xlabel('Array orientation θ (°)')
    axa.set_ylabel('Farms (%)')
    axa.annotate('spread across every angle', xy=(0.70, 0.955), xycoords='axes fraction',
                 fontsize=FS_ANNOT, color=C_ECON, ha='center')
    axa.annotate('pinned to the migration axis', xy=(0.70, 0.045), xycoords='axes fraction',
                 fontsize=FS_ANNOT, color=C_ECO, ha='center')
    axa.legend(loc='lower left', fontsize=FS_ANNOT, handlelength=0.9, borderpad=0.2,
               labelspacing=0.2)
    sty(axa, grid_axis='y')
    plabel(axa, 'a', x=-0.15)

    # (b) 错位角雨云图
    axb = fig.add_subplot(gs1[0, 1])
    dfull = {g: df.d_full.values for g, df in G.items()}
    raincloud(axb, dfull, annotate={g: np.median(v) for g, v in dfull.items()},
              annot_fmt='{:.0f}°')
    axb.set_ylabel('Misalignment Δ (°)')
    axb.set_ylim(-3, 95); axb.set_yticks([0, 30, 60, 90])
    sty(axb, grid_axis='y')
    plabel(axb, 'b', x=-0.46)

    # (c) 联合分布：y 轴按数据收紧，避免 v6 里 80% 空白
    axc = fig.add_subplot(gs1[0, 2])
    hb = axc.hexbin(on_df.theta_econ % 180, on_df.th_min % 180, gridsize=32,
                    cmap=CMAP_ONSHORE, mincnt=1, bins='log', edgecolors='none', zorder=3)
    for g, mk in [('VPTS', 'o'), ('Bauer', 's')]:
        d = G[g]
        axc.scatter(d.theta_econ % 180, d.th_min % 180, s=11, marker=mk,
                    c=GROUP_COLORS[g], edgecolors='white', linewidths=0.35, zorder=5)
    lim = np.percentile(on_df.th_min % 180, [0.5, 99.5])
    ylo, yhi = max(0, lim[0] - 6), min(180, lim[1] + 6)
    axc.plot([0, 180], [0, 180], color=C_ECO, lw=0.8, ls='--', zorder=6)
    axc.set_xlim(0, 180); axc.set_ylim(ylo, yhi)
    axc.set_xticks([0, 45, 90, 135, 180])
    axc.set_xlabel('AEP-optimal θ (°)')
    axc.set_ylabel('Exposure-optimal θ (°)')
    axc.annotate('1:1 (aligned)', xy=(yhi, yhi), xytext=(-3, -8),
                 textcoords='offset points', fontsize=FS_ANNOT, color=C_ECO,
                 ha='right', va='top', rotation=0)
    cbar(fig, hb, axc, 'Onshore farms / hex', size='3.5%')
    plabel(axc, 'c', x=-0.22)

    # ---- 条带 2（R2.2）：AEP 最优处的暴露高出最小值一个数量级 ----------
    band_header(fig, 0.706, 'R2.2',
                'At the AEP optimum the exposure sits an order of magnitude above the '
                'reachable minimum — 603× onshore, 19× VPTS, 10× Bauer')
    gs2 = fig.add_gridspec(1, 3, left=0.062, right=0.975, top=0.6333, bottom=0.3612,
                           wspace=0.72, width_ratios=[0.58, 1.16, 1.0])

    # (d) 暴露比（对数）雨云图
    axd = fig.add_subplot(gs2[0, 0])
    ratio = {g: (df.Ee / df.Emin).replace([np.inf, -np.inf], np.nan).dropna().values
             for g, df in G.items()}
    rl = {g: np.log10(v[v > 0]) for g, v in ratio.items()}
    RCLIP = 5.0  # 10^5；Emin 趋近 0 的纯 sin² 场址会把比值推到 1e30 量级
    rl_c = {g: np.clip(v, -0.2, RCLIP) for g, v in rl.items()}
    n_clip = int(sum((v > RCLIP).sum() for v in rl.values()))
    raincloud(axd, rl_c, annotate={g: np.median(v) for g, v in ratio.items()},
              annot_fmt='{:.0f}×')
    axd.set_ylabel(r'$E(\theta_{\mathrm{econ}})\,/\,E_{\min}$')
    ticks = [0, 1, 2, 3, 4, 5]
    axd.set_yticks(ticks)
    axd.set_yticklabels(['1×', '10×', '100×', r'10$^3$', r'10$^4$', r'$\geq$10$^5$'])
    axd.set_ylim(-0.35, RCLIP + 0.35)
    axd.annotate(f'{n_clip} farms clipped at 10$^5$', xy=(0.5, 0.02),
                 xycoords='axes fraction', fontsize=FS_ANNOT, color=C_MUTED, ha='center')
    sty(axd, grid_axis='y')
    plabel(axd, 'd', x=-0.62)

    # (e) 代表场：暴露曲线与 AEP 曲线的两个最优错开
    axe = fig.add_subplot(gs2[0, 1])
    cand = on_df[(on_df.d_full.between(46, 50))].copy()
    aep_cols = [f'aep_{t:03d}' for t in range(0, 180, 10)]
    aep_idx = on_aep.set_index('farm_id')
    cand = cand[cand.farm_id.isin(aep_idx.index)]
    fid = int(cand.iloc[len(cand) // 2].farm_id)
    r = on1[on1.farm_id == fid].iloc[0]
    E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
    En = (E - E.min()) / (E.max() - E.min())
    a_raw = aep_idx.loc[fid, aep_cols].astype(float).values
    A = np.interp(THETAS, np.arange(0, 180, 10), a_raw, period=180)
    An = A / A.max()
    axe.plot(THETAS, En, color=C_ECO, lw=1.6, zorder=5, label='Exposure E(θ)')
    axe.fill_between(THETAS, 0, En, color=C_ECO, alpha=0.10, lw=0, zorder=3)
    ax2 = axe.twinx()
    ax2.plot(THETAS, An * 100, color=C_ECON, lw=1.4, ls='-', zorder=5,
             label='AEP(θ)')
    ax2.set_ylabel('AEP (% of max)', color=C_ECON, labelpad=1)
    ax2.tick_params(axis='y', colors=C_ECON, labelsize=FS_TICK)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(An.min() * 100 - 0.4, 100.25)
    te = float(r.theta_econ) % 180
    tm = float(THETAS[int(np.argmin(E))])
    axe.axvline(te, color=C_ECON, lw=0.9, ls='--', zorder=4)
    axe.axvline(tm, color=C_ECO, lw=0.9, ls='--', zorder=4)
    axe.annotate('', xy=(tm, 1.06), xytext=(te, 1.06),
                 arrowprops=dict(arrowstyle='<->', color=C_ECO, lw=1.0),
                 annotation_clip=False)
    axe.text((te + tm) / 2, 1.10, f'Δ = {circ(te, tm):.0f}°', fontsize=FS_ANNOT,
             fontweight='bold', color=C_ECO, ha='center', va='bottom')
    axe.text(te, -0.075, r'$\theta_{\mathrm{econ}}$', fontsize=FS_ANNOT, color=C_ECON,
             ha='center', va='top')
    axe.text(tm, -0.075, r'$\theta_{\min}$', fontsize=FS_ANNOT, color=C_ECO,
             ha='center', va='top')
    axe.set_xlim(0, 180); axe.set_xticks([0, 45, 90, 135, 180])
    axe.set_ylim(-0.02, 1.20)
    axe.set_yticks([0, 0.5, 1.0])
    axe.set_xlabel('Array orientation θ (°)')
    axe.set_ylabel('Normalised exposure', color=C_ECO)
    axe.tick_params(axis='y', colors=C_ECO)
    axe.set_title(f'Representative onshore farm (Δ near the median)',
                  fontsize=FS_TITLE, color=C_MUTED, pad=10)
    sty(axe)
    plabel(axe, 'e', x=-0.14, y=1.16)

    # (f) 暴露比随错位角上升——比值的几何来源
    axf = fig.add_subplot(gs2[0, 2])
    sub = on_df[(on_df.Emin > 0)].copy()
    sub['ratio'] = sub.Ee / sub.Emin
    axf.scatter(sub.d_full, sub.ratio, s=1.1, color=C_LAND, alpha=0.16,
                edgecolors='none', rasterized=True, zorder=3)
    edges = np.arange(0, 95, 7.5)
    xm, ym, lo, hi = [], [], [], []
    for i in range(len(edges) - 1):
        q = sub[(sub.d_full >= edges[i]) & (sub.d_full < edges[i + 1])]
        if len(q) < 15:
            continue
        xm.append(q.d_full.mean()); ym.append(np.median(q.ratio))
        lo.append(np.percentile(q.ratio, 25)); hi.append(np.percentile(q.ratio, 75))
    axf.fill_between(xm, lo, hi, color=C_LAND, alpha=0.22, lw=0, zorder=4)
    axf.plot(xm, ym, color=_shift(C_LAND, -0.25), lw=1.5, zorder=5)
    for g, mk in [('VPTS', 'o'), ('Bauer', 's')]:
        d = G[g]
        axf.scatter(d.d_full, d.Ee / d.Emin, s=12, marker=mk, c=GROUP_COLORS[g],
                    edgecolors='white', linewidths=0.35, zorder=6)
    axf.set_yscale('log')
    axf.set_ylim(1, 1e5)
    axf.set_xlabel('Misalignment Δ (°)')
    axf.set_ylabel(r'$E(\theta_{\mathrm{econ}})\,/\,E_{\min}$')
    axf.set_xlim(-2, 92); axf.set_xticks([0, 30, 60, 90])
    axf.set_title('Onshore binned median ± IQR', fontsize=FS_ANNOT, color=C_MUTED, pad=3)
    sty(axf, grid_axis='y')
    plabel(axf, 'f', x=-0.24)

    # ---- 条带 3（R2.3）：这部分暴露绝大多数可通过再朝向消除 ------------
    band_header(fig, 0.412, 'R2.3',
                'Almost all of that excess is removable by re-orientation alone — '
                'median avoidable share 99.8% onshore, 94.6% VPTS, 90.0% Bauer')
    gs3 = fig.add_gridspec(1, 3, left=0.062, right=0.945, top=0.2803, bottom=0.0681,
                           wspace=0.40, width_ratios=[0.62, 0.82, 1.72])

    # (g) 可削减比例雨云图
    axg = fig.add_subplot(gs3[0, 0])
    av = {g: df.avoid.values * 100 for g, df in G.items()}
    GLO = 55.0
    n_lo = int(sum((v < GLO).sum() for v in av.values()))
    raincloud(axg, {g: np.clip(v, GLO, 100.5) for g, v in av.items()},
              annotate={g: np.median(v) for g, v in av.items()}, annot_fmt='{:.1f}%')
    axg.set_ylim(GLO - 2.5, 101.2)
    axg.annotate(f'{n_lo} farms below 55%', xy=(0.5, 0.02), xycoords='axes fraction',
                 fontsize=FS_ANNOT, color=C_MUTED, ha='center')
    axg.set_ylabel('Avoidable exposure share (%)')
    sty(axg, grid_axis='y')
    plabel(axg, 'g', x=-0.66)

    # (h) 互补 ECDF：高于阈值的场址比例
    axh = fig.add_subplot(gs3[0, 1])
    for g in GROUP_ORDER:
        v = np.sort(av[g])
        share = (1 - np.arange(len(v)) / len(v)) * 100
        axh.step(v, share, where='post', color=GROUP_COLORS[g], lw=1.5,
                 zorder=4, label=GROUP_LABEL[g])
    rule_line(axh, x=90, label='90%', shade_to=101, label_pos=0.96)
    axh.set_xlabel('Avoidable exposure share (%)')
    axh.set_ylabel('Farms at or above (%)')
    axh.set_xlim(55, 101); axh.set_ylim(0, 103)
    axh.legend(loc='lower left', fontsize=FS_ANNOT, handlelength=1.2)
    sty(axh, grid_axis='y')
    plabel(axh, 'h', x=-0.30)

    # (i) 可削减比例的空间分布
    if HAS_CARTOPY:
        axi = fig.add_subplot(gs3[0, 2], projection=ccrs.PlateCarree())
        add_basemap(axi, [-10.5, 19.5, 40.5, 57.5])
        s2 = on_df.dropna(subset=['avoid'])
        hb2 = axi.hexbin(s2.centroid_lon, s2.centroid_lat, C=s2.avoid * 100,
                         cmap=CMAP_ONSHORE, gridsize=28, mincnt=1, vmin=97, vmax=100,
                         reduce_C_function=np.median, edgecolors='none',
                         transform=ccrs.PlateCarree(), zorder=3)
        off = od[['farm_id', 'centroid_lon', 'centroid_lat']].merge(
            pd.concat([vp_df.assign(grp='VPTS'), ba_df.assign(grp='Bauer')])
            [['farm_id', 'grp']], on='farm_id', how='inner')
        for grp, mk in [('VPTS', 'o'), ('Bauer', 's')]:
            q = off[off.grp == grp]
            axi.scatter(q.centroid_lon, q.centroid_lat, s=14, marker=mk,
                        c=GROUP_COLORS[grp], edgecolors='white', linewidths=0.4,
                        transform=ccrs.PlateCarree(), zorder=5)
        axi.set_title('Avoidable share across space (onshore hex median)',
                      fontsize=FS_TITLE, pad=3)
        cbar(fig, hb2, axi, 'Avoidable (%)', ticks=[97, 98, 99, 100], size='2.8%')
        flabel(fig, 'i', 0.545, 0.2943)

    return save(fig, 'fig3_misalignment')



# =====================================================================
# Fig 4 —— R3：前 20° 的旋转即可捕获大部分可削减暴露
# =====================================================================
def _capture_matrix(df, on1, cur, is_onshore):
    """逐场从 θ_econ 起旋转 0–60° 的捕获份额矩阵（%）。"""
    dths = np.arange(0, 61)
    out = np.full((len(df), len(dths)), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        fid = row.farm_id
        if is_onshore:
            m = on1[on1.farm_id == fid]
            if m.empty:
                continue
            m = m.iloc[0]
            E = on_Evec(m.spring_dir, m.autumn_dir, m.spring_conc, m.autumn_conc)
        else:
            E = off_Evec(fid, cur)
        te = float(row.theta_econ)
        Ee = E[int(round(te)) % 180]
        Emin = E.min()
        if Ee - Emin < 1e-12:
            continue
        sgn = 1
        for sg in (1, -1):
            if circ((te + sg * row.d_full) % 180, row.th_min) < 1.0:
                sgn = sg
                break
        for j, d in enumerate(dths):
            Ev = E[int(round(te + sgn * d)) % 180]
            out[i, j] = (Ee - Ev) / (Ee - Emin) * 100
    return dths, out


def fig4_capture(on_df, vp_df, ba_df, ctx):
    G = _groups(on_df, vp_df, ba_df)
    on1, cur, od = ctx['on1'], ctx['cur'], ctx['od']

    # 第二行的 panel f 是等纵横比地图：原来列宽不足使它只有 0.94 in（全图最小），
    # d/e 反被拉高。这里把地图列从 1.24 加宽到 1.86，地图长到 1.50 in 并让整行齐平。
    fig = plt.figure(figsize=(W_DOUBLE, 7.056))
    mats = {g: _capture_matrix(df, on1, cur, g == 'Onshore')
            for g, df in G.items()}

    # ---- 条带 1（R3.1）：收益集中在最初 20° -----------------------------
    band_header(fig, 0.982, 'R3.1',
                'The benefit is front-loaded — the first 20° already returns a median 55% '
                'of the avoidable exposure (69% VPTS, 53% Bauer)')
    gs1 = fig.add_gridspec(1, 3, left=0.070, right=0.965, top=0.9443, bottom=0.7175,
                           wspace=0.42, width_ratios=[1.28, 0.98, 0.94])

    # (a) 捕获曲线：中位 + IQR
    axa = fig.add_subplot(gs1[0, 0])
    for g in GROUP_ORDER:
        d, M = mats[g]
        c = GROUP_COLORS[g]
        axa.fill_between(d, np.nanpercentile(M, 25, axis=0),
                         np.nanpercentile(M, 75, axis=0), color=c, alpha=0.16,
                         lw=0, zorder=3)
        axa.plot(d, np.nanmedian(M, axis=0), color=c, lw=1.6, zorder=5,
                 label=GROUP_LABEL[g])
    rule_line(axa, x=20, label='20°', shade_to=0, label_pos=1.0)
    axa.axhline(50, color=C_MUTED, lw=0.7, ls=':', zorder=2)
    axa.annotate('half of the\nreachable gain', xy=(58, 52), fontsize=FS_ANNOT,
                 color=C_MUTED, ha='right', va='bottom')
    axa.set_xlabel('Rotation from the AEP optimum, Δθ (°)')
    axa.set_ylabel('Avoidable exposure removed (%)')
    axa.set_xlim(0, 60); axa.set_ylim(0, 103)
    axa.legend(loc='lower right', fontsize=FS_ANNOT, handlelength=1.2)
    sty(axa, grid_axis='y')
    plabel(axa, 'a', x=-0.15)

    # (b) 四个角度档位的分组条（正文引用的 5/10/20/30° 数字）
    axb = fig.add_subplot(gs1[0, 1])
    steps = [5, 10, 20, 30]
    wdt = 0.26
    for k, g in enumerate(GROUP_ORDER):
        vals = [np.median(G[g][f'frac{t}']) * 100 for t in steps]
        xs = np.arange(len(steps)) + (k - 1) * wdt
        axb.bar(xs, vals, width=wdt, color=GROUP_COLORS[g],
                edgecolor=_shift(GROUP_COLORS[g], -0.3), linewidth=0.4, zorder=3)
        for x, v in zip(xs, vals):
            axb.annotate(f'{v:.0f}', xy=(x, v), xytext=(0, 1.5),
                         textcoords='offset points', fontsize=FS_ANNOT,
                         color=_shift(GROUP_COLORS[g], -0.3), ha='center')
    axb.axhline(50, color=C_ECO, lw=0.8, ls=':', zorder=2)
    axb.set_xticks(range(len(steps)))
    axb.set_xticklabels([f'{t}°' for t in steps])
    axb.set_xlabel('Rotation allowed')
    axb.set_ylabel('')
    axb.set_ylim(0, 100)
    sty(axb, grid_axis='y')
    plabel(axb, 'b', x=-0.26)

    # (c) 边际收益：每多转 1° 还能拿回多少
    axc = fig.add_subplot(gs1[0, 2])
    for g in GROUP_ORDER:
        d, M = mats[g]
        med = np.nanmedian(M, axis=0)
        axc.plot(d[1:], np.diff(med), color=GROUP_COLORS[g], lw=1.4, zorder=4)
    rule_line(axc, x=20, shade_to=0)
    axc.set_xlabel('Rotation Δθ (°)')
    axc.set_ylabel('Marginal gain (% per °)')
    axc.set_xlim(0, 60)
    axc.set_ylim(bottom=0)
    axc.set_title('Return per extra degree', fontsize=FS_ANNOT, color=C_MUTED, pad=3)
    sty(axc, grid_axis='y')
    plabel(axc, 'c', x=-0.28)

    # ---- 条带 2（R3.2）：半收益角远小于全错位角 -------------------------
    band_header(fig, 0.706, 'R3.2',
                'Half the gain needs only about a third of the full turn — Δθ50 = 17° / 14° / 18°, '
                'against a full misalignment of 48° / 40° / 45°')
    gs2 = fig.add_gridspec(1, 3, left=0.078, right=0.945, top=0.6396, bottom=0.4270,
                           wspace=0.36, width_ratios=[0.72, 0.72, 1.86])

    # (d) 哑铃：Δθ50 -> Δθ80 -> 全错位
    axd = fig.add_subplot(gs2[0, 0])
    for i, g in enumerate(GROUP_ORDER):
        df = G[g]
        c = GROUP_COLORS[g]
        num = lambda col: pd.to_numeric(df[col], errors='coerce')
        m50, m80, mf = (np.nanmedian(num('d50')), np.nanmedian(num('d80')),
                        np.nanmedian(num('d_full')))
        axd.plot([m50, mf], [i, i], color=_shift(c, 0.5), lw=4.0,
                 solid_capstyle='round', zorder=3)
        axd.scatter([m50], [i], s=42, marker='o', c=c, edgecolors='white',
                    linewidths=0.7, zorder=6)
        axd.scatter([m80], [i], s=34, marker='D', c=_shift(c, -0.2),
                    edgecolors='white', linewidths=0.6, zorder=6)
        axd.scatter([mf], [i], s=46, marker='X', c=C_ECON, edgecolors='white',
                    linewidths=0.7, zorder=6)
        for v, dy in ((m50, 9), (mf, 9)):
            axd.annotate(f'{v:.0f}°', xy=(v, i), xytext=(0, dy),
                         textcoords='offset points', fontsize=FS_ANNOT,
                         fontweight='bold', color=C_TXT, ha='center')
    axd.set_yticks(range(3))
    axd.set_yticklabels([GROUP_LABEL[g].replace('Offshore · ', '') for g in GROUP_ORDER])
    axd.set_ylim(2.8, -0.9)
    axd.set_xlabel('Rotation (°)')
    axd.set_xlim(0, 58)
    axd.legend(handles=[
        Line2D([], [], marker='o', ls='', color='#888888', markersize=4.6, label=r'$\Delta\theta_{50}$'),
        Line2D([], [], marker='D', ls='', color='#888888', markersize=4.0, label=r'$\Delta\theta_{80}$'),
        Line2D([], [], marker='X', ls='', color=C_ECON, markersize=5.0, label='full Δ')],
        loc='upper right', fontsize=FS_ANNOT, handletextpad=0.25, labelspacing=0.22,
        ncol=3, columnspacing=0.7)
    sty(axd, grid_axis='x')
    plabel(axd, 'd', x=-0.40)

    # (e) 逐场 Δθ50 对全错位角
    axe = fig.add_subplot(gs2[0, 1])
    axe.scatter(on_df.d_full, on_df.d50, s=1.3, color=C_LAND, alpha=0.18,
                edgecolors='none', rasterized=True, zorder=3)
    for g, mk in [('VPTS', 'o'), ('Bauer', 's')]:
        d = G[g]
        axe.scatter(d.d_full, d.d50, s=13, marker=mk, c=GROUP_COLORS[g],
                    edgecolors='white', linewidths=0.35, zorder=5)
    xx = np.linspace(0, 90, 50)
    axe.plot(xx, xx, color=C_MUTED, lw=0.8, ls='--', zorder=4)
    axe.plot(xx, xx / 3, color=C_ECO, lw=1.0, ls='-', zorder=4)
    axe.annotate('1:1', xy=(80, 80), fontsize=FS_ANNOT, color=C_MUTED, ha='left')
    axe.annotate('one third', xy=(84, 28), fontsize=FS_ANNOT, color=C_ECO, ha='right')
    axe.set_xlabel('Full misalignment Δ (°)')
    axe.set_ylabel(r'$\Delta\theta_{50}$ (°)')
    axe.set_xlim(0, 92); axe.set_ylim(0, 92)
    sty(axe, grid_axis='y')
    plabel(axe, 'e', x=-0.30)

    # (f) Δθ50 的空间分布（v6 里这是 Fig 1h，Fig 1 改版后移到这里保留）
    if HAS_CARTOPY:
        axf = fig.add_subplot(gs2[0, 2], projection=ccrs.PlateCarree())
        add_basemap(axf, [-10.5, 19.5, 40.5, 57.5])
        s2 = on_df.dropna(subset=['d50'])
        hbf = axf.hexbin(s2.centroid_lon, s2.centroid_lat, C=s2.d50,
                         cmap=CMAP_ONSHORE, gridsize=26, mincnt=1, vmin=8, vmax=30,
                         reduce_C_function=np.median, edgecolors='none',
                         transform=ccrs.PlateCarree(), zorder=3)
        off = od[['farm_id', 'centroid_lon', 'centroid_lat']].merge(
            pd.concat([vp_df.assign(grp='VPTS'), ba_df.assign(grp='Bauer')])
            [['farm_id', 'grp']], on='farm_id', how='inner')
        for grp, mk in [('VPTS', 'o'), ('Bauer', 's')]:
            q = off[off.grp == grp]
            axf.scatter(q.centroid_lon, q.centroid_lat, s=13, marker=mk,
                        c=GROUP_COLORS[grp], edgecolors='white', linewidths=0.4,
                        transform=ccrs.PlateCarree(), zorder=5)
        axf.set_title(r'$\Delta\theta_{50}$ across space (onshore hex median)',
                      fontsize=FS_TITLE, pad=3)
        cbar(fig, hbf, axf, r'$\Delta\theta_{50}$ (°)', ticks=[10, 20, 30], size='2.8%')
        flabel(fig, 'f', 0.515, 0.6596)

    # ---- 条带 3（R3.3）：这是场址级规律，不是均值假象 -------------------
    band_header(fig, 0.412, 'R3.3',
                'It holds farm by farm, not just on average — 58% / 69% / 54% of farms clear the '
                'half-gain mark within 20°')
    gs3 = fig.add_gridspec(1, 2, left=0.070, right=0.975, top=0.3491, bottom=0.0656,
                           wspace=0.30, width_ratios=[1.06, 1.0])

    # (g) 阈值达成率
    axg = fig.add_subplot(gs3[0, 0])
    combos = [('frac20', 50, '≥50% within 20°'), ('frac20', 80, '≥80% within 20°'),
              ('frac30', 50, '≥50% within 30°'), ('frac30', 80, '≥80% within 30°')]
    wdt = 0.26
    for k, g in enumerate(GROUP_ORDER):
        vals = [(G[g][col] * 100 >= thr).mean() * 100 for col, thr, _ in combos]
        xs = np.arange(len(combos)) + (k - 1) * wdt
        axg.bar(xs, vals, width=wdt, color=GROUP_COLORS[g],
                edgecolor=_shift(GROUP_COLORS[g], -0.3), linewidth=0.4, zorder=3,
                label=GROUP_LABEL[g])
        for x, v in zip(xs, vals):
            axg.annotate(f'{v:.0f}', xy=(x, v), xytext=(0, 1.5),
                         textcoords='offset points', fontsize=FS_ANNOT,
                         color=_shift(GROUP_COLORS[g], -0.3), ha='center')
    axg.axhline(50, color=C_MUTED, lw=0.7, ls=':', zorder=2)
    axg.set_xticks(range(len(combos)))
    axg.set_xticklabels(['50% @20°', '80% @20°', '50% @30°', '80% @30°'], fontsize=FS_ANNOT)
    axg.set_ylabel('Farms meeting the target (%)')
    axg.set_ylim(0, 92)
    axg.legend(loc='upper right', fontsize=FS_ANNOT, ncol=3, handlelength=0.9,
               columnspacing=0.8)
    sty(axg, grid_axis='y')
    plabel(axg, 'g', x=-0.13)

    # (h) Δθ50 的分组 ECDF，直接读出 20°/30° 处的达成率
    axh = fig.add_subplot(gs3[0, 1])
    for g in GROUP_ORDER:
        v = np.sort(G[g].d50.dropna().values)
        axh.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where='post',
                 color=GROUP_COLORS[g], lw=1.5, zorder=4, label=GROUP_LABEL[g])
    for xv in (20, 30):
        axh.axvline(xv, color=C_ECO, ls='--', lw=0.8, zorder=2)
        for g in GROUP_ORDER:
            share = (G[g].d50 <= xv).mean() * 100
            axh.scatter([xv], [share], s=16, color=GROUP_COLORS[g],
                        edgecolors='white', linewidths=0.5, zorder=6)
    axh.axvspan(0, 20, color=C_ECO, alpha=0.07, lw=0, zorder=0)
    axh.set_xlabel(r'$\Delta\theta_{50}$ (°)')
    axh.set_ylabel('Farms at or below (%)')
    axh.set_xlim(0, 60); axh.set_ylim(0, 103)
    axh.legend(loc='lower right', fontsize=FS_ANNOT, handlelength=1.2)
    sty(axh, grid_axis='y')
    plabel(axh, 'h', x=-0.16)

    return save(fig, 'fig4_capture')


# =====================================================================
# Fig 5 —— R4：生态收益的 AEP 代价几乎为零
# =====================================================================
def fig5_tradeoff(on_df, vp_df, ba_df, ctx):
    G = _groups(on_df, vp_df, ba_df)
    on, to, od = ctx['on'], ctx['to'], ctx['od']
    BUD = [0.005, 0.01, 0.02, 0.05]

    def by_budget(g):
        """返回 {budget: DataFrame(farm_id, aep_cost_pct, rr)}。"""
        out = {}
        for b in BUD:
            if g == 'Onshore':
                sub = on[on.budget == b][['farm_id', 'aep_cost_pct', 'risk_reduction']]
                sub = sub.rename(columns={'risk_reduction': 'rr'})
            else:
                ids = ctx['vpts_ids'] if g == 'VPTS' else ctx['bauer_ids']
                sub = to[(to.budget == b) & (to.farm_id.isin(ids))][
                    ['farm_id', 'aep_cost_pct', 'risk_reduction_pct']]
                sub = sub.rename(columns={'risk_reduction_pct': 'rr'})
            out[b] = sub
        return out

    BB = {g: by_budget(g) for g in GROUP_ORDER}

    # 第三行的地图是等纵横比轴，实测只占 1.03 in；若整行仍按 2.67 in 分配，
    # f、g 会被拉成 2.6 倍于地图的高瘦图。这里把整行压到地图高度并缩短画布。
    fig = plt.figure(figsize=(W_DOUBLE, 6.667))

    # ---- 条带 1（R4.1）：1% 预算即可买到大部分削减 ----------------------
    band_header(fig, 0.982, 'R4.1',
                'A 1% allowance is nearly free and buys most of the cut — realised AEP loss '
                '0.51% / 0.38% / 0.49%, median cut 96.9% / 84.2% / 47.6%')
    gs1 = fig.add_gridspec(1, 3, left=0.062, right=0.975, top=0.9411, bottom=0.7173,
                           wspace=0.48, width_ratios=[1.18, 0.98, 0.80])

    # (a) 能源-暴露 Pareto
    axa = fig.add_subplot(gs1[0, 0])
    for g, mk, sz, al in [('Onshore', 'o', 1.2, 0.10), ('VPTS', 'o', 11, 0.75),
                          ('Bauer', 's', 11, 0.75)]:
        pts = pd.concat([BB[g][b] for b in BUD])
        axa.scatter(pts.aep_cost_pct, pts.rr, s=sz, marker=mk, c=GROUP_COLORS[g],
                    alpha=al, edgecolors='none', rasterized=True, zorder=3,
                    label=GROUP_LABEL[g])
    for g, mk in [('Onshore', 'o'), ('VPTS', 'o'), ('Bauer', 's')]:
        d = BB[g][0.01]
        axa.scatter([d.aep_cost_pct.median()], [d.rr.median()], s=52, marker=mk,
                    c=GROUP_COLORS[g], edgecolors='white', linewidths=1.0, zorder=7)
    rule_line(axa, x=1, label='1% AEP', shade_to=0, label_pos=1.0)
    axa.set_xlabel('Realised AEP loss (%)')
    axa.set_ylabel('Exposure reduction (%)')
    axa.set_xlim(-0.12, 5.4); axa.set_ylim(0, 104)
    axa.legend(loc='lower right', fontsize=FS_ANNOT, handlelength=0.9, scatterpoints=1,
               markerscale=2.6)
    axa.set_title('Large markers: per-group median at the 1% budget',
                  fontsize=FS_ANNOT, color=C_MUTED, pad=3)
    sty(axa, grid_axis='y')
    plabel(axa, 'a', x=-0.15)

    # (b) 1% 预算下"付出 vs 得到"的成对条
    axb = fig.add_subplot(gs1[0, 1])
    for i, g in enumerate(GROUP_ORDER):
        d = BB[g][0.01]
        c = GROUP_COLORS[g]
        cost, gain = d.aep_cost_pct.mean(), d.rr.mean()
        axb.barh(i + 0.16, gain, height=0.30, left=0.1, color=c,
                 edgecolor=_shift(c, -0.3), linewidth=0.4, zorder=3)
        axb.barh(i - 0.16, cost, height=0.30, left=0.1, color=_shift(C_ECON, 0.35),
                 edgecolor=C_ECON, linewidth=0.4, zorder=3)
        axb.annotate(f'{gain:.1f}% cut', xy=(gain, i + 0.16), xytext=(3, 0),
                     textcoords='offset points', fontsize=FS_ANNOT, fontweight='bold',
                     color=_shift(c, -0.3), va='center')
        axb.annotate(f'{cost:.2f}% AEP', xy=(cost, i - 0.16), xytext=(3, 0),
                     textcoords='offset points', fontsize=FS_ANNOT, color=C_ECON,
                     va='center')
    axb.set_yticks(range(3))
    axb.set_yticklabels([GROUP_LABEL[g].replace('Offshore · ', '') for g in GROUP_ORDER])
    axb.set_ylim(2.6, -0.6)
    axb.set_xscale('log')
    axb.set_xlim(0.1, 900)
    axb.set_xticks([0.1, 1, 10, 100])
    axb.set_xticklabels(['0.1', '1', '10', '100'])
    axb.set_xlabel('Mean value at the 1% budget (%, log scale)')
    axb.set_title('Grey: energy paid   ·   colour: exposure bought',
                  fontsize=FS_ANNOT, color=C_MUTED, pad=3)
    sty(axb, grid_axis='x')
    plabel(axb, 'b', x=-0.30)

    # (c) 1% 预算下削减率的分布
    axc = fig.add_subplot(gs1[0, 2])
    rr1 = {g: BB[g][0.01].rr.values for g in GROUP_ORDER}
    raincloud(axc, rr1, annotate={g: np.median(v) for g, v in rr1.items()},
              annot_fmt='{:.0f}%')
    axc.set_ylabel('Exposure reduction at 1% (%)')
    axc.set_ylim(-3, 108)
    sty(axc, grid_axis='y')
    plabel(axc, 'c', x=-0.44)

    # ---- 条带 2（R4.2）：前沿不对称的几何来源 --------------------------
    band_header(fig, 0.706, 'R4.2',
                'The asymmetry is geometric — AEP is flat at its own optimum while exposure is '
                'steep there, so a tiny concession buys a large cut')
    gs2 = fig.add_gridspec(1, 2, left=0.070, right=0.975, top=0.6348, bottom=0.3733,
                           wspace=0.42, width_ratios=[1.0, 1.06])

    # (d) 两条曲线在 θ_econ 附近的陡峭度对比（全体中位曲线）
    axd = fig.add_subplot(gs2[0, 0])
    on1, on_aep = ctx['on1'], ctx['on_aep']
    aep_cols = [f'aep_{t:03d}' for t in range(0, 180, 10)]
    ai = on_aep.set_index('farm_id')
    rng = np.random.default_rng(7)
    ids = rng.choice(on_df.farm_id.values, 500, replace=False)
    offs = np.arange(-60, 61)
    Ecur, Acur = [], []
    for fid in ids:
        m = on1[on1.farm_id == fid]
        if m.empty or fid not in ai.index:
            continue
        m = m.iloc[0]
        E = on_Evec(m.spring_dir, m.autumn_dir, m.spring_conc, m.autumn_conc)
        A = np.interp(THETAS, np.arange(0, 180, 10),
                      ai.loc[fid, aep_cols].astype(float).values, period=180)
        te = int(round(float(m.theta_econ))) % 180
        rE = E.max() - E.min()
        if rE < 1e-12:
            continue
        Ecur.append([(E[(te + d) % 180] - E.min()) / rE * 100 for d in offs])
        Acur.append([A[(te + d) % 180] / A[te] * 100 for d in offs])
    Ecur, Acur = np.array(Ecur), np.array(Acur)
    axd.plot(offs, np.median(Ecur, axis=0), color=C_ECO, lw=1.7, zorder=5,
             label='Exposure (% of own range)')
    axd.fill_between(offs, np.percentile(Ecur, 25, axis=0),
                     np.percentile(Ecur, 75, axis=0), color=C_ECO, alpha=0.15,
                     lw=0, zorder=3)
    ax2 = axd.twinx()
    ax2.plot(offs, np.median(Acur, axis=0), color=C_ECON, lw=1.5, zorder=5,
             label='AEP (% of value at θ_econ)')
    ax2.fill_between(offs, np.percentile(Acur, 25, axis=0),
                     np.percentile(Acur, 75, axis=0), color=C_ECON, alpha=0.12,
                     lw=0, zorder=3)
    ax2.set_ylabel('AEP (% of value at θ$_{econ}$)', color=C_ECON, labelpad=1)
    ax2.tick_params(axis='y', colors=C_ECON, labelsize=FS_TICK)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(94.5, 100.4)
    axd.axvline(0, color=C_MUTED, lw=0.7, ls=':', zorder=2)
    axd.set_xlim(-60, 60); axd.set_ylim(0, 102)
    axd.set_xlabel('Rotation from the AEP optimum (°)')
    axd.set_ylabel('Exposure (% of own range)', color=C_ECO)
    axd.tick_params(axis='y', colors=C_ECO)
    axd.set_title('Onshore, median ± IQR over 500 farms', fontsize=FS_ANNOT,
                  color=C_MUTED, pad=3)
    axd.legend(loc='lower left', fontsize=FS_ANNOT, handlelength=1.0, borderpad=0.2)
    ax2.legend(loc='upper right', fontsize=FS_ANNOT, handlelength=1.0, borderpad=0.2)
    sty(axd)
    plabel(axd, 'd', x=-0.14)

    # (e) 预算 -> 削减：每组的饱和曲线 + 已实现/理论上限之比
    axe = fig.add_subplot(gs2[0, 1])
    for g in GROUP_ORDER:
        c = GROUP_COLORS[g]
        xs = [b * 100 for b in BUD]
        ys = [np.median(BB[g][b].rr) for b in BUD]
        axe.plot(xs, ys, color=c, lw=1.5, marker='o', markersize=4,
                 markeredgecolor='white', markeredgewidth=0.6, zorder=5,
                 label=GROUP_LABEL[g])
        for x, y in zip(xs, ys):
            axe.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 5),
                         textcoords='offset points', fontsize=FS_ANNOT,
                         color=_shift(c, -0.3), ha='center')
    rule_line(axe, x=1, label='1% AEP', shade_to=0, label_pos=1.0)
    axe.set_xscale('log')
    axe.set_xticks([0.5, 1, 2, 5])
    axe.set_xticklabels(['0.5%', '1%', '2%', '5%'])
    axe.set_xlabel('AEP budget')
    axe.set_ylabel('Median exposure reduction (%)')
    axe.set_ylim(28, 108)
    axe.legend(loc='lower right', fontsize=FS_ANNOT, handlelength=1.2)
    sty(axe, grid_axis='y')
    plabel(axe, 'e', x=-0.13)

    # ---- 条带 3（R4.3）：1% 之后迅速饱和 --------------------------------
    sat = fs.r43_saturation(ctx)
    band_header(fig, 0.412, 'R4.3',
                'Past ~1% the return dies out — going from a 2% to a 5% budget adds nothing '
                f"for {sat['Onshore']:.0f}% / {sat['VPTS']:.0f}% / {sat['Bauer']:.0f}% of farms")
    gs3 = fig.add_gridspec(1, 3, left=0.088, right=0.945, top=0.2908, bottom=0.0738,
                           wspace=0.36, width_ratios=[0.58, 0.85, 1.75])

    # (f) 无新增收益的场址比例
    axf = fig.add_subplot(gs3[0, 0])
    for i, g in enumerate(GROUP_ORDER):
        c = GROUP_COLORS[g]
        axf.bar([i], [sat[g]], width=0.6, color=c, edgecolor=_shift(c, -0.3),
                linewidth=0.4, zorder=3)
        axf.annotate(f'{sat[g]:.0f}%', xy=(i, sat[g]), xytext=(0, 2),
                     textcoords='offset points', fontsize=FS_ANNOT, fontweight='bold',
                     color=_shift(c, -0.3), ha='center')
    axf.set_xticks(range(3))
    axf.set_xticklabels([GROUP_LABEL[g].replace('Offshore · ', '') for g in GROUP_ORDER])
    axf.set_ylabel('No extra gain, 2% → 5% (%)')
    axf.set_ylim(0, 108)
    sty(axf, grid_axis='y')
    plabel(axf, 'f', x=-0.50)

    # (g) 政策情境：能源让步上限 × 生态底线 -> 达标场址占比
    # 两种政策定法（定死预算 / 定死底线）在同一张图上交叉；VPTS 的 ≥90% 点线
    # 自 1% 起走平，即「再加预算也买不到生态收益」的直接证据。
    axg = fig.add_subplot(gs3[0, 1])
    STYLE = [(50, '-'), (80, '--'), (90, ':')]
    xs = [b * 100 for b in BUD]
    for g in GROUP_ORDER:
        for tgt, ls in STYLE:
            ys = [float((BB[g][b].rr >= tgt).mean() * 100) for b in BUD]
            axg.plot(xs, ys, color=GROUP_COLORS[g], ls=ls, lw=1.4, zorder=4,
                     marker='o', markersize=2.4, markeredgewidth=0)
    yv = [float((BB['VPTS'][b].rr >= 90).mean() * 100) for b in BUD]
    # 直接贴在那条走平的绿色点线上方 —— x 2–4、y 42–52 是全图唯一的空隙
    axg.text(1.95, 43, 'VPTS: extra budget\nbuys no extra farms',
             fontsize=FS_ANNOT, color=_shift(C_VPTS, -0.2), ha='left', va='bottom')
    rule_line(axg, x=1, shade_to=0.45)
    axg.set_xscale('log')
    axg.set_xticks(xs); axg.set_xticklabels(['0.5%', '1%', '2%', '5%'])
    axg.set_xlabel('AEP budget ceiling set by policy')
    axg.set_ylabel('Farms meeting the floor (%)')
    axg.set_ylim(0, 103)
    # 组色在本图 a–f 已建立，这里只需线型键，图例因此从 6 条压到 3 条
    axg.legend(handles=[Line2D([], [], color=C_MUTED, ls=ls, lw=1.2,
                               label=f'floor ≥{t}%') for t, ls in STYLE],
               loc='upper left', fontsize=FS_ANNOT, handlelength=1.6,
               labelspacing=0.18, borderpad=0.2)
    sty(axg, grid_axis='y')
    plabel(axg, 'g', x=-0.22)

    # (h) 1% 预算下削减率的空间分布
    if HAS_CARTOPY:
        axh = fig.add_subplot(gs3[0, 2], projection=ccrs.PlateCarree())
        add_basemap(axh, [-10.5, 19.5, 40.5, 57.5])
        d1 = BB['Onshore'][0.01].merge(
            on_df[['farm_id', 'centroid_lon', 'centroid_lat']], on='farm_id')
        hbh = axh.hexbin(d1.centroid_lon, d1.centroid_lat, C=d1.rr,
                         cmap=CMAP_ONSHORE, gridsize=26, mincnt=1, vmin=40, vmax=100,
                         reduce_C_function=np.median, edgecolors='none',
                         transform=ccrs.PlateCarree(), zorder=3)
        off = od[['farm_id', 'centroid_lon', 'centroid_lat']].merge(
            pd.concat([vp_df.assign(grp='VPTS'), ba_df.assign(grp='Bauer')])
            [['farm_id', 'grp']], on='farm_id', how='inner')
        for grp, mk in [('VPTS', 'o'), ('Bauer', 's')]:
            q = off[off.grp == grp]
            axh.scatter(q.centroid_lon, q.centroid_lat, s=13, marker=mk,
                        c=GROUP_COLORS[grp], edgecolors='white', linewidths=0.4,
                        transform=ccrs.PlateCarree(), zorder=5)
        axh.set_title('Reduction at the 1% budget (hex median)',
                      fontsize=FS_TITLE, pad=3)
        cbar(fig, hbh, axh, 'Reduction (%)', ticks=[40, 60, 80, 100], size='2.8%')
        flabel(fig, 'h', 0.520, 0.3108)

    return save(fig, 'fig5_tradeoff')


# =====================================================================
def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else 'all').lower()
    v7.use_v7_style()
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}')

    if which in ('2', 'all'):
        print('Fig 2 (R1 · orientation sensitivity) ...')
        fig2_sensitivity(on_df, vp_df, ba_df, ctx)
    if which in ('3', 'all'):
        print('Fig 3 (R2 · misalignment) ...')
        fig3_misalignment(on_df, vp_df, ba_df, ctx)
    if which in ('4', 'all'):
        print('Fig 4 (R3 · capture) ...')
        fig4_capture(on_df, vp_df, ba_df, ctx)
    if which in ('5', 'all'):
        print('Fig 5 (R4 · trade-off) ...')
        fig5_tradeoff(on_df, vp_df, ba_df, ctx)


if __name__ == '__main__':
    main()
