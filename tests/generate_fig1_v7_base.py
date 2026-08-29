# -*- coding: utf-8 -*-
"""
generate_fig1_v7_base.py —— Fig 1 v7 重设计的「底图 + 数据」生产脚本。

新版 Fig 1 的构图（用户 2026-08-28 定稿：环绕式）：
  * 研究区地图铺满整幅（full-bleed），不再只占 v6 底部两格；
  * 原 a–f 的机理面板压缩成 5 步机理链 b→c→d→e→f，作为流程卡
    **环绕地图外圈**排布（左上/左下 → 底左/底中 → 右侧折上），
    把中间那条 SW→NE 走向的数据带整条让出来；
  * 节点 e、f 内嵌真实数据图表（旋转-收益曲线、AEP 预算-削减柱）。

被否掉的第一版把 5 张卡横排在底部一条 band 里，留档于
figures_v7/fig1_map_flow_band_variant.*，正式版为 fig1_map_flow.*。

本脚本只负责「不可重建的部分」：
  1. fig1_map_base.png —— 纯地理底图 + 数据点/迁徙方向场，**不含任何文字、
     标题、图例、色条、边框标注**。这些在 PowerPoint 里作为原生可编辑对象绘制，
     因此本 PNG 是一个原子不可再分的栅格单元。
  2. fig1_flow_data.json —— 流程图节点里要用的真实数值（原生图表/文字用），
     全部来自 figure_style.compute_metrics()，与 final_numbers.txt 同源。

用法：python tests/generate_fig1_v7_base.py
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, CMAP_CONC, THETAS,
                          circ, on_Evec, off_Evec)
from generate_paper_figures_v2 import RADAR_LOC, LAND_F, OCEAN_F
from generate_paper_figures_v6 import _capture_curves_iqr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, '..', 'figures_v7')
os.makedirs(OUT, exist_ok=True)

# --- 画布几何：Nature 双栏宽 7.09 in；地图 extent 的经纬跨度决定高度 --------
FIG_W = 7.09
EXTENT = [-14.0, 21.0, 31.0, 63.0]          # lon_min, lon_max, lat_min, lat_max
LON_SPAN = EXTENT[1] - EXTENT[0]            # 35°
LAT_SPAN = EXTENT[3] - EXTENT[2]            # 32°
FIG_H = FIG_W * LAT_SPAN / LON_SPAN         # PlateCarree 等纵横比 -> 6.482 in
# 南界压到 31°N：数据实测只占 lon -5.7..18.8 / lat 41.2..56.8，把南界放到
# 31°N 后，31–41°N 与 lon<-6、lon>19 一起构成一圈不含任何研究数据的外环，
# 供 b–f 流程卡环绕落位；地图本身仍是 full-bleed，且无一个数据点被遮挡。
DPI = 600

# 方向集中度色标：v6 用的 0.55-0.90 是错的——格网 spring_conc 实测域为
# 0.841-0.989，旧色标下段 80% 从未被用到、上段还把 >0.9 的格子截断成同一个黄。
# 这里改成 0.84-1.00（覆盖实测域，且刻度落在 0.85/0.90/0.95/1.00 等距整数上），
# 方向集中度的空间差异才真的画得出来。
CONC_LO, CONC_HI = 0.84, 1.00
C_RADAR = '#C0392B'


def _lonlat_to_frac(lon, lat):
    """经纬度 -> 画布归一化坐标（左下原点），供 PPT 定位流程图节点时对齐地图要素。"""
    return ((lon - EXTENT[0]) / LON_SPAN, (lat - EXTENT[2]) / LAT_SPAN)


def build_base_map(ctx):
    """渲染 full-bleed 地理底图（无任何文字元素）。"""
    on1 = ctx['on1']; od = ctx['od']; grid = ctx['grid']
    vpts = od[od.source == 'VPTS']
    bauer = od[od.source == 'Bauer_grid']

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor=LAND_F, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_F, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#9A9A9A', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor='#C2C2C2', zorder=1)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # 春季迁徙方向场（抽稀，颜色 = 方向集中度）
    gsub = grid[(grid.row % 2 == 0) & (grid.col % 2 == 0)]
    u = np.sin(np.radians(gsub.spring_dir.values))
    v = np.cos(np.radians(gsub.spring_dir.values))
    q = ax.quiver(gsub.lon.values, gsub.lat.values, u, v, gsub.spring_conc.values,
                  cmap=CMAP_CONC, scale=40, width=0.0020, headwidth=2.6,
                  headlength=3.0, alpha=0.72,
                  transform=ccrs.PlateCarree(), zorder=3)
    q.set_clim(CONC_LO, CONC_HI)

    # 风电场：陆上（密集小点）/ 海上 VPTS / 海上 Bauer / 雷达站
    ax.scatter(on1.centroid_lon, on1.centroid_lat, s=0.9, c=C_LAND, alpha=0.40,
               edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)
    ax.scatter(vpts.centroid_lon, vpts.centroid_lat, s=22, c=C_VPTS, marker='o',
               edgecolors='white', linewidths=0.4,
               transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(bauer.centroid_lon, bauer.centroid_lat, s=20, c=C_BAUER, marker='s',
               edgecolors='white', linewidths=0.4,
               transform=ccrs.PlateCarree(), zorder=5)
    for _sn, (la, lo) in RADAR_LOC.items():
        ax.scatter(lo, la, marker='^', s=30, c=C_RADAR, edgecolors='white',
                   linewidths=0.4, transform=ccrs.PlateCarree(), zorder=6)

    ax.set_frame_on(False)
    png = os.path.join(OUT, 'fig1_map_base.png')
    fig.savefig(png, dpi=DPI, facecolor='white', pad_inches=0)
    plt.close(fig)
    return png


def colorbar_stops(n=24):
    """导出 viridis 色条的离散色标，供 PPT 用原生矩形逐格重建（不栅格化色条）。"""
    cmap = plt.get_cmap(CMAP_CONC)
    stops = []
    for i in range(n):
        t = (i + 0.5) / n
        r, g, b, _ = cmap(t)
        stops.append('#%02X%02X%02X' % (int(r * 255), int(g * 255), int(b * 255)))
    return stops


def capture_curves(on_df, vp_df, ba_df, ctx):
    """节点④：旋转 Δθ -> 已捕获的最大暴露削减份额（%），三组中位数曲线。"""
    out = {}
    for name, df, is_on in [('Onshore', on_df, True),
                            ('VPTS', vp_df, False),
                            ('Bauer', ba_df, False)]:
        dths, mat = _capture_curves_iqr(df, ctx['on1'], ctx['cur'], is_on)
        out[name] = dict(
            dtheta=[int(d) for d in dths],
            median=[round(float(v), 2) for v in np.nanpercentile(mat, 50, axis=0)],
            q25=[round(float(v), 2) for v in np.nanpercentile(mat, 25, axis=0)],
            q75=[round(float(v), 2) for v in np.nanpercentile(mat, 75, axis=0)],
        )
    return out


def pareto_points(ctx):
    """节点⑤：能源-暴露权衡，三组在 0.5/1/2/5% 预算下的中位数 (AEP loss, 暴露削减)。"""
    on = ctx['on']
    to = ctx['to'].merge(ctx['od'][['farm_id', 'source']], on='farm_id', how='left')
    out = {}
    for name in ('Onshore', 'VPTS', 'Bauer'):
        pts = []
        for b in (0.005, 0.01, 0.02, 0.05):
            if name == 'Onshore':
                sub = on[on.budget == b]
                x, y = sub.aep_cost_pct, sub.risk_reduction
            else:
                src = 'VPTS' if name == 'VPTS' else 'Bauer_grid'
                sub = to[(to.budget == b) & (to.source == src)]
                x, y = sub.aep_cost_pct, sub.risk_reduction_pct
            if len(sub) == 0:
                continue
            pts.append(dict(budget_pct=b * 100,
                            aep_loss=round(float(np.median(x)), 3),
                            reduction=round(float(np.median(y)), 2),
                            n=int(len(sub))))
        out[name] = pts
    return out


# --- PPT 环绕布局（pt，原点左上；72 pt = 1 in）------------------------------
# 定稿几何，随 JSON 一起导出，便于底图重绘后按同一坐标重建 PowerPoint 版面。
# 每个矩形都经过“数据像素掩膜”验证：饱和度 >0.30 的像素（迁徙箭头/场址点/雷达）
# 在框内计数为 0，即卡片不遮挡任何一个数据要素。
RING_LAYOUT = dict(
    canvas_pt=dict(width=510.5, height=466.75),
    pt_per_deg=dict(lon=510.5 / LON_SPAN, lat=466.75 / LAT_SPAN),
    map_image=dict(left=0, top=0, width=510.5, height=466.75),
    annotations=dict(
        panel_label_a=dict(left=7, top=3),
        map_title=dict(left=22, top=4),
        legend_rows_top=[20, 31, 42, 53],
        chain_header=dict(left=8, top=67, width=234),
        colorbar=dict(left=350, top=26, width=140, height=7,
                      tick_centers={0.85: 358.75, 0.90: 402.5,
                                    0.95: 446.25, 1.00: 490.0}),
    ),
    # 环绕顺序：b 左上 -> c 左下 -> d 底左 -> e 底中 -> f 右侧（折上）
    cards=dict(
        b=dict(left=8,   top=84,  width=76,     height=118),
        c=dict(left=8,   top=214, width=76,     height=118),
        d=dict(left=124, top=340, width=78,     height=118),
        e=dict(left=240, top=340, width=101.25, height=118),
        f=dict(left=400, top=240, width=101.25, height=118),
    ),
    arrows=dict(
        bc=dict(x0=46,  y0=203.5, x1=46,  y1=212.5),
        cd=dict(x0=50,  y0=334.0, x1=120, y1=388.0),
        de=dict(x0=204, y0=399.0, x1=238, y1=399.0),
        ef=dict(x0=345, y0=392.0, x1=396, y1=354.0),
    ),
)


def headline_numbers(on_df, vp_df, ba_df, ctx):
    """流程图节点①②③要引用的关键数字，全部现算，不写死。"""
    stats = {}
    for name, df in [('Onshore', on_df), ('VPTS', vp_df), ('Bauer', ba_df)]:
        stats[name] = dict(
            n=int(len(df)),
            misalign_median=round(float(df.d_full.median()), 1),
            misalign_q25=round(float(df.d_full.quantile(0.25)), 1),
            misalign_q75=round(float(df.d_full.quantile(0.75)), 1),
            theta_econ_median=round(float(df.theta_econ.median()), 1),
            theta_eco_median=round(float(df.th_min.median()), 1),
            avoidable_median_pct=round(float(df.avoid.median() * 100), 1),
            d50_median=round(float(df.d50.median()), 1),
            d80_median=round(float(df.d80.median()), 1),
            frac20_median_pct=round(float(df.frac20.median() * 100), 1),
            rr_at_1pct_median=round(float(df.rr.median()), 1),
            aep_at_1pct_median=round(float(df.aep.median()), 3),
        )
    # 迁徙方向场：格网 spring_dir 的圆均值与集中度范围
    grid = ctx['grid']
    ang = np.radians(grid.spring_dir.values)
    mean_dir = float(np.degrees(np.arctan2(np.nanmean(np.sin(ang)),
                                           np.nanmean(np.cos(ang)))) % 360)
    stats['migration_field'] = dict(
        n_cells=int(len(grid)),
        mean_spring_dir=round(mean_dir, 1),
        conc_median=round(float(np.nanmedian(grid.spring_conc)), 3),
        conc_min=round(float(np.nanmin(grid.spring_conc)), 3),
        conc_max=round(float(np.nanmax(grid.spring_conc)), 3),
    )
    stats['n_radar'] = len(RADAR_LOC)
    return stats


def main():
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}')

    print('Rendering full-bleed base map ...')
    png = build_base_map(ctx)
    print(f'  -> {png}  ({os.path.getsize(png)/1e6:.2f} MB, '
          f'{FIG_W:.2f} x {FIG_H:.3f} in @ {DPI} dpi)')

    payload = dict(
        canvas=dict(width_in=FIG_W, height_in=round(FIG_H, 4), dpi=DPI,
                    extent=EXTENT, lon_span=LON_SPAN, lat_span=LAT_SPAN),
        colorbar=dict(label='Direction concentration', vmin=CONC_LO, vmax=CONC_HI,
                      ticks=[0.85, 0.90, 0.95, 1.00], stops=colorbar_stops()),
        legend=[
            dict(label='Onshore wind farm', color=C_LAND, marker='o'),
            dict(label='Offshore (VPTS)', color=C_VPTS, marker='o'),
            dict(label='Offshore (Bauer)', color=C_BAUER, marker='s'),
            dict(label='Weather radar', color=C_RADAR, marker='^'),
        ],
        capture_curves=capture_curves(on_df, vp_df, ba_df, ctx),
        pareto=pareto_points(ctx),
        stats=headline_numbers(on_df, vp_df, ba_df, ctx),
        ring_layout=RING_LAYOUT,
    )
    jp = os.path.join(OUT, 'fig1_flow_data.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'  -> {jp}')

    s = payload['stats']
    print('\nKey numbers for the flowchart nodes:')
    for g in ('Onshore', 'VPTS', 'Bauer'):
        d = s[g]
        print(f"  {g:8s} n={d['n']:5d}  misalign={d['misalign_median']:5.1f}deg  "
              f"avoidable={d['avoidable_median_pct']:5.1f}%  d50={d['d50_median']:4.1f}deg  "
              f"frac20={d['frac20_median_pct']:5.1f}%  RR@1%={d['rr_at_1pct_median']:5.1f}%")
    print(f"  migration field: n={s['migration_field']['n_cells']} cells, "
          f"mean spring dir={s['migration_field']['mean_spring_dir']}deg, "
          f"conc median={s['migration_field']['conc_median']}")


if __name__ == '__main__':
    main()
