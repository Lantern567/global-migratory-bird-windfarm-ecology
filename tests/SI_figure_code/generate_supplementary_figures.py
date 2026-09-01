# -*- coding: utf-8 -*-
"""
generate_supplementary_figures.py —— 中文主稿补充材料制图（S1–S8 全部图片）。

用途：为《风电场阵列朝向_补充材料》生成全部图片（PDF 矢量 + 300dpi PNG），
     复用 figure_style.compute_metrics()（口径 4191/29/26，与 final_numbers.txt 一致），
     读取 sensitivity_compute.py 产出的 sensitivity_summary.json 与逐场 CSV。

铁律：所有数字来自当前数据（compute_metrics / sensitivity_summary.json），
     不引用任何废弃口径（41/62/78、70.4%、*_57 等）。

图内文字一律英文（学长要求），正文仍中文；色盲安全 Okabe-Ito 色板。
输出目录：our_work/figures_supp/（与 figures_v2/v4 同风格）。
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figure_style as fs
from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN, C_ECON, C_ECO,
                          GROUP_COLORS, GROUP_ORDER, THETAS, style_ax, panel_label,
                          circ, on_Evec, off_Evec, analyze)

BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, '..', 'data', 'processed')
FIG_SUPP = os.path.join(BASE, '..', 'figures_supp')
FIG_V2 = os.path.join(BASE, '..', 'figures_v2')
FIG_V4 = os.path.join(BASE, '..', 'figures_v4')
os.makedirs(FIG_SUPP, exist_ok=True)

ORIENT = np.arange(0, 180, 10)      # 18 角度格（AEP/权衡同口径）
BUDGETS = [0.005, 0.01, 0.02, 0.05]  # 四档 AEP 预算
BUDGET_LBL = ['0.5%', '1%', '2%', '5%']


def save_supp(fig, name):
    """同时导出 PDF（矢量）与 PNG（300dpi，Word 用），输出到 figures_supp/。"""
    pdf = os.path.join(FIG_SUPP, name + '.pdf')
    png = os.path.join(FIG_SUPP, name + '.png')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)
    return png


def load_summary():
    with open(os.path.join(PROC, 'sensitivity_summary.json'), encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 复用已存在的图（figures_v2 / figures_v4 已生成，直接复制并统一命名）
# ---------------------------------------------------------------------------
def reuse_existing_figures():
    """把 4 张已定稿的补充图复制进 figures_supp/ 并统一编号。

    图 S1 = 研究区（fig0_study_area）
    图 S2 = 候鸟威胁（figS1_threat）
    图 S3 = 跨源普遍性（figS2_R1）
    图 S6 = 论证链总览（fig0_overview）
    """
    mapping = [
        (os.path.join(FIG_V2, 'fig0_study_area'), 'figS1_study_area'),
        (os.path.join(FIG_V2, 'figS1_threat'), 'figS2_threat'),
        (os.path.join(FIG_V4, 'figS2_R1'), 'figS3_universality'),
        (os.path.join(FIG_V2, 'fig0_overview'), 'figS6_overview'),
    ]
    for src, dst in mapping:
        for ext in ('png', 'pdf'):
            s = src + '.' + ext
            if os.path.exists(s):
                shutil.copy(s, os.path.join(FIG_SUPP, dst + '.' + ext))


def rep_onshore(on_df, ctx):
    """返回代表陆上场（avoid 最接近中位）：逐场指标 row、元数据 meta、E(θ) 曲线。"""
    idx = (on_df.avoid - on_df.avoid.median()).abs().idxmin()
    row = on_df.loc[idx]
    meta = ctx['on1'][ctx['on1'].farm_id == row.farm_id].iloc[0]
    E = on_Evec(meta.spring_dir, meta.autumn_dir, meta.spring_conc, meta.autumn_conc)
    return row, meta, E


def rep_onshore_aep(farm_id, ctx):
    """返回代表陆上场的 AEP(θ) 曲线（18 角度，对应 ORIENT）。"""
    sub = ctx['on_aep'][ctx['on_aep'].farm_id == farm_id]
    if len(sub) == 0:
        return None
    aep = np.array([sub.iloc[0][f'aep_{a:03d}'] for a in ORIENT], dtype=float)
    return aep


# ---------------------------------------------------------------------------
# 图 S4 —— 几何暴露模型 E(θ) 说明 + 尾流 AEP 极坐标（S1 + S7.10）
# ---------------------------------------------------------------------------
def figS4_exposure_model(on_df, ctx):
    row, meta, E = rep_onshore(on_df, ctx)
    aep = rep_onshore_aep(meta.farm_id, ctx)

    ws = meta.spring_conc / (meta.spring_conc + meta.autumn_conc)
    wa = 1 - ws
    sp = np.sin(np.radians(THETAS - meta.spring_dir)) ** 2
    au = np.sin(np.radians(THETAS - meta.autumn_dir)) ** 2

    fig = plt.figure(figsize=(12.0, 4.0))
    gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.07, right=0.96, top=0.88, bottom=0.16)

    # (a) 两季 sin² 分量 + 浓度加权 E(θ)
    axa = fig.add_subplot(gs[0, 0])
    axa.plot(THETAS, sp, color=C_SPRING, lw=1.4, label='Spring component')
    axa.plot(THETAS, au, color=C_AUTUMN, lw=1.4, label='Autumn component')
    axa.plot(THETAS, E, color=C_ECON, lw=2.0, label='E($\\theta$) weighted')
    axa.set_xlabel('Array orientation $\\theta$ (deg)')
    axa.set_ylabel('Exposure (normalized)')
    axa.set_title('Seasonal components & weighted E($\\theta$)', fontsize=9)
    axa.set_xlim(0, 180); axa.set_ylim(0, 1.02)
    axa.legend(loc='upper right', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) E(θ) 全曲线 + θ_econ / θ_eco + 可避免区
    axb = fig.add_subplot(gs[0, 1])
    axb.plot(THETAS, E, color=C_ECON, lw=2.0)
    tecon = meta.theta_econ % 180
    teco = row.theta_eco % 180
    Ee = E[int(round(tecon)) % 180]
    Eeco = E[int(round(teco)) % 180]
    axb.axvline(tecon, color=C_ECON, ls='--', lw=1.2)
    axb.axvline(teco, color=C_ECO, ls='-', lw=1.2)
    axb.plot(tecon, Ee, 'o', color=C_ECON, ms=6, zorder=5)
    axb.plot(teco, Eeco, 'o', color=C_ECO, ms=6, zorder=5)
    axb.annotate('', xy=(teco, Eeco), xytext=(tecon, Ee),
                 arrowprops=dict(arrowstyle='<->', color=C_ECO, lw=1.2))
    axb.set_xlabel('Array orientation $\\theta$ (deg)')
    axb.set_ylabel('Exposure (normalized)')
    axb.set_title('$\\theta_{econ}$ vs $\\theta_{eco}$ (avoidable exposure)', fontsize=9)
    axb.set_xlim(0, 180); axb.set_ylim(0, 1.02)
    handles = [
        Line2D([], [], color=C_ECON, lw=2.0, label='E($\\theta$)'),
        Line2D([], [], color=C_ECON, lw=1.2, ls='--', label='$\\theta_{econ}$'),
        Line2D([], [], color=C_ECO, lw=1.2, label='$\\theta_{eco}$'),
    ]
    axb.legend(handles=handles, loc='upper right', fontsize=7, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    # (c) AEP(θ) 极坐标（尾流模型，θ_econ = argmax AEP）
    axc = fig.add_subplot(gs[0, 2])
    if aep is not None:
        axc.plot(ORIENT, aep / aep.max(), color=C_LAND, lw=1.8)
        iecon = int(np.argmax(aep))
        axc.plot(ORIENT[iecon], 1.0, 'o', color=C_ECO, ms=7, zorder=5)
        axc.axhline(1.0, color='#999999', ls=':', lw=0.8)
        axc.set_xlabel('Array orientation $\\theta$ (deg)')
        axc.set_ylabel('AEP (normalized)')
        axc.set_title('Wake-modelled AEP($\\theta$) — $\\theta_{econ}$=argmax', fontsize=9)
        axc.set_xlim(0, 170)
    else:
        axc.text(0.5, 0.5, 'AEP curve unavailable', ha='center', va='center',
                 transform=axc.transAxes, color='#999999')
        axc.set_axis_off()
    panel_label(axc, 'c'); style_ax(axc)

    save_supp(fig, 'figS4_exposure_model')


# ---------------------------------------------------------------------------
# 图 S5 —— 预算敏感性（RR 与 AEP 代价随预算，四档 × 三组）
# ---------------------------------------------------------------------------
def _budget_medians(ctx):
    on = ctx['on']; to = ctx['to']
    vp_ids = ctx['vpts_ids']; ba_ids = ctx['bauer_ids']
    tovp = to[to.farm_id.isin(vp_ids)]
    toba = to[to.farm_id.isin(ba_ids)]
    rr = {'Onshore': [on[on.budget == b].risk_reduction.median() for b in BUDGETS],
          'VPTS': [tovp[tovp.budget == b].risk_reduction_pct.median() for b in BUDGETS],
          'Bauer': [toba[toba.budget == b].risk_reduction_pct.median() for b in BUDGETS]}
    aep = {'Onshore': [on[on.budget == b].aep_cost_pct.median() for b in BUDGETS],
           'VPTS': [tovp[tovp.budget == b].aep_cost_pct.median() for b in BUDGETS],
           'Bauer': [toba[toba.budget == b].aep_cost_pct.median() for b in BUDGETS]}
    return rr, aep


def figS5_budget(ctx):
    rr, aep = _budget_medians(ctx)
    x = np.arange(len(BUDGETS))

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.86, bottom=0.14)

    axa = fig.add_subplot(gs[0, 0])
    for grp in GROUP_ORDER:
        axa.plot(x, rr[grp], marker='o', ms=5, lw=1.6, color=GROUP_COLORS[grp], label=grp)
    axa.set_xticks(x); axa.set_xticklabels(BUDGET_LBL)
    axa.set_xlabel('AEP budget')
    axa.set_ylabel('Median risk reduction (%)')
    axa.set_title('RR vs budget', fontsize=9)
    axa.set_ylim(0, 100)
    axa.legend(loc='lower right', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    for grp in GROUP_ORDER:
        axb.plot(x, aep[grp], marker='s', ms=5, lw=1.6, color=GROUP_COLORS[grp], label=grp)
    axb.set_xticks(x); axb.set_xticklabels(BUDGET_LBL)
    axb.set_xlabel('AEP budget')
    axb.set_ylabel('Median AEP cost (%)')
    axb.set_title('AEP cost vs budget', fontsize=9)
    axb.legend(loc='upper left', fontsize=7, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS5_budget')


# ---------------------------------------------------------------------------
# 图 S7.1 + S7.2 —— 尾流惩罚放大 与 衰减常数 α
# ---------------------------------------------------------------------------
def figS7_1_wake_alpha():
    SUM = load_summary()
    wp = pd.read_csv(os.path.join(PROC, 'sensitivity_wake_penalty_perfarm.csv'), encoding='utf-8-sig')
    cols = ['rr_1.0', 'rr_1.5', 'rr_2.0', 'rr_3.0']
    ks = ['1.0', '1.5', '2.0', '3.0']

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.86, bottom=0.14)

    # (a) 尾流惩罚放大 k —— 逐场 RR 小提琴
    axa = fig.add_subplot(gs[0, 0])
    data = [wp[c].dropna().values for c in cols]
    vp = axa.violinplot(data, positions=np.arange(len(cols)), showmedians=True, widths=0.7)
    for body in vp['bodies']:
        body.set_facecolor(C_LAND); body.set_alpha(0.45); body.set_edgecolor(C_LAND)
    for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp[part].set_color(C_ECON); vp[part].set_linewidth(1.0)
    meds = [SUM['S71_wake_penalty_rr_med'][k] for k in ks]
    axa.plot(np.arange(len(cols)), meds, 'o-', color=C_ECO, lw=1.8, ms=5, zorder=5, label='Median')
    axa.set_xticks(np.arange(len(cols))); axa.set_xticklabels([f'{k}×' for k in ks])
    axa.set_xlabel('Wake-penalty scale factor k')
    axa.set_ylabel('Per-farm risk reduction (%)')
    axa.set_title('Wake penalty amplification', fontsize=9)
    axa.set_ylim(0, 100)
    axa.legend(loc='lower left', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 衰减常数 α 0.075 → 0.05
    axb = fig.add_subplot(gs[0, 1])
    a075 = SUM['S72_alpha_075_rr_med']
    a050 = SUM['S72_alpha_050_rr_med']
    bars = axb.bar([0, 1], [a075, a050], 0.5, color=C_LAND, edgecolor='none')
    bars[0].set_alpha(0.9); bars[1].set_alpha(0.5)
    for i, v in enumerate([a075, a050]):
        axb.text(i, v + 1.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8, color='black')
    axb.set_xticks([0, 1]); axb.set_xticklabels(['α = 0.075', 'α = 0.05'])
    axb.set_ylabel('Median risk reduction (%)')
    axb.set_title('Wake decay constant α', fontsize=9)
    axb.set_ylim(0, 108)
    axb.annotate(f'Δ = {SUM["S72_alpha_drop_pp"]:.1f} pp', xy=(0.5, 0.5), xytext=(0.62, 0.72),
                 textcoords='axes fraction', fontsize=8, color=C_ECO,
                 arrowprops=dict(arrowstyle='->', color=C_ECO, lw=0.9))
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS7_1_wake_alpha')


# ---------------------------------------------------------------------------
# 图 S7.3 —— ERA5 不可替代性（平均角差 + 30° 内占比）
# ---------------------------------------------------------------------------
def figS7_3_era5():
    SUM = load_summary()
    e = SUM['S73_era5']
    labels = ['VPTS\nspring', 'VPTS\nautumn', 'Bauer\nspring', 'Bauer\nautumn']
    mean360 = [e['S73_VPTS_spring_mean360'], e['S73_VPTS_autumn_mean360'],
               e['S73_Bauer_spring_mean360'], e['S73_Bauer_autumn_mean360']]
    within30 = [e['S73_VPTS_spring_within30'], e['S73_VPTS_autumn_within30'],
                e['S73_Bauer_spring_within30'], e['S73_Bauer_autumn_within30']]
    x = np.arange(4)
    colors = [C_SPRING, C_AUTUMN, C_SPRING, C_AUTUMN]

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.09, right=0.96, top=0.86, bottom=0.16)

    axa = fig.add_subplot(gs[0, 0])
    axa.bar(x, mean360, 0.55, color=colors, alpha=0.85, edgecolor='none')
    axa.axhline(30, color='#999999', ls='--', lw=1.0)
    axa.text(3.4, 32, '30°', fontsize=7, color='#666666', ha='right')
    for i, v in enumerate(mean360):
        axa.text(i, v + 2, f'{v:.0f}°', ha='center', va='bottom', fontsize=8)
    axa.set_xticks(x); axa.set_xticklabels(labels, fontsize=7)
    axa.set_ylabel('Mean circular error (deg)')
    axa.set_title('ERA5 vs measured direction', fontsize=9)
    axa.set_ylim(0, 125)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    axb.bar(x, within30, 0.55, color=colors, alpha=0.85, edgecolor='none')
    for i, v in enumerate(within30):
        axb.text(i, v + 1.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=7)
    axb.set_ylabel('Fraction within 30° (%)')
    axb.set_title('Agreement within 30°', fontsize=9)
    axb.set_ylim(0, 60)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS7_3_era5')


# ---------------------------------------------------------------------------
# 图 S7.4 —— 海上方向口径 VPTS vs Bauer（极坐标）
# ---------------------------------------------------------------------------
def _polar_arrow(ax, deg, color, label, lw=2.0):
    ax.annotate('', xy=(np.deg2rad(deg), 1.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, shrinkA=0, shrinkB=0))
    ax.plot(np.deg2rad(deg), 1.0, 'o', color=color, ms=7, zorder=5, label=label)


def figS7_4_offshore():
    SUM = load_summary()
    s = SUM['S74_offshore_convention']
    fig = plt.figure(figsize=(8.4, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.02, left=0.04, right=0.96, top=0.88, bottom=0.08)

    for i, (season, vpts_deg, bauer_deg, div) in enumerate([
        ('spring', s['S74_spring_vpts_med'], s['S74_spring_bauer_med'], s['S74_spring_divergence']),
        ('autumn', s['S74_autumn_vpts_med'], s['S74_autumn_bauer_med'], s['S74_autumn_divergence']),
    ]):
        ax = fig.add_subplot(gs[0, i], projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        _polar_arrow(ax, vpts_deg, C_VPTS, 'VPTS')
        _polar_arrow(ax, bauer_deg, C_BAUER, 'Bauer')
        ax.set_ylim(0, 1.15)
        ax.set_yticklabels([])
        ax.set_title(f'{season.capitalize()}  (Δ = {div:.1f}°)', fontsize=9, pad=18)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.10), fontsize=7, frameon=False)
        ax.grid(color='#CCCCCC', lw=0.5)
        panel_label(ax, 'a' if i == 0 else 'b', x=0.02, y=0.02, fs=12, va='bottom')

    fig.suptitle('Offshore migration direction: VPTS (radar) vs Bauer (grid)', fontsize=9, y=0.98)
    save_supp(fig, 'figS7_4_offshore')


# ---------------------------------------------------------------------------
# 图 S7.5 —— 季节加权 集中度 vs 等权（RR 差分布）
# ---------------------------------------------------------------------------
def figS7_5_seasonal():
    SUM = load_summary()
    sw = pd.read_csv(os.path.join(PROC, 'sensitivity_seasonal_weight_perfarm.csv'), encoding='utf-8-sig')
    d = sw['diff_pp'].dropna().values

    fig = plt.figure(figsize=(5.6, 4.0))
    ax = fig.add_subplot(111)
    ax.hist(d, bins=60, color=C_LAND, alpha=0.7, edgecolor='none')
    ax.axvline(0, color=C_ECO, lw=1.4)
    ax.axvline(SUM['S75_seasonal_weight']['diff_mean_pp'], color='#999999', ls='--', lw=1.2)
    ax.text(0.03, 0.94, f"median = {SUM['S75_seasonal_weight']['diff_med_pp']:.2f} pp\n"
                        f"mean = {SUM['S75_seasonal_weight']['diff_mean_pp']:.2f} pp\n"
                        f"n = {SUM['S75_seasonal_weight']['n']}",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC'))
    ax.set_xlabel('RR$_{conc}$ − RR$_{equal}$ (percentage points)')
    ax.set_ylabel('Number of farms')
    ax.set_title('Seasonal weighting: concentration vs equal', fontsize=9)
    style_ax(ax)
    save_supp(fig, 'figS7_5_seasonal')


# ---------------------------------------------------------------------------
# 图 S7.6 —— DBSCAN eps × min_samples 扫描
# ---------------------------------------------------------------------------
def figS7_6_dbscan():
    SUM = load_summary()
    rows = SUM['S76_dbscan']
    epss = sorted({r['eps_km'] for r in rows})
    mss = sorted({r['min_samples'] for r in rows})
    M = np.zeros((len(mss), len(epss)))
    for r in rows:
        i = mss.index(r['min_samples']); j = epss.index(r['eps_km'])
        M[i, j] = r['n_clusters']

    fig = plt.figure(figsize=(5.8, 4.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, cmap='cividis', aspect='auto')
    for i in range(len(mss)):
        for j in range(len(epss)):
            txt = f'{int(M[i, j])}'
            col = 'white' if M[i, j] > (M.max() + M.min()) / 2 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=col)
    # 高亮实际采用格 (eps=3, min_samples=2)
    i_use, j_use = mss.index(2), epss.index(3)
    ax.add_patch(plt.Rectangle((j_use - 0.5, i_use - 0.5), 1, 1, fill=False,
                               edgecolor=C_ECO, lw=2.5))
    ax.set_xticks(range(len(epss))); ax.set_xticklabels([f'{e} km' for e in epss])
    ax.set_yticks(range(len(mss))); ax.set_yticklabels([f'{m}' for m in mss])
    ax.set_xlabel('eps (km)'); ax.set_ylabel('min_samples')
    ax.set_title('DBSCAN cluster count — highlighted cell used', fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Number of clusters', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    style_ax(ax)
    save_supp(fig, 'figS7_6_dbscan')


# ---------------------------------------------------------------------------
# 图 S7.7 —— Bauer 匹配半径扫描
# ---------------------------------------------------------------------------
def figS7_7_radius():
    SUM = load_summary()
    rows = SUM['S77_match_radius']
    radii = [r['radius_km'] for r in rows]
    nfarms = [r['n_with_aep'] for r in rows]
    rrmed = [r['rr_med_1pct'] for r in rows]

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.09, right=0.96, top=0.86, bottom=0.15)

    axa = fig.add_subplot(gs[0, 0])
    axa.plot(radii, nfarms, 'o-', color=C_LAND, lw=1.8, ms=6)
    axa.scatter([200], [4191], s=90, facecolors='none', edgecolors=C_ECO, lw=1.6, zorder=5)
    for r, n in zip(radii, nfarms):
        axa.text(r, n + 60, f'{n:,}', ha='center', fontsize=7.5)
    axa.set_xlabel('Match radius (km)')
    axa.set_ylabel('Farms with AEP (n)')
    axa.set_title('Matching radius vs sample size', fontsize=9)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    axb.plot(radii, rrmed, 's-', color=C_VPTS, lw=1.8, ms=6)
    axb.scatter([200], [96.9], s=90, facecolors='none', edgecolors=C_ECO, lw=1.6, zorder=5)
    for r, v in zip(radii, rrmed):
        axb.text(r, v + 0.05, f'{v:.1f}%', ha='center', fontsize=7.5)
    axb.set_xlabel('Match radius (km)')
    axb.set_ylabel('Median RR at 1% budget (%)')
    axb.set_title('Matching radius vs RR', fontsize=9)
    axb.set_ylim(96.0, 98.0)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS7_7_radius')


# ---------------------------------------------------------------------------
# 图 S7.11 —— 陆上风场规模分层
# ---------------------------------------------------------------------------
def figS7_11_size():
    SUM = load_summary()
    rows = SUM['S711_stratification']
    keys = [r['key'] for r in rows]
    n = [r['n'] for r in rows]
    rr_med = [r['rr_med'] for r in rows]
    rr_mean = [r['rr_mean'] for r in rows]
    x = np.arange(len(keys))

    fig = plt.figure(figsize=(8.0, 4.2))
    ax = fig.add_subplot(111)
    w = 0.36
    ax.bar(x - w / 2, rr_med, w, color=C_LAND, alpha=0.9, edgecolor='none', label='Median RR')
    ax.bar(x + w / 2, rr_mean, w, color=C_VPTS, alpha=0.9, edgecolor='none', label='Mean RR')
    for i, (m, mn) in enumerate(zip(rr_med, rr_mean)):
        ax.text(i - w / 2, m + 1, f'{m:.1f}', ha='center', fontsize=7)
        ax.text(i + w / 2, mn + 1, f'{mn:.1f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f'{k}\n(n={ni:,})' for k, ni in zip(keys, n)], fontsize=7)
    ax.set_xlabel('Farm size (turbines)')
    ax.set_ylabel('Risk reduction (%)')
    ax.set_title('Stratification by farm size', fontsize=9)
    ax.set_ylim(0, 108)
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    style_ax(ax)
    save_supp(fig, 'figS7_11_size')


# ---------------------------------------------------------------------------
# 图 S7.12 —— PCA 方差解释率分布
# ---------------------------------------------------------------------------
def figS7_12_pca():
    SUM = load_summary()
    p = SUM['S712_pca_explained_var']
    pca = pd.read_csv(os.path.join(PROC, 'osm_farm_pca_orientations.csv'), encoding='utf-8-sig')
    evr = pca['explained_var_ratio'].dropna().values

    fig = plt.figure(figsize=(5.8, 4.0))
    ax = fig.add_subplot(111)
    ax.hist(evr, bins=50, color=C_BAUER, alpha=0.7, edgecolor='none')
    ax.axvline(p['mean'], color=C_ECO, lw=1.4, label=f"mean = {p['mean']:.3f}")
    ax.axvline(p['med'], color='#333333', ls='--', lw=1.2, label=f"median = {p['med']:.3f}")
    ax.set_xlabel('Explained variance ratio (PC1)')
    ax.set_ylabel('Number of farms')
    ax.set_title(f"PCA orientation fit (n = {p['n']:,})", fontsize=9)
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    style_ax(ax)
    save_supp(fig, 'figS7_12_pca')


# ---------------------------------------------------------------------------
# 图 S7.13 —— 陆上 vs 海上 AEP 朝向敏感性
# ---------------------------------------------------------------------------
def figS7_13_aep():
    SUM = load_summary()
    s = SUM['S713_aep_sens']
    groups = ['Onshore', 'Offshore']
    mean_v = [s['onshore_mean'], s['offshore_mean']]
    med_v = [s['onshore_med'], s['offshore_med']]

    fig = plt.figure(figsize=(5.6, 4.0))
    ax = fig.add_subplot(111)
    x = np.arange(2); w = 0.34
    ax.bar(x - w / 2, mean_v, w, color=C_LAND, alpha=0.9, edgecolor='none', label='Mean')
    ax.bar(x + w / 2, med_v, w, color=C_VPTS, alpha=0.9, edgecolor='none', label='Median')
    for i, (m, mn) in enumerate(zip(mean_v, med_v)):
        ax.text(i - w / 2, m + 0.2, f'{m:.2f}%', ha='center', fontsize=8)
        ax.text(i + w / 2, mn + 0.2, f'{mn:.2f}%', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel('AEP sensitivity (max−min / max, %)')
    ax.set_title('Orientation sensitivity of AEP', fontsize=9)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    style_ax(ax)
    save_supp(fig, 'figS7_13_aep')


# ---------------------------------------------------------------------------
# 图 S8.1 —— 权衡散点（AEP 代价 vs RR，三组）
# ---------------------------------------------------------------------------
def figS8_1_tradeoff(on_df, vp_df, ba_df):
    fig = plt.figure(figsize=(5.8, 4.4))
    ax = fig.add_subplot(111)
    ax.scatter(on_df.aep, on_df.rr, s=3, c=C_LAND, alpha=0.20, edgecolors='none',
               label=f'Onshore (n={len(on_df)})')
    ax.scatter(vp_df.aep, vp_df.rr, s=28, c=C_VPTS, alpha=0.9, edgecolors='white',
               linewidths=0.4, label=f'VPTS (n={len(vp_df)})', zorder=3)
    ax.scatter(ba_df.aep, ba_df.rr, s=28, c=C_BAUER, alpha=0.9, edgecolors='white',
               linewidths=0.4, label=f'Bauer (n={len(ba_df)})', zorder=3)
    ax.set_xlabel('AEP cost at 1% budget (%)')
    ax.set_ylabel('Risk reduction at 1% budget (%)')
    ax.set_title('Energy–ecology trade-off (per farm)', fontsize=9)
    ax.set_xlim(-0.5, 5); ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    style_ax(ax)
    save_supp(fig, 'figS8_1_tradeoff')


# ---------------------------------------------------------------------------
# 图 S8.2 —— 三组分布（rel / avoid / RR 小提琴）
# ---------------------------------------------------------------------------
def figS8_2_dist(on_df, vp_df, ba_df):
    dfs = {'Onshore': on_df, 'VPTS': vp_df, 'Bauer': ba_df}
    metrics = [('rel', 'Relative range'), ('avoid', 'Avoidable fraction'), ('rr', 'Risk reduction (%)')]

    fig = plt.figure(figsize=(10.0, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.07, right=0.96, top=0.85, bottom=0.14)
    for j, (col, ylab) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        data = [dfs[g][col].dropna().values for g in GROUP_ORDER]
        vp = ax.violinplot(data, positions=np.arange(3), showmedians=True, widths=0.65)
        for k, body in enumerate(vp['bodies']):
            body.set_facecolor(GROUP_COLORS[GROUP_ORDER[k]])
            body.set_alpha(0.5); body.set_edgecolor(GROUP_COLORS[GROUP_ORDER[k]])
        for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp[part].set_color(C_ECON); vp[part].set_linewidth(0.9)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(GROUP_ORDER, fontsize=7)
        ax.set_ylabel(ylab)
        panel_label(ax, chr(ord('a') + j)); style_ax(ax)

    save_supp(fig, 'figS8_2_dist')


# ---------------------------------------------------------------------------
# 图 S8.4 —— 风场规模：分布 + rel 随规模（补充敏感性支撑）
# ---------------------------------------------------------------------------
def figS9_size_rel(on_df, ctx):
    on1 = ctx['on1']
    m = on_df[['farm_id', 'rel']].merge(on1[['farm_id', 'n_turbines']], on='farm_id', how='inner')
    n = m.n_turbines.clip(upper=60)

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.86, bottom=0.14)

    axa = fig.add_subplot(gs[0, 0])
    axa.hist(n, bins=60, color=C_LAND, alpha=0.7, edgecolor='none')
    axa.axvline(n.median(), color=C_ECO, lw=1.4, label=f'median = {n.median():.0f}')
    axa.set_xlabel('Farm size (turbines, clipped at 60)')
    axa.set_ylabel('Number of farms')
    axa.set_title('Onshore farm size distribution', fontsize=9)
    axa.legend(loc='upper right', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    axb.scatter(n, m.rel * 100, s=3, c=C_LAND, alpha=0.2, edgecolors='none')
    bins = [0, 5, 10, 20, 50, 10 ** 9]
    centers, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = m[(m.n_turbines >= lo) & (m.n_turbines < hi)]
        if len(sel) > 0:
            centers.append((lo + min(hi, 60)) / 2)
            meds.append(sel.rel.median() * 100)
    axb.plot(centers, meds, 'o-', color=C_ECO, lw=2, ms=6, label='Median rel per size bin')
    axb.set_xlabel('Farm size (turbines)')
    axb.set_ylabel('Relative range rel (%)')
    axb.set_title('Direction sensitivity vs farm size', fontsize=9)
    axb.set_ylim(90, 100)
    axb.legend(loc='lower right', fontsize=7, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS9_size_rel')


# ---------------------------------------------------------------------------
# 图 S8.5 —— AEP 平坦 vs 暴露敏感（核心论证的对比）
# ---------------------------------------------------------------------------
def figS10_aep_vs_exposure(ctx):
    on1 = ctx['on1']; on_aep = ctx['on_aep']
    idxs = np.linspace(0, len(on1) - 1, 15).astype(int)
    sample = on1.iloc[idxs]

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.86, bottom=0.14)

    axa = fig.add_subplot(gs[0, 0])
    for _, r in sample.iterrows():
        sub = on_aep[on_aep.farm_id == r.farm_id]
        if len(sub) == 0:
            continue
        aep = np.array([sub.iloc[0][f'aep_{a:03d}'] for a in ORIENT], dtype=float)
        axa.plot(ORIENT, aep / aep.max(), lw=0.8, alpha=0.5, color=C_LAND)
    axa.set_xlabel('Array orientation θ (deg)')
    axa.set_ylabel('AEP (normalized)')
    axa.set_title('AEP(θ) — near-flat (range ≈2.7%)', fontsize=9)
    axa.set_xlim(0, 170); axa.set_ylim(0.90, 1.01)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    for _, r in sample.iterrows():
        E = on_Evec(r.spring_dir, r.autumn_dir, r.spring_conc, r.autumn_conc)
        axb.plot(THETAS, E, lw=0.8, alpha=0.5, color=C_VPTS)
    axb.set_xlabel('Array orientation θ (deg)')
    axb.set_ylabel('Exposure E(θ) (normalized)')
    axb.set_title('E(θ) — highly variable (rel ≈100%)', fontsize=9)
    axb.set_xlim(0, 180); axb.set_ylim(0, 1.02)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS10_aep_vs_exposure')


# ---------------------------------------------------------------------------
# 图 S8.6 —— 春/秋方向反向平行 + 错位角分布
# ---------------------------------------------------------------------------
def figS11_direction_misalignment(on_df, ctx):
    on1 = ctx['on1']
    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.86, bottom=0.14)

    axa = fig.add_subplot(gs[0, 0])
    axa.scatter(on1.spring_dir, on1.autumn_dir, s=2, c=C_LAND, alpha=0.15, edgecolors='none')
    xs = np.linspace(0, 360, 400)
    axa.plot(xs, (xs + 180) % 360, color='#999999', ls='--', lw=1.4, label='anti-parallel (Δ=180°)')
    axa.set_xlabel('Spring direction (deg)')
    axa.set_ylabel('Autumn direction (deg)')
    axa.set_title('Spring vs autumn migration direction', fontsize=9)
    axa.set_xlim(0, 360); axa.set_ylim(0, 360)
    axa.legend(loc='lower right', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    d = on_df.d_full.dropna()
    axb.hist(d, bins=40, color=C_VPTS, alpha=0.7, edgecolor='none')
    axb.axvline(d.median(), color=C_ECO, lw=1.6, label=f'median = {d.median():.0f}°')
    axb.set_xlabel('Misalignment d_full (deg)')
    axb.set_ylabel('Number of farms')
    axb.set_title('Misalignment between θ_econ and θ_eco', fontsize=9)
    axb.legend(loc='upper right', fontsize=7, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS11_direction_misalignment')


# ---------------------------------------------------------------------------
# 图 S8.7 —— 代表性风场的逐场权衡曲线（饱和）
# ---------------------------------------------------------------------------
def figS12_tradeoff_saturation(ctx):
    on = ctx['on']
    on1 = on[on.budget == 0.01].sort_values('risk_reduction')
    idxs = np.linspace(0, len(on1) - 1, 12).astype(int)
    fids = on1.iloc[idxs].farm_id.values
    cmap = plt.get_cmap('cividis')

    fig = plt.figure(figsize=(5.8, 4.2))
    ax = fig.add_subplot(111)
    for k, fid in enumerate(fids):
        sub = on[on.farm_id == fid].sort_values('budget')
        ax.plot(sub.budget * 100, sub.risk_reduction, 'o-', lw=1.0, ms=3,
                color=cmap(k / (len(fids) - 1)), alpha=0.85)
    ax.set_xlabel('AEP budget (%)')
    ax.set_ylabel('Risk reduction (%)')
    ax.set_title('Per-farm trade-off curves (12 representative farms)', fontsize=9)
    ax.set_ylim(0, 105)
    style_ax(ax)
    save_supp(fig, 'figS12_tradeoff_saturation')


# ---------------------------------------------------------------------------
# 图 S8.8 —— 海上 E(θ) 曲线样本（VPTS vs Bauer）
# ---------------------------------------------------------------------------
def figS13_offshore_exposure(ctx):
    cur = ctx['cur']; vpts_ids = ctx['vpts_ids']; bauer_ids = ctx['bauer_ids']
    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.96, top=0.86, bottom=0.14)

    for i, (ids, c, name) in enumerate([
        (sorted(vpts_ids), C_VPTS, 'VPTS (radar)'),
        (sorted(bauer_ids), C_BAUER, 'Bauer (grid)')]):
        ax = fig.add_subplot(gs[0, i])
        idxs = np.linspace(0, len(ids) - 1, min(len(ids), 15)).astype(int)
        for j in idxs:
            E = off_Evec(ids[j], cur)
            ax.plot(THETAS, E, lw=0.8, alpha=0.5, color=c)
        ax.set_xlabel('Array orientation θ (deg)')
        ax.set_ylabel('Exposure E(θ) (normalized)')
        ax.set_title(f'{name} — E(θ) curves (n={len(ids)})', fontsize=9)
        ax.set_xlim(0, 180); ax.set_ylim(0, 1.05)
        panel_label(ax, 'a' if i == 0 else 'b'); style_ax(ax)

    save_supp(fig, 'figS13_offshore_exposure')


# ---------------------------------------------------------------------------
# 图 S8.9 —— Bauer 格网春季迁徙方向场
# ---------------------------------------------------------------------------
def figS14_bauer_grid(ctx):
    grid = ctx['grid']
    fig = plt.figure(figsize=(9.0, 5.2))
    ax = fig.add_subplot(111)
    u = np.sin(np.radians(grid.spring_dir))
    v = np.cos(np.radians(grid.spring_dir))
    q = ax.quiver(grid.lon, grid.lat, u, v, grid.spring_conc,
                  cmap='cividis', scale=40, width=0.003, alpha=0.85)
    cb = fig.colorbar(q, ax=ax, pad=0.02)
    cb.set_label('Spring concentration', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Bauer grid — spring migration direction (arrow) & concentration (color)', fontsize=9)
    style_ax(ax)
    save_supp(fig, 'figS14_bauer_grid')


# ---------------------------------------------------------------------------
# 图 S7.3b —— ERA5 vs 候鸟方向逐场散点（补充 S7.3，逐场口径，非仅汇总条形）
# ---------------------------------------------------------------------------
def figS15_era5_scatter():
    e = pd.read_csv(os.path.join(PROC, 'sensitivity_era5_vs_bird.csv'), encoding='utf-8-sig')

    fig = plt.figure(figsize=(9.0, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.09, right=0.96, top=0.86, bottom=0.15)

    axa = fig.add_subplot(gs[0, 0])
    for src, c in [('VPTS', C_VPTS), ('Bauer', C_BAUER)]:
        s = e[e.source == src]
        axa.scatter(s.era5_wind_dir, s.bird_dir, s=22, c=c, alpha=0.75,
                    edgecolors='white', linewidths=0.4, label=f'{src} (n={len(s)})')
    xs = np.linspace(0, 360, 400)
    axa.plot(xs, xs, color='#999999', ls='--', lw=1.2, label='identity (Δ=0°)')
    axa.plot(xs, (xs + 180) % 360, color=C_ECO, ls=':', lw=1.2, label='anti-parallel (Δ=180°)')
    axa.set_xlabel('ERA5 wind direction (deg)')
    axa.set_ylabel('Measured bird direction (deg)')
    axa.set_title('ERA5 vs measured direction (per farm × season)', fontsize=9)
    axa.set_xlim(0, 360); axa.set_ylim(0, 360); axa.set_aspect('equal')
    axa.legend(loc='upper left', fontsize=6.5, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    for src, c in [('VPTS', C_VPTS), ('Bauer', C_BAUER)]:
        s = e[e.source == src]
        axb.hist(s.circ360, bins=18, range=(0, 180), color=c, alpha=0.6,
                 edgecolor='none', label=src)
    axb.axvline(30, color='#999999', ls='--', lw=1.0)
    axb.text(31, axb.get_ylim()[1] * 0.95, '30°', fontsize=7, color='#666666', ha='left')
    axb.set_xlabel('Circular angular difference (deg)')
    axb.set_ylabel('Number of farm–season pairs')
    axb.set_title('Angular difference distribution', fontsize=9)
    axb.legend(loc='upper right', fontsize=6.5, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS15_era5_scatter')


# ---------------------------------------------------------------------------
# 图 S7.2b —— 衰减常数 α 的 AEP(θ) 曲线对比（补充 S7.2，曲线口径，非仅条形）
# ---------------------------------------------------------------------------
def figS16_alpha_curves(on_df, ctx):
    row, meta, E = rep_onshore(on_df, ctx)
    fid = row.farm_id
    a75 = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
    a50 = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves_alpha050.csv'), encoding='utf-8-sig')

    def curve(df):
        sub = df[df.farm_id == fid]
        if len(sub) == 0:
            return None
        return np.array([sub.iloc[0][f'aep_{a:03d}'] for a in ORIENT], dtype=float)

    c75, c50 = curve(a75), curve(a50)

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.08, right=0.96, top=0.86, bottom=0.15)

    axa = fig.add_subplot(gs[0, 0])
    if c75 is not None:
        axa.plot(ORIENT, c75 / c75.max(), color=C_LAND, lw=1.8, label='α = 0.075 (baseline)')
    if c50 is not None:
        axa.plot(ORIENT, c50 / c50.max(), color=C_ECO, lw=1.8, ls='--', label='α = 0.05')
    axa.set_xlabel('Array orientation θ (deg)')
    axa.set_ylabel('AEP (normalized)')
    axa.set_title(f'Representative farm (id={int(fid)}) AEP(θ)', fontsize=9)
    axa.set_xlim(0, 170)
    axa.legend(loc='lower left', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    axb.plot(THETAS, E, color=C_VPTS, lw=1.8, label='E(θ)')
    axb.set_xlabel('Array orientation θ (deg)')
    axb.set_ylabel('Exposure E(θ) (normalized)')
    axb.set_title('Same farm exposure E(θ)', fontsize=9)
    axb.set_xlim(0, 180); axb.set_ylim(0, 1.02)
    axb.legend(loc='upper right', fontsize=7, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS16_alpha_curves')


# ---------------------------------------------------------------------------
# 图 S4b —— 全部 171 个海上风场暴露曲线 + 三组中位（补充 S4，全集而非样本）
# ---------------------------------------------------------------------------
def figS17_exposure_gallery_171(ctx):
    cur = ctx['cur']
    vp = set(ctx['vpts_ids']); ba = set(ctx['bauer_ids'])
    th = np.arange(0, 180, 10)
    other = sorted(f for f in cur.farm_id.unique() if f not in vp and f not in ba)

    def _dedup(sub):
        # 每场 19 行：theta=0 处有重复行，去重后回到 18 角度统一网格
        return sub.sort_values('theta_deg').drop_duplicates('theta_deg', keep='last')

    def med_curve(ids):
        arrs = [_dedup(cur[cur.farm_id == f]).risk_score.values for f in sorted(ids)]
        return np.nanmedian(np.vstack(arrs), axis=0)

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.08, right=0.96, top=0.86, bottom=0.14)

    axa = fig.add_subplot(gs[0, 0])
    for fid, sub in cur.groupby('farm_id'):
        s = _dedup(sub)
        axa.plot(s.theta_deg, s.risk_score, lw=0.5, alpha=0.22, color='#999999')
    axa.plot(th, med_curve(sorted(vp)), lw=2.2, color=C_VPTS, label=f'VPTS median (n={len(vp)})')
    axa.plot(th, med_curve(sorted(ba)), lw=2.2, color=C_BAUER, label=f'Bauer median (n={len(ba)})')
    axa.plot(th, med_curve(other), lw=2.2, color='#555555', label=f'Other median (n={len(other)})')
    axa.set_xlabel('Array orientation θ (deg)')
    axa.set_ylabel('Risk score')
    axa.set_title(f'All {cur.farm_id.nunique()} offshore exposure curves', fontsize=9)
    axa.set_xlim(0, 180)
    axa.legend(loc='upper right', fontsize=6.5, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    axb = fig.add_subplot(gs[0, 1])
    for ids, color, label in [(sorted(vp), C_VPTS, 'VPTS'), (sorted(ba), C_BAUER, 'Bauer'),
                              (other, '#555555', 'Other')]:
        m = med_curve(ids)
        axb.plot(th, m / (m.max() + 1e-9), lw=2.0, color=color, label=label)
    axb.set_xlabel('Array orientation θ (deg)')
    axb.set_ylabel('Median risk score (normalized)')
    axb.set_title('Group-median curves (normalized)', fontsize=9)
    axb.set_xlim(0, 180); axb.set_ylim(0, 1.05)
    axb.legend(loc='upper right', fontsize=6.5, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS17_exposure_gallery_171')


# ---------------------------------------------------------------------------
# 图 S6b —— 海上逐场权衡曲线（VPTS vs Bauer，补充 S6，海上口径）
# ---------------------------------------------------------------------------
def figS18_offshore_tradeoff(ctx):
    to = ctx['to']
    vp = set(ctx['vpts_ids']); ba = set(ctx['bauer_ids'])

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.08, right=0.96, top=0.86, bottom=0.14)

    for i, (ids, c, name) in enumerate([(sorted(vp), C_VPTS, 'VPTS'),
                                        (sorted(ba), C_BAUER, 'Bauer')]):
        ax = fig.add_subplot(gs[0, i])
        for fid in ids:
            s = to[to.farm_id == fid].sort_values('budget')
            ax.plot(s.budget * 100, s.risk_reduction_pct, 'o-', lw=0.8, ms=2.5,
                    color=c, alpha=0.55)
        ax.set_xlabel('AEP budget (%)')
        ax.set_ylabel('Risk reduction (%)')
        ax.set_title(f'{name} offshore — per-farm trade-off (n={len(ids)})', fontsize=9)
        ax.set_ylim(0, 105)
        panel_label(ax, 'a' if i == 0 else 'b'); style_ax(ax)

    save_supp(fig, 'figS18_offshore_tradeoff')


# ---------------------------------------------------------------------------
# 图 S7.15 —— 春秋单季分解（spring-only / autumn-only / 集中度加权三口径 RR）
# ---------------------------------------------------------------------------
def figS7_15_seasonal():
    SUM = load_summary()
    s = SUM['S715_seasonal_decompose']
    d = pd.read_csv(os.path.join(PROC, 'onshore_seasonal_decompose.csv'), encoding='utf-8-sig')

    fig = plt.figure(figsize=(9.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.09, right=0.96, top=0.84, bottom=0.15)

    # (a) 三口径 RR 的累积分布（ECDF）
    axa = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, 100, 500)
    for col, c, lab in [('rr_full', C_ECON, 'Full (conc.-weighted)'),
                        ('rr_spring', C_SPRING, 'Spring only'),
                        ('rr_autumn', C_AUTUMN, 'Autumn only')]:
        v = d[col].values
        y = np.array([(v <= x).mean() for x in xs])
        axa.plot(xs, y, color=c, lw=1.8, label=lab)
    axa.set_xlabel('Risk reduction RR (%)')
    axa.set_ylabel('Cumulative fraction')
    axa.set_title('Single-season vs full — RR distribution', fontsize=9)
    axa.set_xlim(0, 100); axa.set_ylim(0, 1.02)
    axa.legend(loc='lower right', fontsize=6.5, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 逐场 spring−autumn RR 差
    axb = fig.add_subplot(gs[0, 1])
    diff = (d['rr_spring'] - d['rr_autumn']).values
    axb.hist(diff, bins=60, color=C_LAND, alpha=0.7, edgecolor='none')
    axb.axvline(0, color=C_ECO, lw=1.4)
    axb.axvline(s['diff_spring_autumn_mean_pp'], color='#999999', ls='--', lw=1.2)
    axb.text(0.03, 0.94,
             f"median = {s['diff_spring_autumn_med_pp']:.2f} pp\n"
             f"mean = {s['diff_spring_autumn_mean_pp']:.2f} pp\n"
             f"axis sep median = {s['axis_sep_med_deg']:.1f}°\n"
             f"n = {s['n']}",
             transform=axb.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC'))
    axb.set_xlabel('RR$_{spring}$ − RR$_{autumn}$ (percentage points)')
    axb.set_ylabel('Number of farms')
    axb.set_title('Spring vs autumn (single season)', fontsize=9)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS7_15_seasonal')


# =====================================================================
# 图 S7.8 / S7.9 —— 飞行高度 & 迁移时序（原始 VPTS 雷达，4 站）
# =====================================================================
VPTS_DIR = os.path.join(BASE, '..', 'data', 'raw', 'radar_vpts')
HEIGHT_STATIONS = {'nlhrw', 'bejab', 'deess', 'frabb'}
ST_ORDER = sorted(HEIGHT_STATIONS)
ST_COLORS = {'nlhrw': C_LAND, 'bejab': C_VPTS, 'deess': C_BAUER, 'frabb': '#E69F00'}
OFF_AEP_CSV = r'D:\1风力发电实习\offshore-task3\output\task3_s1_optimal_orientation.csv'
AEP_COLS = [f'aep_{a:03d}' for a in ORIENT]


def _vpts_profiles():
    """读原始 VPTS（4 站）：高度分层占比 + 月相 + 逐站转子层(0-200m)占比。

    过滤口径与旧稿 compute_height_profile / recompute_radar_signatures_corrected 一致：
    夜迁 20:00-06:00、dens>10、春 3-5 月 / 秋 8-11 月；dens 取 p[12]（birds/km^3）。
    """
    from collections import defaultdict

    def _f(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float('nan')

    bins = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200)]
    height = {'spring': defaultdict(float), 'autumn': defaultdict(float)}
    month = {'spring': defaultdict(float), 'autumn': defaultdict(float)}
    rotor = {'spring': defaultdict(float), 'autumn': defaultdict(float)}
    total = {'spring': defaultdict(float), 'autumn': defaultdict(float)}

    for fname in sorted(os.listdir(VPTS_DIR)):
        if not fname.endswith('.txt'):
            continue
        st = fname.split('_')[0]
        if st not in HEIGHT_STATIONS:
            continue
        with open(os.path.join(VPTS_DIR, fname), encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                p = line.split()
                if len(p) < 13:
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
                total[season][st] += dens
                if hght == 0:
                    rotor[season][st] += dens

    hfrac = {}
    for season in ('spring', 'autumn'):
        fracs = []
        for lo, hi in bins:
            fracs.append(sum(height[season][h] for h in height[season] if lo <= h < hi))
        fracs.append(sum(height[season][h] for h in height[season] if h >= 1200))
        tot = sum(fracs)
        hfrac[season] = [f / tot * 100 if tot > 0 else 0.0 for f in fracs]

    rotor_frac = {}
    for season in ('spring', 'autumn'):
        rotor_frac[season] = {
            st: (rotor[season][st] / total[season][st] * 100 if total[season][st] > 0 else np.nan)
            for st in HEIGHT_STATIONS}
    return hfrac, dict(month), rotor_frac, total


def figS7_8_height():
    hfrac, month, rotor_frac, _total = _vpts_profiles()
    bin_lbl = ['0–200', '200–400', '400–600', '600–800', '800–1000', '1000–1200', '>1200']
    x = np.arange(len(bin_lbl))
    w = 0.36

    fig = plt.figure(figsize=(9.5, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.97, top=0.84, bottom=0.16)

    # (a) 高度廓线（占全列通量 %）
    axa = fig.add_subplot(gs[0, 0])
    axa.bar(x - w / 2, hfrac['spring'], w, color=C_SPRING, alpha=0.85, edgecolor='none', label='Spring')
    axa.bar(x + w / 2, hfrac['autumn'], w, color=C_AUTUMN, alpha=0.85, edgecolor='none', label='Autumn')
    axa.axvspan(-0.5, 0.5, color=C_ECO, alpha=0.10)
    axa.text(0.02, 0.96, 'rotor layer\n(0–200 m)', transform=axa.transAxes, fontsize=7,
             ha='left', va='top', color=C_ECO)
    axa.set_xticks(x); axa.set_xticklabels(bin_lbl, fontsize=7)
    axa.set_xlabel('Altitude band (m)')
    axa.set_ylabel('Fraction of total flux (%)')
    axa.set_title('Vertical profile of nocturnal migrant flux', fontsize=9)
    axa.legend(loc='upper right', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 逐站转子层占比
    axb = fig.add_subplot(gs[0, 1])
    sp = [rotor_frac['spring'][s] for s in ST_ORDER]
    au = [rotor_frac['autumn'][s] for s in ST_ORDER]
    xx = np.arange(len(ST_ORDER))
    axb.bar(xx - w / 2, sp, w, color=C_SPRING, alpha=0.85, edgecolor='none', label='Spring')
    axb.bar(xx + w / 2, au, w, color=C_AUTUMN, alpha=0.85, edgecolor='none', label='Autumn')
    allv = np.array([v for v in sp + au if np.isfinite(v)])
    med = float(np.median(allv))
    axb.axhline(med, color=C_ECO, ls='--', lw=1.2)
    axb.text(0.98, 0.04,
             f'median = {med:.1f}%   range = {allv.min():.0f}–{allv.max():.0f}%',
             transform=axb.transAxes, ha='right', va='bottom', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC'))
    axb.set_xticks(xx); axb.set_xticklabels(ST_ORDER, fontsize=8)
    axb.set_xlabel('Radar station')
    axb.set_ylabel('Rotor-layer fraction (%)')
    axb.set_title('Rotor-layer (0–200 m) fraction by station', fontsize=9)
    axb.legend(loc='upper right', fontsize=7, frameon=False)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS7_8_height')


def figS7_9_timing():
    hfrac, month, rotor_frac, total = _vpts_profiles()
    order = [3, 4, 5, 8, 9, 10, 11]
    spring_months = {3, 4, 5}
    vals = [month['spring'].get(m, 0.0) if m in spring_months else month['autumn'].get(m, 0.0)
            for m in order]
    cols = [C_SPRING if m in spring_months else C_AUTUMN for m in order]
    # 逐站总通量（春+秋合并）
    st_flux = [total['spring'][st] + total['autumn'][st] for st in ST_ORDER]

    fig = plt.figure(figsize=(9.5, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.08, right=0.97, top=0.84, bottom=0.16)

    # (a) 逐月总通量（双峰：春 3–5 / 秋 8–11）
    axa = fig.add_subplot(gs[0, 0])
    axa.bar(order, vals, color=cols, alpha=0.85, edgecolor='none')
    axa.set_xticks(order); axa.set_xticklabels([f'{m}' for m in order])
    axa.set_xlabel('Month')
    axa.set_ylabel('Nocturnal migrant flux (Σ dens)')
    axa.set_title('Migration timing (spring + autumn)', fontsize=9)
    handles = [Line2D([], [], color=C_SPRING, lw=6, label='Spring (Mar–May)'),
               Line2D([], [], color=C_AUTUMN, lw=6, label='Autumn (Aug–Nov)')]
    axa.legend(handles=handles, loc='upper right', fontsize=7, frameon=False)
    panel_label(axa, 'a'); style_ax(axa)

    # (b) 逐站总通量（春+秋合并）
    axb = fig.add_subplot(gs[0, 1])
    xb = np.arange(len(ST_ORDER))
    axb.bar(xb, st_flux, 0.6, color=[ST_COLORS[s] for s in ST_ORDER], edgecolor='none')
    axb.set_xticks(xb); axb.set_xticklabels(ST_ORDER, fontsize=8)
    axb.set_xlabel('Radar station')
    axb.set_ylabel('Total flux (Σ dens, spring + autumn)')
    axb.set_title('Total migrant flux by station', fontsize=9)
    panel_label(axb, 'b'); style_ax(axb)

    save_supp(fig, 'figS7_9_timing')


# =====================================================================
# 图 S7.10 —— 尾流极坐标：陆上 Jensen vs 海上 FLORIS 归一化 AEP(θ)
# =====================================================================
def figS7_10_wakepolar(ctx):
    on_aep = ctx['on_aep']
    A = on_aep[AEP_COLS].values.astype(float)
    mx = A.max(axis=1); ok = mx > 0
    A2 = A[ok]; mx2 = mx[ok]
    on_sens = (mx2 - A2.min(axis=1)) / mx2 * 100
    on_med = float(np.median(on_sens))
    j = int(np.argmin(np.abs(on_sens - on_med)))
    on_a = A2[j]
    on_curve = on_a / on_a.max()
    on_range = (on_a.max() - on_a.min()) / on_a.max() * 100

    off = pd.read_csv(OFF_AEP_CSV, encoding='utf-8-sig')
    off.columns = [c.strip().lstrip('﻿') for c in off.columns]
    curves, sens = [], []
    for fid, g in off.groupby('farm_id'):
        g = g.drop_duplicates('angle_deg').sort_values('angle_deg')
        a = g['expected_AEP_kWh'].values.astype(float)
        if a.max() <= 0:
            continue
        curves.append(a); sens.append((a.max() - a.min()) / a.max() * 100)
    sens = np.array(sens)
    off_med = float(np.median(sens))
    jj = int(np.argmin(np.abs(sens - off_med)))
    off_a = curves[jj]
    off_curve = off_a / off_a.max()
    off_range = (off_a.max() - off_a.min()) / off_a.max() * 100

    th360 = np.concatenate([ORIENT, ORIENT + 180])
    on_full = np.concatenate([on_curve, on_curve])
    off_full = np.concatenate([off_curve, off_curve])
    rlo = max(0.0, min(on_curve.min(), off_curve.min()) - 0.02)

    fig = plt.figure(figsize=(9.5, 4.6))
    axa = fig.add_subplot(1, 2, 1, projection='polar')
    axa.plot(np.radians(th360), on_full, color=C_LAND, lw=1.7)
    axa.set_theta_zero_location('N'); axa.set_theta_direction(-1)
    axa.set_rlim(rlo, 1.0); axa.set_rlabel_position(135)
    axa.tick_params(labelsize=6.5)
    axa.set_title(f'Onshore Jensen — AEP range {on_range:.1f}%', fontsize=9, pad=18)
    panel_label(axa, 'a')

    axb = fig.add_subplot(1, 2, 2, projection='polar')
    axb.plot(np.radians(th360), off_full, color=C_VPTS, lw=1.7)
    axb.set_theta_zero_location('N'); axb.set_theta_direction(-1)
    axb.set_rlim(rlo, 1.0); axb.set_rlabel_position(135)
    axb.tick_params(labelsize=6.5)
    axb.set_title(f'Offshore FLORIS — AEP range {off_range:.1f}%', fontsize=9, pad=18)
    panel_label(axb, 'b')

    fig.text(0.5, 0.015, 'Normalized AEP (r = AEP / AEP$_{max}$); representative farm '
                         '(closest to median sensitivity)',
             ha='center', fontsize=7, color='#555555')
    save_supp(fig, 'figS7_10_wakepolar')


# =====================================================================
# 图 S7.16 —— 缺省值扫描（转子直径 / 轮毂高度 / 容量）
# =====================================================================
BASE_DEFAULT = {'rotor': 70.0, 'hub': 70.0, 'capacity': 2000.0}
PANEL_KEYS = [('rotor', 'Rotor diameter default (m)'),
              ('hub', 'Hub height default (m)'),
              ('capacity', 'Capacity default (kW)')]


def figS7_16_defaults():
    SUM = load_summary()
    if 'S716_defaults_scan' not in SUM:
        print('  [skip] figS7_16_defaults (S716_defaults_scan not yet available)')
        return
    s = SUM['S716_defaults_scan']
    base = s['baseline_rr_med']

    fig = plt.figure(figsize=(11.0, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.07, right=0.97, top=0.78, bottom=0.17)
    for i, (key, xlab) in enumerate(PANEL_KEYS):
        d = s[key]
        xs = np.array(d['defaults'], dtype=float)
        ys = np.array(d['rr_med_1pct'], dtype=float)
        base_idx = list(d['defaults']).index(BASE_DEFAULT[key])
        ax = fig.add_subplot(gs[0, i])
        ax.plot(xs, ys, 'o-', color=C_LAND, lw=1.8, ms=6, zorder=3)
        ax.plot(xs[base_idx], ys[base_idx], 'o', ms=12, mfc='none', mec=C_ECO,
                mew=2.0, zorder=4, label='Baseline')
        ax.text(0.03, 0.03,
                f"max |Δ| = {d['max_abs_delta_pp']:.2f} pp\n"
                f"n$_{{imputed}}$ = {d['n_imputed']}",
                transform=ax.transAxes, fontsize=7.5, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC'))
        ax.set_xlabel(xlab)
        ax.set_ylabel('Median RR (%)')
        ax.set_title(key.capitalize(), fontsize=9)
        lo = min(ys.min(), base) - 1.0
        hi = max(ys.max(), base) + 1.0
        ax.set_ylim(lo, hi)
        ax.legend(loc='upper right', fontsize=6.5, frameon=False)
        panel_label(ax, ['a', 'b', 'c'][i]); style_ax(ax)

    save_supp(fig, 'figS7_16_defaults')


# =====================================================================
# 图 S7.17 —— 区域分层（RR 中位 by region，1% 预算）
# =====================================================================
def figS7_17_region():
    SUM = load_summary()
    regions = SUM['S717_region']['regions']
    names = [r['region'] for r in regions]
    rr = [r['rr_med'] for r in regions]
    n = [r['n'] for r in regions]

    fig = plt.figure(figsize=(7.5, 4.2))
    ax = fig.add_subplot(111)
    y = np.arange(len(names))[::-1]
    colors = [C_LAND if r['region'] != 'Other' else '#999999' for r in regions]
    ax.barh(y, rr, 0.6, color=colors, edgecolor='none')
    for yi, v, ni in zip(y, rr, n):
        ax.text(v + 1, yi, f'{v:.1f}%  (n={ni})', va='center', fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Median risk reduction RR (%)')
    ax.set_title('RR by region (1% AEP budget)', fontsize=9)
    ax.set_xlim(0, 108)
    panel_label(ax, 'a'); style_ax(ax)

    save_supp(fig, 'figS7_17_region')


# =====================================================================
# 图 S7.18 —— 更细 AEP 预算扫描（RR 中位 vs 预算，log-x）
# =====================================================================
def figS7_18_budget():
    SUM = load_summary()
    rows = SUM['S718_budget_extended']['budgets']
    x = np.array([r['budget'] for r in rows]) * 100
    y = np.array([r['rr_med'] for r in rows])

    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'o-', color=C_LAND, lw=1.8, ms=6)
    ax.set_xscale('log')
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1.8, f'{yi:.1f}%', ha='center', fontsize=6.5)
    ax.set_xlabel('AEP budget (%)')
    ax.set_ylabel('Median risk reduction RR (%)')
    ax.set_title('RR vs AEP budget (extended)', fontsize=9)
    ax.set_ylim(0, 106)
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    panel_label(ax, 'a'); style_ax(ax)

    save_supp(fig, 'figS7_18_budget')


# =====================================================================
# 图 S7.19 —— 浓度权重连续扫描（RR 中位 vs λ）
# =====================================================================
def figS7_19_concweight():
    SUM = load_summary()
    s = SUM['S719_concweight_sweep']
    lam = np.array(s['lambda'])
    rr = np.array(s['rr_med'])

    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(lam, rr, 'o-', color=C_LAND, lw=1.8, ms=6)
    ax.axvline(0.5, color='#999999', ls='--', lw=1.0)
    ax.annotate('autumn only\n(λ=0)', xy=(0.02, rr[0]), xytext=(0.08, rr[0] - 6),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))
    ax.annotate('spring only\n(λ=1)', xy=(0.98, rr[-1]), xytext=(0.72, rr[-1] - 6),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))
    ax.set_xlabel('Spring weight λ  (autumn weight = 1 − λ)')
    ax.set_ylabel('Median risk reduction RR (%)')
    ax.set_title('RR vs seasonal concentration weight', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    panel_label(ax, 'a'); style_ax(ax)

    save_supp(fig, 'figS7_19_concweight')


# ---------------------------------------------------------------------------
def main():
    print('Computing metrics ...')
    on_df, vp_df, ba_df, ctx = fs.compute_metrics()
    print(f'  Onshore n={len(on_df)}, VPTS n={len(vp_df)}, Bauer n={len(ba_df)}')

    print('Reusing existing figures ...'); reuse_existing_figures()

    print('Fig S4 exposure model ...'); figS4_exposure_model(on_df, ctx)
    print('Fig S5 budget ...'); figS5_budget(ctx)
    print('Fig S7.1/S7.2 wake+alpha ...'); figS7_1_wake_alpha()
    print('Fig S7.3 ERA5 ...'); figS7_3_era5()
    print('Fig S7.4 offshore convention ...'); figS7_4_offshore()
    print('Fig S7.5 seasonal weight ...'); figS7_5_seasonal()
    print('Fig S7.6 DBSCAN ...'); figS7_6_dbscan()
    print('Fig S7.7 radius ...'); figS7_7_radius()
    print('Fig S7.11 size ...'); figS7_11_size()
    print('Fig S7.12 PCA ...'); figS7_12_pca()
    print('Fig S7.13 AEP sensitivity ...'); figS7_13_aep()
    print('Fig S7.15 seasonal decompose ...'); figS7_15_seasonal()
    print('Fig S7.8 flight height ...'); figS7_8_height()
    print('Fig S7.9 migration timing ...'); figS7_9_timing()
    print('Fig S7.10 wake polar ...'); figS7_10_wakepolar(ctx)
    print('Fig S7.16 defaults scan ...'); figS7_16_defaults()
    print('Fig S7.17 region ...'); figS7_17_region()
    print('Fig S7.18 budget extended ...'); figS7_18_budget()
    print('Fig S7.19 conc weight sweep ...'); figS7_19_concweight()
    print('Fig S8.1 tradeoff ...'); figS8_1_tradeoff(on_df, vp_df, ba_df)
    print('Fig S8.2 distributions ...'); figS8_2_dist(on_df, vp_df, ba_df)
    print('Fig S8.4 size vs rel ...'); figS9_size_rel(on_df, ctx)
    print('Fig S8.5 AEP vs exposure ...'); figS10_aep_vs_exposure(ctx)
    print('Fig S8.6 direction misalignment ...'); figS11_direction_misalignment(on_df, ctx)
    print('Fig S8.7 tradeoff saturation ...'); figS12_tradeoff_saturation(ctx)
    print('Fig S8.8 offshore exposure ...'); figS13_offshore_exposure(ctx)
    print('Fig S8.9 bauer grid ...'); figS14_bauer_grid(ctx)
    print('Fig S7.3b ERA5 scatter ...'); figS15_era5_scatter()
    print('Fig S7.2b alpha curves ...'); figS16_alpha_curves(on_df, ctx)
    print('Fig S4b exposure gallery 171 ...'); figS17_exposure_gallery_171(ctx)
    print('Fig S6b offshore tradeoff ...'); figS18_offshore_tradeoff(ctx)

    print('\nDone. Outputs in figures_supp/:')
    for f in sorted(os.listdir(FIG_SUPP)):
        p = os.path.join(FIG_SUPP, f)
        print(f'  {f}  ({os.path.getsize(p)/1e3:.0f} KB)')


if __name__ == '__main__':
    main()
