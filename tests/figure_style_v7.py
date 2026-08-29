# -*- coding: utf-8 -*-
"""figure_style_v7.py —— v7 论文图的共享外观层（在 figure_style 之上）。

统一三件事，供 Fig 2–5 与附图复用：

1. **单一色系**。分类量一律 Okabe-Ito 组色（陆上蓝 / VPTS 绿 / Bauer 粉）；
   连续量不再混用 cividis / magma / viridis 三套色标 ——
     * 与"某一组"绑定的连续场（如陆上 RR 空间分布）用该组色的单色渐变，
       颜色因此自带组身份，和散点、小提琴、图例天然同族；
     * 跨组的物理量（迁徙方向集中度等）统一用 viridis，与 Fig 1 的色条一致；
     * 阈值/规则线一律用生态朱红 C_ECO，与 Fig 1 的 ≤20°、≤1% 标注一致。

2. **子论点分带**。每张图按 R*.1 / R*.2 / R*.3 分成横向条带，
   条带首行写明该带回答的子论点，子图与论断一一对应。

3. **少留白**。tight_limits() 按数据实际范围收紧坐标轴（保留一点边距），
   避免 v6 里"数据挤在 93–100、轴却画到 85–102"那种大片空白。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_style import (C_LAND, C_VPTS, C_BAUER, C_SPRING, C_AUTUMN,
                          C_ECON, C_ECO, GROUP_COLORS, GROUP_ORDER)

# --- 画布尺寸（Nature 双栏 7.09 in；单栏 3.46 in；最大高 9.72 in）-----------
W_SINGLE, W_ONEHALF, W_DOUBLE, H_MAX = 3.46, 4.72, 7.09, 9.72

GROUP_LABEL = {'Onshore': 'Onshore', 'VPTS': 'Offshore · VPTS', 'Bauer': 'Offshore · Bauer'}

# --- 排版尺度：全图只用四档字号，最小 6 pt（Nature 图内文字要求 5–7 pt）------
FS_PANEL = 8.0   # 面板字母
FS_TITLE = 7.0   # 子图标题 / 坐标轴标题 / 条带首行
FS_TICK = 6.0    # 刻度
FS_ANNOT = 6.0   # 图例 / 面板内注记 / 柱顶数值

CMAP_PHYS = 'viridis'          # 跨组物理量（方向集中度等），与 Fig 1 色条同族
C_GRID = '#DDDDDD'
C_TXT = '#333333'
C_MUTED = '#777777'

RC = {
    'font.size': FS_TICK,
    'axes.titlesize': FS_TITLE,
    'axes.labelsize': FS_TITLE,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_ANNOT,
    'axes.labelpad': 2.0,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
    'legend.frameon': False,
    'axes.unicode_minus': False,
    'savefig.dpi': 300,
}


def use_v7_style():
    plt.rcParams.update(RC)


# ---------------------------------------------------------------------------
# 单色渐变：把组色扩成连续色标，让连续场自带组身份
# ---------------------------------------------------------------------------
def _shift(hex_color, k):
    """k>0 变亮（向白插值），k<0 变暗（向黑插值）。"""
    r, g, b = mcolors.to_rgb(hex_color)
    if k >= 0:
        return (r + (1 - r) * k, g + (1 - g) * k, b + (1 - b) * k)
    return (r * (1 + k), g * (1 + k), b * (1 + k))


def seq_cmap(base, name=None, n=256):
    """由一个组色生成 白→组色→深色 的顺序色标（色盲安全，明度单调）。"""
    return LinearSegmentedColormap.from_list(
        name or f'seq_{base}',
        [_shift(base, 0.94), _shift(base, 0.72), _shift(base, 0.34),
         base, _shift(base, -0.34)], N=n)


CMAP_ONSHORE = seq_cmap(C_LAND, 'seq_onshore')
CMAP_VPTS = seq_cmap(C_VPTS, 'seq_vpts')
CMAP_BAUER = seq_cmap(C_BAUER, 'seq_bauer')
GROUP_CMAP = {'Onshore': CMAP_ONSHORE, 'VPTS': CMAP_VPTS, 'Bauer': CMAP_BAUER}


# ---------------------------------------------------------------------------
# 版面工具
# ---------------------------------------------------------------------------
def band_header(fig, y, tag, text, x=0.012, color=C_ECO, size=FS_TITLE):
    """条带分组标注 —— **不向图上绘制任何内容**。

    tag（'R1.1' 等）与 text（该带回答的论断）都只作代码内注释，标明这一带对应
    哪条子论点。图内既不出现内部编号，也不出现整句论断：面板与论断的对应关系
    由图注承担，图面只留数据。保留本函数与调用点，是为了让读者在读代码时仍能
    一眼看出分带依据。
    """
    return None


def plabel(ax, txt, x=-0.16, y=1.10, fs=FS_PANEL):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs, fontweight='bold',
            va='top', ha='left', color='black')


def flabel(fig, txt, x, y, fs=FS_PANEL):
    fig.text(x, y, txt, fontsize=fs, fontweight='bold', va='top', ha='left')


def sty(ax, grid_axis=None):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6, length=2.2, direction='out')
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def tight_limits(ax, values, axis='y', pad=0.06, lo=None, hi=None):
    """按数据实际范围收紧坐标轴，两端各留 pad 比例的余量。"""
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], float)
    if v.size == 0:
        return
    a, b = float(v.min()), float(v.max())
    if b - a < 1e-9:
        a, b = a - 1, b + 1
    m = (b - a) * pad
    a, b = a - m, b + m
    if lo is not None:
        a = max(a, lo)
    if hi is not None:
        b = min(b, hi)
    (ax.set_ylim if axis == 'y' else ax.set_xlim)(a, b)


def group_legend(ax, loc='best', groups=GROUP_ORDER, marker='o', ncol=1, **kw):
    h = [Line2D([], [], marker=marker, ls='', color=GROUP_COLORS[g],
                markersize=4, label=GROUP_LABEL[g]) for g in groups]
    return ax.legend(handles=h, loc=loc, ncol=ncol, handletextpad=0.3,
                     borderpad=0.25, labelspacing=0.25, **kw)


# ---------------------------------------------------------------------------
# 复用图元
# ---------------------------------------------------------------------------
def raincloud(ax, data_by_group, groups=GROUP_ORDER, width=0.32, jitter=0.055,
              point_size=1.4, max_points=400, seed=0, annotate=None,
              annot_fmt='{:.1f}', annot_dy=0.012):
    """半小提琴 + 箱 + 抖点：一次给出分布形状、四分位与个体，替代信息量低的柱状图。"""
    rng = np.random.default_rng(seed)
    for i, g in enumerate(groups):
        v = np.asarray([x for x in np.ravel(data_by_group[g]) if np.isfinite(x)], float)
        if v.size == 0:
            continue
        c = GROUP_COLORS[g]
        parts = ax.violinplot([v], positions=[i], widths=width * 2,
                              showextrema=False, showmedians=False)
        for b in parts['bodies']:
            # 只保留右半边，左半边留给抖点
            p = b.get_paths()[0]
            p.vertices[:, 0] = np.clip(p.vertices[:, 0], i, np.inf)
            b.set_facecolor(_shift(c, 0.55))
            b.set_edgecolor(c)
            b.set_linewidth(0.5)
            b.set_alpha(0.9)
        s = v if v.size <= max_points else rng.choice(v, max_points, replace=False)
        ax.scatter(i - jitter - rng.random(s.size) * jitter * 1.6, s,
                   s=point_size, color=c, alpha=0.35, edgecolors='none',
                   zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.012, i - 0.012], [q1, q3], color=_shift(c, -0.3), lw=2.4,
                solid_capstyle='butt', zorder=4)
        ax.plot([i - 0.075, i + 0.075], [med, med], color='white', lw=1.5, zorder=6)
        ax.plot([i - 0.075, i + 0.075], [med, med], color=_shift(c, -0.45), lw=0.9,
                zorder=7)
        if annotate is not None:
            ax.annotate(annot_fmt.format(annotate[g]), xy=(i + width * 0.75, med),
                        xytext=(2, 0), textcoords='offset points',
                        fontsize=FS_ANNOT, fontweight='bold', color=_shift(c, -0.3),
                        ha='left', va='center')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_LABEL[g].replace('Offshore · ', '') for g in groups])


def dumbbell(ax, y, x0, x1, color, lw=1.6, m0='X', m1='o', ms=5.5, zorder=4):
    """哑铃图：一条线连起同一对象的两个状态（AEP 最优 vs 生态最优等）。"""
    ax.plot([x0, x1], [y, y], color=_shift(color, 0.45), lw=lw,
            solid_capstyle='round', zorder=zorder)
    ax.scatter([x0], [y], marker=m0, s=ms ** 2, c=C_ECON, edgecolors='white',
               linewidths=0.6, zorder=zorder + 1)
    ax.scatter([x1], [y], marker=m1, s=ms ** 2, c=C_ECO, edgecolors='white',
               linewidths=0.6, zorder=zorder + 1)


def rule_line(ax, x=None, y=None, label=None, color=C_ECO, lw=0.9,
              shade_to=None, label_pos=None, fontsize=FS_ANNOT, va='bottom'):
    """阈值/规则线（≤20°、≤1% AEP 等），全图统一用生态朱红，呼应 Fig 1。"""
    if x is not None:
        if shade_to is not None:
            ax.axvspan(min(x, shade_to), max(x, shade_to), color=color, alpha=0.08,
                       zorder=0, lw=0)
        ax.axvline(x, color=color, ls='--', lw=lw, zorder=2)
        if label:
            ax.annotate(label, xy=(x, label_pos if label_pos is not None else 1.0),
                        xycoords=('data', 'axes fraction'), xytext=(2, -2),
                        textcoords='offset points', fontsize=fontsize,
                        fontweight='bold', color=color, ha='left', va='top')
    if y is not None:
        ax.axhline(y, color=color, ls=':', lw=lw, zorder=2)
        if label:
            ax.annotate(label, xy=(label_pos if label_pos is not None else 0.99, y),
                        xycoords=('axes fraction', 'data'), xytext=(-2, 2),
                        textcoords='offset points', fontsize=fontsize,
                        color=color, ha='right', va=va)


def cbar(fig, mappable, ax, label, ticks=None, size='4%', pad=0.05,
         labelsize=FS_TICK):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes('right', size=size, pad=pad,
                                              axes_class=plt.Axes)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=labelsize)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(width=0.4, length=1.8, labelsize=labelsize)
    cb.outline.set_linewidth(0.4)
    return cb
