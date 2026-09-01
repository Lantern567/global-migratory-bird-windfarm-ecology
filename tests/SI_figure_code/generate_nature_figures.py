# -*- coding: utf-8 -*-
"""
Nature/Elsevier 期刊级制图脚本（中文标签，Arial 风格无衬线字体）。
4 幅图：
  图1 研究区域与空间格局（全地图：海上/陆上/方向场/暴露降空间）
  图2 候鸟威胁数量、高度与方向特征（密度/通量/高度层分布/方向对比）
  图3 几何暴露模型与能源-生态交换（理论折线 + 交换散点 + 预算敏感度）
  图4 三组暴露降分布（小提琴/ECDF/RR>90%柱状）
颜色方案统一：
  蓝 #4A90E2（陆上/秋季）  绿 #2ECC71（VPTS）  紫 #9B59B6（Bauer）
  橙 #E67E22（初始假设）   红 #E74C3C（春季）
  陆地 #F5F5DC  海洋 #D4E6F1
注：Arial/Helvetica 无中文字形，中文标签用黑体(SimHei)渲染，其拉丁字形为 Arial 风格。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, '..', 'figures')
PROC = os.path.join(BASE, '..', 'data', 'processed')
VPTS_DIR = os.path.join(BASE, '..', 'data', 'raw', 'radar_vpts')
os.makedirs(FIG, exist_ok=True)

# ---------- 全局样式 ----------
def setup():
    wanted = ['SimHei', 'Microsoft YaHei', 'SimSun']
    have = {f.name for f in fm.fontManager.ttflist}
    first = next((w for w in wanted if w in have), 'SimHei')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [first, 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 8.5
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6

setup()

# ---------- 统一色板 ----------
C_BLUE   = '#4A90E2'
C_GREEN  = '#2ECC71'
C_PURPLE = '#9B59B6'
C_ORANGE = '#E67E22'
C_RED    = '#E74C3C'
C_BLACK  = '#2C3E50'
LAND     = '#F5F5DC'
OCEAN    = '#D4E6F1'

def style_ax(ax, ticks_in=False):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.6)
    d = 'in' if ticks_in else 'out'
    ax.tick_params(width=0.6, length=2.5, direction=d)

def panel_label(ax, txt):
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='top', ha='left', color='black')

# ---------- 数据 ----------
print('Loading data ...')
tradeoff = pd.read_csv(os.path.join(PROC, 'onshore_tradeoff_results.csv'), encoding='utf-8-sig')
onshore1 = tradeoff[tradeoff['budget'] == 0.01].copy()

off_dir = pd.read_csv(os.path.join(PROC, 'offshore_farm_directions_55.csv'), encoding='utf-8-sig')
off_tr  = pd.read_csv(os.path.join(PROC, 'tradeoff_offshore_55farms.csv'), encoding='utf-8-sig')
off_tr1 = off_tr[off_tr['budget'] == 0.01].copy()
off = off_dir.merge(off_tr1[['farm_id', 'theta_econ', 'theta_eco', 'aep_cost_pct',
                             'risk_reduction', 'risk_reduction_pct']], on='farm_id', how='inner')
off['source'] = off['source'].map({'VPTS': 'VPTS', 'Bauer_grid': 'Bauer'})
vpts  = off[off['source'] == 'VPTS'].copy()
bauer = off[off['source'] == 'Bauer'].copy()
print(f'  onshore 1%: {len(onshore1)}; VPTS {len(vpts)}; Bauer {len(bauer)}')

bauer_grid = pd.read_csv(os.path.join(PROC, 'bauer_grid_cell_directions.csv'), encoding='utf-8-sig')
radar = pd.read_csv(os.path.join(PROC, 'radar_station_signatures_v3.csv'), encoding='utf-8-sig')

# 汇总统计（用于信息框/标注）
med_on = onshore1['risk_reduction'].median()
aep_on = onshore1['aep_cost_pct'].mean()
med_vp = vpts['risk_reduction_pct'].median()
med_ba = bauer['risk_reduction_pct'].median()
rr90 = [(onshore1['risk_reduction'] > 90).mean() * 100,
        (vpts['risk_reduction_pct'] > 90).mean() * 100,
        (bauer['risk_reduction_pct'] > 90).mean() * 100]
print(f'  stats: med_on={med_on:.1f} aep_on={aep_on:.2f} med_vp={med_vp:.1f} med_ba={med_ba:.1f} rr90={[round(x,1) for x in rr90]}')

# 预算敏感性聚合（图3 区块三）
def _circ_dist(a, b):
    d = np.abs(a - b) % 180
    return np.minimum(d, 180 - d)

budgets = sorted(tradeoff['budget'].unique())
budget_agg = pd.DataFrame({
    'budget': budgets,
    'rr_med': [tradeoff[tradeoff['budget'] == b]['risk_reduction'].median() for b in budgets],
    'aep_mean': [tradeoff[tradeoff['budget'] == b]['aep_cost_pct'].mean() for b in budgets],
    'adj_med': [_circ_dist(tradeoff[tradeoff['budget'] == b]['theta_eco'],
                           tradeoff[tradeoff['budget'] == b]['theta_econ']).median() for b in budgets],
})
theta_econ_med = onshore1['theta_econ'].median()
theta_eco_med  = onshore1['theta_eco'].median()
print(f'  theta_econ_med={theta_econ_med:.0f}  theta_eco_med={theta_eco_med:.0f}')

# 理论 AEP 损失曲线（图3 X3）：逐场归一化后取总体均值
aep_curves = pd.read_csv(os.path.join(PROC, 'onshore_aep_curves.csv'), encoding='utf-8-sig')
aep_curves = aep_curves[aep_curves['farm_id'].isin(set(tradeoff['farm_id'].unique()))]
aep_cols = [c for c in aep_curves.columns if c.startswith('aep_')]
_aep_angles = np.array([int(c.split('_')[1]) for c in aep_cols])
_M = aep_curves[aep_cols].values.astype(float)
_M = _M / _M.max(axis=1, keepdims=True)          # 逐场归一化
aep_loss_mean = (1 - _M).mean(axis=0) * 100.0    # 总体平均损失（%）
print(f'  AEP loss curve: n={len(aep_curves)} farms, min={aep_loss_mean.min():.2f}% at {_aep_angles[int(aep_loss_mean.argmin())]}°')

# ---------- 高度层分布（图2 c）：原始 VPTS 200m 分层 ----------
def compute_height_profile():
    """逐层密度占比（200m 分层），夜迁(20-06)、dens>10、春(3-5月)/秋(8-11月)过滤。"""
    STATIONS = {'nlhrw', 'bejab', 'deess', 'frabb'}
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(float))
    def _f(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float('nan')
    for fname in sorted(os.listdir(VPTS_DIR)):
        if not fname.endswith('.txt'):
            continue
        st = fname.split('_')[0]
        if st not in STATIONS:
            continue
        with open(os.path.join(VPTS_DIR, fname), encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                p = line.split()
                if len(p) < 18:
                    continue
                try:
                    hght = int(p[2]); dens = _f(p[12])   # dens = 只/km³（第13列）
                except (ValueError, IndexError):
                    continue
                if dens != dens or dens < 10:
                    continue
                hour = int(p[1][:2])
                if 6 < hour < 20:
                    continue
                m = int(p[0][4:6])
                if 3 <= m <= 5:
                    season = 'spring'
                elif 8 <= m <= 11:
                    season = 'autumn'
                else:
                    continue
                acc[season][hght] += dens
    # 分箱：0-200, 200-400, ..., 1000-1200, >1200
    bins = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200)]
    out = {}
    for season in ('spring', 'autumn'):
        fracs = []
        for lo, hi in bins:
            v = sum(acc[season][h] for h in acc[season] if lo <= h < hi)
            fracs.append(v)
        fracs.append(sum(acc[season][h] for h in acc[season] if h >= 1200))
        tot = sum(fracs)
        out[season] = [f / tot * 100 if tot > 0 else 0 for f in fracs]
    return out

height_profile = compute_height_profile()
print('  height profile (spring rotor 0-200m): %.1f%%' % height_profile['spring'][0])
print('  height profile (autumn rotor 0-200m): %.1f%%' % height_profile['autumn'][0])

# ---------- cartopy ----------
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception as e:
    print('  cartopy unavailable:', e)
    HAS_CARTOPY = False

def add_basemap(ax, extent):
    ax.add_feature(cfeature.LAND, facecolor=LAND, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#999999', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#BBBBBB', zorder=1)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

def bauer_rect(ax, lw=1.0):
    ax.add_patch(Rectangle((-4.75, 43), 20.75, 12, fill=False, linestyle='--',
                           edgecolor='#888888', linewidth=lw, transform=ccrs.PlateCarree(), zorder=4))

# ============================================================
# 图 1：研究区域与空间格局（a 海上+雷达；b 陆上风场；c 春季方向场；
#        d 秋季方向场；e 暴露降空间分布）
# ============================================================
print('Fig 1 ...')
cmap_py = LinearSegmentedColormap.from_list('purple2yellow', ['#5B2C6F', '#9B59B6', '#E67E22', '#F7DC6F'])
if HAS_CARTOPY:
    fig = plt.figure(figsize=(9.5, 5.4))
    # 2 行 3 列网格：a,b,e 上排；c,d 下排；色条用 ax= 自动置于图右侧（细长竖条）
    gs = fig.add_gridspec(2, 3, wspace=0.02, hspace=0.05)
    axa = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())   # a 海上+雷达
    axb = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())   # b 陆上风场
    axe = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())   # e 暴露降空间
    axc = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())   # c 春季方向场
    axd = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())   # d 秋季方向场

    add_basemap(axa, [-12, 18, 41, 62])
    for ax in (axb, axc, axd, axe):
        add_basemap(ax, [-6, 17, 42, 56])
    bauer_rect(axa)

    # a: 海上 VPTS 绿圆 / Bauer 紫圆 + 雷达红三角
    axa.scatter(vpts['centroid_lon'], vpts['centroid_lat'], marker='o', s=16, c=C_GREEN,
                edgecolors='white', linewidth=0.4, label=f'海上·VPTS (n={len(vpts)})',
                transform=ccrs.PlateCarree(), zorder=5)
    axa.scatter(bauer['centroid_lon'], bauer['centroid_lat'], marker='o', s=16, c=C_PURPLE,
                edgecolors='white', linewidth=0.4, label=f'海上·Bauer (n={len(bauer)})',
                transform=ccrs.PlateCarree(), zorder=5)
    radar_loc = {'nlhrw': (52.95, 4.75), 'bejab': (51.18, 3.07), 'deess': (51.40, 6.97), 'frabb': (50.13, 1.83)}
    for sn, (la, lo) in radar_loc.items():
        axa.scatter(lo, la, marker='^', s=45, c='#C0392B', edgecolors='white', linewidth=0.6,
                    transform=ccrs.PlateCarree(), zorder=6)
        axa.text(lo, la + 0.55, sn, fontsize=6, ha='center', va='bottom', color='#C0392B',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.35),
                 transform=ccrs.PlateCarree(), zorder=7)
    axa.legend(loc='lower left', fontsize=5.5, frameon=True, facecolor='white',
               edgecolor='none', framealpha=1.0, borderpad=0.3)
    panel_label(axa, 'a')

    # b: 陆上极小浅蓝点（缩小 + alpha=0.5 防糊团）
    axb.scatter(onshore1['centroid_lon'], onshore1['centroid_lat'], s=0.8, c=C_BLUE,
                alpha=0.5, transform=ccrs.PlateCarree(), zorder=3)
    panel_label(axb, 'b')

    # c/d: 春季/秋季方向场（共享色标，置于 d 右侧）
    def quiver_map(ax, season):
        d = bauer_grid
        dirs = d[f'{season}_dir'].values; conc = d[f'{season}_conc'].values
        u = np.sin(np.radians(dirs)); v = np.cos(np.radians(dirs))
        sc = ax.quiver(d['lon'].values, d['lat'].values, u, v, conc, cmap=cmap_py,
                       scale=55, width=0.0018, headwidth=2.4, headlength=2.8,
                       transform=ccrs.PlateCarree(), zorder=3)
        sc.set_clim(0.5, 0.9)
        return sc
    sca = quiver_map(axc, 'spring'); scb = quiver_map(axd, 'autumn')
    panel_label(axc, 'c'); panel_label(axd, 'd')
    # 色条独立于地图轴（make_axes_locatable 不缩小地图），保证 b/d/e 与 a/c 主体同尺寸
    cax = make_axes_locatable(axd).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
    cbar = fig.colorbar(scb, cax=cax, orientation='vertical')
    cbar.ax.set_title('方向集中度', fontsize=7, pad=4)
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9])
    cbar.ax.tick_params(width=0.6, length=2.5, labelsize=6.5)

    # e: 暴露降空间分布（竖向色标置于右侧）
    sc = axe.scatter(onshore1['centroid_lon'], onshore1['centroid_lat'],
                     c=onshore1['risk_reduction'], cmap='RdYlGn', s=2.0, alpha=0.6,
                     edgecolors='none', linewidths=0,
                     vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=3)
    panel_label(axe, 'e')
    cax2 = make_axes_locatable(axe).append_axes('right', size='4%', pad=0.06, axes_class=plt.Axes)
    cbar2 = fig.colorbar(sc, cax=cax2, orientation='vertical')
    cbar2.set_label('暴露降 (%)', fontsize=7)
    cbar2.set_ticks([0, 25, 50, 75, 100])
    cbar2.ax.tick_params(width=0.6, length=2.5, labelsize=6.5)

    fig.savefig(os.path.join(FIG, 'fig_n1_study_area.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ============================================================
# 图 2：数量、高度与方向特征（a 密度；b 通量；c 高度层分布；d 方向对比）
# ============================================================
print('Fig 2 ...')
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
stations = ['deess', 'behel', 'bejab', 'frabb', 'nlhrw', 'nldhl']
spring_d = {r['station']: r['avg_density'] for _, r in radar[radar['season'] == 'spring'].iterrows()}
autumn_d = {r['station']: r['avg_density'] for _, r in radar[radar['season'] == 'autumn'].iterrows()}
spring_f = {r['station']: r['total_flux'] for _, r in radar[radar['season'] == 'spring'].iterrows()}
autumn_f = {r['station']: r['total_flux'] for _, r in radar[radar['season'] == 'autumn'].iterrows()}
x = np.arange(len(stations)); w = 0.36
sp_d = [spring_d.get(s, 0) for s in stations]; au_d = [autumn_d.get(s, 0) for s in stations]
sp_f = [spring_f.get(s, 0) for s in stations]; au_f = [autumn_f.get(s, 0) for s in stations]

# a: 平均密度
ax = axes[0, 0]
ax.bar(x - w/2, sp_d, w, color=C_RED, label='春季', alpha=0.9)
ax.bar(x + w/2, au_d, w, color=C_BLUE, label='秋季', alpha=0.9)
ax.set_yscale('log'); ax.set_ylim(10, 500)
ax.set_xticks(x); ax.set_xticklabels(stations, rotation=45, ha='right')
ax.set_ylabel('平均密度（只/km$^3$）')
panel_label(ax, 'a')
ax.legend(loc='upper right'); style_ax(ax, ticks_in=True)

# b: 总通量
ax = axes[0, 1]
ax.bar(x - w/2, sp_f, w, color=C_RED, label='春季', alpha=0.9)
ax.bar(x + w/2, au_f, w, color=C_BLUE, label='秋季', alpha=0.9)
ax.set_yscale('log'); ax.set_ylim(1e4, 1e6)
ax.set_xticks(x); ax.set_xticklabels(stations, rotation=45, ha='right')
ax.set_ylabel('总通量（密度积分，相对单位）')
panel_label(ax, 'b')
ax.legend(loc='upper right'); style_ax(ax, ticks_in=True)

# c: 飞行高度层分布直方图（200m 分层，聚合所有 VPTS 站点）
ax = axes[1, 0]
bin_labels = ['0–200', '200–400', '400–600', '600–800', '800–1000', '1000–1200', '>1200']
bx = np.arange(len(bin_labels)); bw = 0.36
ax.bar(bx - bw/2, height_profile['spring'], bw, color=C_RED, label='春季', alpha=0.9)
ax.bar(bx + bw/2, height_profile['autumn'], bw, color=C_BLUE, label='秋季', alpha=0.9)
ax.set_xticks(bx); ax.set_xticklabels(bin_labels, rotation=45, ha='right')
ax.set_ylabel('密度占比（%）')
ax.set_xlabel('飞行高度层（m）')
ax.set_ylim(0, max(height_profile['spring'] + height_profile['autumn']) * 1.18)
panel_label(ax, 'c')
ax.legend(loc='upper right'); style_ax(ax, ticks_in=True)

# d: 春秋飞行方向（极坐标散点，配对站点连线显示反平行偏移）
axes[1, 1].remove()
ax = fig.add_subplot(2, 2, 4, projection='polar')
def _has_dir(v):
    return v == v and not np.isnan(v)

dir_s = {r['station']: r['direction_deg'] for _, r in radar[radar['season'] == 'spring'].iterrows()
         if _has_dir(r['direction_deg'])}
dir_a = {r['station']: r['direction_deg'] for _, r in radar[radar['season'] == 'autumn'].iterrows()
         if _has_dir(r['direction_deg'])}
paired = [s for s in dir_s if s in dir_a]
ax.set_theta_zero_location('N')          # 0° 朝上（北）
ax.set_theta_direction('clockwise')      # 顺时针（罗盘约定）
# 配对站点连线（单位圆弦，反平行时近似穿过圆心）
for s in paired:
    ax.plot([np.radians(dir_s[s]), np.radians(dir_a[s])], [1, 1],
            color='#999999', lw=0.8, ls='--', alpha=0.7, zorder=3)
ax.scatter(np.radians([dir_s[s] for s in dir_s]), [1]*len(dir_s), marker='o', s=48,
           c=C_RED, edgecolors='white', linewidth=0.7, zorder=5, label=f'春季（n={len(dir_s)}）')
ax.scatter(np.radians([dir_a[s] for s in dir_a]), [1]*len(dir_a), marker='o', s=48,
           c=C_BLUE, edgecolors='white', linewidth=0.7, zorder=5, label=f'秋季（n={len(dir_a)}）')
ax.set_ylim(0, 1.15)
ax.set_yticks([])                        # 半径刻度无实际含义，隐藏
ax.tick_params(labelsize=6.5, pad=-14)
panel_label(ax, 'd')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=6,
          frameon=False, ncol=2, handletextpad=0.4)
ax.text(0.5, -0.14, f'虚线连接 {len(paired)} 个同时有春/秋方向观测的站点（近乎反平行）',
        transform=ax.transAxes, ha='center', va='top', fontsize=5.5, color='#555555')

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_n2_abundance.png'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============================================================
# 图 3：几何暴露模型与能源-生态交换（3 区块）
#        区块一：几何原理折线（a E(θ)；b 暴露降 1−E(θ)；c 理论 AEP 损失）
#        区块二：交换散点（d 陆上；e 海上）
#        区块三：预算敏感性折线（f 中位暴露降；g 平均 AEP 代价；h 生态调整角）
# ============================================================
print('Fig 3 ...')
fig = plt.figure(figsize=(7.4, 6.8))
gs = fig.add_gridspec(3, 6, hspace=0.55, wspace=0.85,
                      height_ratios=[1, 1, 1])
ax_X1 = fig.add_subplot(gs[0, 0:2])   # a E(θ)
ax_X2 = fig.add_subplot(gs[0, 2:4])   # b 暴露降 1−E(θ)
ax_X3 = fig.add_subplot(gs[0, 4:6])   # c 理论 AEP 损失
ax_land = fig.add_subplot(gs[1, 0:3]) # d 陆上散点
ax_sea  = fig.add_subplot(gs[1, 3:6]) # e 海上散点
ax_b1 = fig.add_subplot(gs[2, 0:2])   # f 中位暴露降
ax_b2 = fig.add_subplot(gs[2, 2:4])   # g 平均 AEP 代价
ax_b3 = fig.add_subplot(gs[2, 4:6])   # h 生态调整角

theta = np.linspace(0, 180, 400)
PHI_S, PHI_A = 51.5, 55.0
E_s = np.sin(np.radians(theta - PHI_S))**2
E_a = np.sin(np.radians(theta - PHI_A))**2
E_avg = 0.5 * (E_s + E_a)

# 区块一：几何原理折线（共用一个横轴 θ，配色一致）
C_DGREEN = '#228B22'   # 深绿（暴露≈0 / 平行）
C_RBROWN = '#8B4513'   # 红褐（暴露≈1 / 垂直）
ANN_BOX = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9)
E_55  = 0.5 * (np.sin(np.radians(55 - PHI_S))**2 + np.sin(np.radians(55 - PHI_A))**2)   # ≈0
E_145 = 0.5 * (np.sin(np.radians(145 - PHI_S))**2 + np.sin(np.radians(145 - PHI_A))**2)  # ≈1

def _plot_theory(ax, ys, ylab, lab):
    ax.plot(theta, ys[0], color=C_RED, lw=1.4, label='春季')
    ax.plot(theta, ys[1], color=C_BLUE, lw=1.4, label='秋季')
    ax.plot(theta, ys[2], color='black', lw=1.2, ls='--', label='平均')
    ax.axvline(theta_eco_med, color=C_GREEN, ls='--', lw=1.0, alpha=0.7)
    ax.set_xlim(0, 180); ax.set_ylim(0, 1.03)
    ax.set_xlabel('阵列朝向 θ（°）'); ax.set_ylabel(ylab)
    panel_label(ax, lab)
    style_ax(ax)

# a: 几何暴露 E(θ)
_plot_theory(ax_X1, (E_s, E_a, E_avg), '几何暴露 E(θ)', 'a')
ax_X1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), fontsize=6, frameon=False, ncol=3)
ax_X1.annotate('平均暴露≈0，θ≈55°', xy=(55, E_55), xytext=(12, 0.14),
               fontsize=7, color=C_DGREEN, ha='left', va='bottom', bbox=ANN_BOX,
               arrowprops=dict(arrowstyle='->', color=C_DGREEN, lw=1.1))
ax_X1.annotate('垂直暴露≈1，θ≈145°', xy=(145, E_145), xytext=(135, 0.40),
               fontsize=7, color=C_RBROWN, ha='center', va='bottom', bbox=ANN_BOX,
               arrowprops=dict(arrowstyle='->', color=C_RBROWN, lw=1.1))

# b: 暴露降潜力 1−E(θ)
_plot_theory(ax_X2, (1 - E_s, 1 - E_a, 1 - E_avg), '暴露降潜力 1-E(θ)', 'b')
ax_X2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), fontsize=6, frameon=False, ncol=3)
ax_X2.annotate('暴露降潜力≈100%，θ≈55°', xy=(55, 1 - E_55), xytext=(12, 0.40),
               fontsize=7, color=C_DGREEN, ha='left', va='bottom', bbox=ANN_BOX,
               arrowprops=dict(arrowstyle='->', color=C_DGREEN, lw=1.1))
ax_X2.annotate('暴露降潜力≈0，θ≈145°', xy=(145, 1 - E_145), xytext=(100, 0.14),
               fontsize=7, color=C_RBROWN, ha='center', va='bottom', bbox=ANN_BOX,
               arrowprops=dict(arrowstyle='->', color=C_RBROWN, lw=1.1))

# c: 理论 AEP 损失（总体平均，归一化后）
ax_X3.plot(_aep_angles, aep_loss_mean, 'o-', color=C_BLACK, lw=1.5, markersize=3)
ax_X3.axvline(theta_econ_med, color='black', ls=':', lw=1.0, alpha=0.8)
ax_X3.axvline(theta_eco_med, color=C_GREEN, ls='--', lw=1.0, alpha=0.7)
ax_X3.set_xlim(0, 180); ax_X3.set_ylim(0, 2.0)
ax_X3.set_xlabel('阵列朝向 θ（°）'); ax_X3.set_ylabel('平均 AEP 损失（%）')
panel_label(ax_X3, 'c')
style_ax(ax_X3)

# 区块二：交换散点
rng = np.random.default_rng(0)
jx = rng.normal(0, 0.006, len(onshore1)); jy = rng.normal(0, 0.7, len(onshore1))
ax_land.scatter(onshore1['aep_cost_pct'] + jx, onshore1['risk_reduction'] + jy,
                s=2, c=C_BLUE, alpha=0.12, edgecolors='none')
ax_land.axvline(0.5, color=C_BLUE, ls='--', lw=0.9, alpha=0.55)
ax_land.text(0.97, 0.10, f'中位暴露降：{med_on:.0f}%\n平均 AEP 代价：{aep_on:.2f}%',
             transform=ax_land.transAxes, ha='right', va='bottom', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='none'))
ax_land.set_xlabel('AEP 代价（%）'); ax_land.set_ylabel('暴露降（%）')
ax_land.set_xlim(0, 1.0); ax_land.set_ylim(-5, 105)
panel_label(ax_land, 'd')
style_ax(ax_land)

ax_sea.scatter(vpts['aep_cost_pct'], vpts['risk_reduction_pct'], marker='o', s=26, c=C_GREEN,
               edgecolors='white', linewidth=0.4, label=f'VPTS (n={len(vpts)})')
ax_sea.scatter(bauer['aep_cost_pct'], bauer['risk_reduction_pct'], marker='o', s=26, c=C_PURPLE,
               edgecolors='white', linewidth=0.4, label=f'Bauer (n={len(bauer)})')
ax_sea.axhline(med_vp, color=C_GREEN, ls='--', lw=0.9, alpha=0.55)
ax_sea.axhline(med_ba, color=C_PURPLE, ls='--', lw=0.9, alpha=0.55)
ax_sea.axvline(0.5, color=C_BLUE, ls='--', lw=0.9, alpha=0.55)
ax_sea.set_xlabel('AEP 代价（%）'); ax_sea.set_xlim(0, 1.0); ax_sea.set_ylim(-5, 105)
panel_label(ax_sea, 'e')
ax_sea.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6,
              frameon=True, facecolor='white', edgecolor='none', framealpha=0.85)
style_ax(ax_sea)

# 区块三：预算敏感性折线组
cats = ['0.5', '1', '2', '5']; xpos = np.arange(len(cats))
# f: 中位暴露降
ax_b1.plot(xpos, budget_agg['rr_med'], 'o-', color=C_BLUE, lw=1.5, markersize=4)
for i, v in enumerate(budget_agg['rr_med']):
    ax_b1.annotate(f'{v:.0f}', (xpos[i], v), textcoords='offset points', xytext=(0, 5),
                   fontsize=6.5, ha='center', color=C_BLUE)
ax_b1.set_xticks(xpos); ax_b1.set_xticklabels(cats)
ax_b1.set_xlabel('AEP 预算（%）'); ax_b1.set_ylabel('中位暴露降（%）')
ax_b1.set_ylim(70, 105)
panel_label(ax_b1, 'f')
style_ax(ax_b1)
# g: 平均 AEP 代价
ax_b2.plot(xpos, budget_agg['aep_mean'], 's-', color=C_ORANGE, lw=1.5, markersize=4)
for i, v in enumerate(budget_agg['aep_mean']):
    ax_b2.annotate(f'{v:.2f}', (xpos[i], v), textcoords='offset points', xytext=(0, 5),
                   fontsize=6.5, ha='center', color=C_ORANGE)
ax_b2.set_xticks(xpos); ax_b2.set_xticklabels(cats)
ax_b2.set_xlabel('AEP 预算（%）'); ax_b2.set_ylabel('平均 AEP 代价（%）')
ax_b2.set_ylim(0, 1.25)
panel_label(ax_b2, 'g')
style_ax(ax_b2)
# h: 生态调整角（|θ_eco − θ_econ| 中位）
ax_b3.plot(xpos, budget_agg['adj_med'], '^-', color=C_GREEN, lw=1.5, markersize=4)
for i, v in enumerate(budget_agg['adj_med']):
    ax_b3.annotate(f'{v:.0f}°', (xpos[i], v), textcoords='offset points', xytext=(0, 5),
                   fontsize=6.5, ha='center', color=C_GREEN)
ax_b3.set_xticks(xpos); ax_b3.set_xticklabels(cats)
ax_b3.set_xlabel('AEP 预算（%）'); ax_b3.set_ylabel('生态调整角（°）')
ax_b3.set_ylim(20, 60)
panel_label(ax_b3, 'h')
style_ax(ax_b3)

fig.savefig(os.path.join(FIG, 'fig_n3_tradeoff.png'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============================================================
# 图 4：三组暴露降分布（a 小提琴；b ECDF；c RR>90% 柱状）
# ============================================================
print('Fig 4 ...')
fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(6.6, 2.4),
                                    gridspec_kw={'width_ratios': [1.25, 1.1, 0.85]})
df_vio = pd.DataFrame({
    '暴露降 (%)': np.concatenate([onshore1['risk_reduction'].values,
                                  vpts['risk_reduction_pct'].values,
                                  bauer['risk_reduction_pct'].values]),
    '组': (['陆上'] * len(onshore1) + ['VPTS'] * len(vpts) + ['Bauer'] * len(bauer)),
})
sns.violinplot(data=df_vio, x='组', y='暴露降 (%)', ax=axa, order=['陆上', 'VPTS', 'Bauer'],
               hue='组', palette={'陆上': C_BLUE, 'VPTS': C_GREEN, 'Bauer': C_PURPLE},
               inner=None, linewidth=0.6, saturation=0.85, legend=False)
for i, vals in enumerate([onshore1['risk_reduction'].values,
                          vpts['risk_reduction_pct'].values,
                          bauer['risk_reduction_pct'].values]):
    m = np.median(vals)
    axa.hlines(m, i - 0.28, i + 0.28, color='black', lw=1.4, zorder=5)
    axa.text(i, 102, f'中位 {m:.0f}%', ha='center', va='top', fontsize=7, color='black', zorder=6)
axa.set_xlabel(''); axa.set_ylabel('暴露降（%）'); axa.set_ylim(-5, 112)
panel_label(axa, 'a')
style_ax(axa)

df_ecdf = pd.DataFrame({
    '暴露降 (%)': np.concatenate([onshore1['risk_reduction'].values,
                                  vpts['risk_reduction_pct'].values,
                                  bauer['risk_reduction_pct'].values]),
    '组': (['陆上（中位 %.0f%%）' % med_on] * len(onshore1) +
           ['VPTS（中位 %.0f%%）' % med_vp] * len(vpts) +
           ['Bauer（中位 %.0f%%）' % med_ba] * len(bauer)),
})
sns.ecdfplot(data=df_ecdf, x='暴露降 (%)', hue='组', ax=axb,
             palette={'陆上（中位 %.0f%%）' % med_on: C_BLUE, 'VPTS（中位 %.0f%%）' % med_vp: C_GREEN,
                      'Bauer（中位 %.0f%%）' % med_ba: C_PURPLE}, linewidth=1.4)
axb.set_xlim(0, 100); axb.set_ylim(0, 1.02)
axb.set_xlabel('暴露降（%）'); axb.set_ylabel('累积比例')
panel_label(axb, 'b')
sns.move_legend(axb, 'upper left', fontsize=6.5, frameon=False)
style_ax(axb)

axc.bar([1, 2, 3], rr90, color=[C_BLUE, C_GREEN, C_PURPLE], width=0.55, alpha=0.9)
for i, v in enumerate(rr90):
    axc.text(i + 1, v + 2, f'{v:.0f}%', ha='center', fontsize=7)
axc.set_xticks([1, 2, 3]); axc.set_xticklabels(['陆上', 'VPTS', 'Bauer'])
axc.set_ylabel('RR>90% 比例（%）'); axc.set_ylim(0, 100)
panel_label(axc, 'c')
style_ax(axc)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_n4_distribution.png'), bbox_inches='tight', facecolor='white')
plt.close(fig)

print('\nDone. Figures:')
for f in ['fig_n1_study_area.png', 'fig_n2_abundance.png', 'fig_n3_tradeoff.png',
          'fig_n4_distribution.png']:
    p = os.path.join(FIG, f)
    print(f'  {f}  ({os.path.getsize(p)/1e3:.0f} KB)' if os.path.exists(p) else f'  {f}  MISSING')

# 清理旧文件名（重构后不再使用）
for stale in ['fig_n3_direction_field.png', 'fig_n5_tradeoff.png', 'fig_n7_conclusion.png', 'fig_n8_spatial_rr.png']:
    sp = os.path.join(FIG, stale)
    if os.path.exists(sp):
        os.remove(sp)
        print('  removed stale', stale)