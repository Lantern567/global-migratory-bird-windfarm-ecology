# -*- coding: utf-8 -*-
"""
build_word_v7.py —— 将 manuscript_en.md 编译为 Nature Energy 版 Word（v7），
                    图片来自 figures_v6/（新的多子图重设计），并动态替换过时的
                    figure caption 段落以匹配 v6 的 panel 结构。

改动与 v5 差异：
  * 图源目录：figures_v6/*.png（而非 outputs/figure）
  * 图 caption 采用 v7 panel 结构（sin² surface / Pareto frontier / ridgeline
    / hex-bin / case-study / saturation heatmap 等）
  * 移除 Table 2（并入 Fig 5i heatmap）
  * 输出至 outputs/reports/…v7.docx
"""
import os
import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, '..'))
MD_PATH = os.path.join(REPO, 'outputs', 'reports', 'manuscript_en.md')
FIG_DIR = os.path.join(REPO, 'figures_v6')
OUT_PATH = os.path.join(REPO, 'outputs', 'reports',
                        'Wind farm array orientation strongly shapes migratory bird collision exposure (v7).docx')

FIG_FILES = {
    '1': os.path.join(FIG_DIR, 'fig1_mechanism.png'),
    '2': os.path.join(FIG_DIR, 'fig2_directional.png'),
    '3': os.path.join(FIG_DIR, 'fig3_misalign.png'),
    '4': os.path.join(FIG_DIR, 'fig4_capture.png'),
    '5': os.path.join(FIG_DIR, 'fig5_tradeoff.png'),
    'S1': os.path.join(FIG_DIR, 'figS1_threat.png'),
    'S2': os.path.join(FIG_DIR, 'figS2_universality.png'),
}
FIG_WIDTH = {k: Inches(6.5) for k in FIG_FILES}

# 新版 caption（英文，对应 v6 panel 结构）
NEW_CAPTIONS_EN = {
    '1': (
        "**Figure 1 | Mechanism of orientation-driven exposure and regional evidence.** "
        "**a**, Top-down schematic of parallel (low exposure) vs perpendicular (high exposure) "
        "array geometries relative to migration. **b**, Full exposure surface E(θ, φ) = "
        "sin²(θ − φ) with contour lines; the AEP-opt and eco-opt orientations diverge by ≈55° "
        "at an example migration axis of 100°. **c**, Compass-ring view of the median AEP-opt "
        "vs eco-opt orientations for the three groups (Δ = per-group offset). **d**, Geometric "
        "model callout with the E ∝ sin²(θ − φ) relation. **e**, Capture curves (median ± IQR "
        "band) of exposure reduction against rotation from AEP-opt for the three groups; the "
        "first 20° captures most of the gain. **f**, Energy–exposure Pareto frontier: each "
        "small point is a farm × budget outcome, coloured by group; large markers mark per-"
        "budget medians; the shaded band is the ≤1% AEP corridor. **g**, Study region overlaid "
        "with the spring migration direction quiver (colour = directional concentration) and "
        "wind-farm locations (4,191 onshore + 29 VPTS + 26 Bauer + 6 radar stations; red "
        "rectangle marks the North-Sea zoom of panels 5e,f). **h**, Onshore Δθ50 hex-bin map "
        "(median rotation required to capture half the gain), with offshore farms overlaid."
    ),
    '2': (
        "**Figure 2 | Directional structure of migration and orientation sensitivity are "
        "cross-source robust.** **a**, Rose diagram of spring (orange) and autumn (blue) "
        "migration directions across 2,025 Bauer grid cells; arrows show vector means. "
        "**b**, Ridgeline of normalised exposure curves E(θ) for representative farms of each "
        "group, showing that the sin²-family shape is not model-specific. **c**, Per-farm "
        "sensitivity amplitude (rel = (E_max − E_min)/E_max) shown as violin + jitter; every "
        "farm exceeds 85% and group medians are 99.9% / 97.6% / 93.4%. **d**, Cumulative "
        "share of farms above a rel threshold with a bootstrap 95% CI band per group. "
        "**e**, Hex-bin map of median onshore rel across space with offshore farms overlaid; "
        "sensitivity is spatially uniform. **f**, Mechanism scatter: onshore per-farm rel "
        "against spring directional concentration, with a binned median±IQR band; higher "
        "concentration produces marginally higher sensitivity as the sin² geometry predicts."
    ),
    '3': (
        "**Figure 3 | AEP-optimal and exposure-optimal orientations are systematically "
        "misaligned.** **a**, Hex-bin scatter of eco-opt θ_min against AEP-opt θ_econ for "
        "all 4,246 farms (log-scale hex density); the red diagonal marks alignment; median "
        "positions of each group are shown as large markers. **b**, Marginal density of the "
        "misalignment angle d_full on top; dumbbell of per-group medians below with per-group "
        "offset labelled (25°/20°/5° in this snapshot; see main text for full IQR). "
        "**c**, Log-scale bars of E(θ_econ)/E_min with median±IQR whiskers and sample sizes "
        "(≈600×, 19×, 10×). **d**, Polar overlay for a representative onshore farm: spring "
        "and autumn migration direction lobes vs the AEP-opt array orientation. **e**, Case-"
        "study farms (worst-aligned d ≈ 90° vs best-aligned d ≈ 0°) showing normalised "
        "exposure E(θ) (solid) and normalised AEP(θ) (dashed) on twin axes; the ecological "
        "and energy optima are visibly staggered on the worst-aligned farm and coincide on "
        "the best-aligned one."
    ),
    '4': (
        "**Figure 4 | A limited rotation captures most of the reducible exposure.** "
        "**a**, Capture curves (median ± IQR) of exposure reduction versus rotation from "
        "AEP-opt for the three groups, with vertical drops at each group's Δθ50 (17°/14°/18°) "
        "and horizontal reference lines at 50%/80%; the inset shows the marginal gain (%/°), "
        "which peaks near the AEP-opt and decays past 20°. **b**, Half-gain rotation Δθ50 "
        "against full misalignment d_full: onshore farms as coloured points (colour = rel, "
        "size = n_turbines), offshore farms as bold markers; the 1:1 diagonal is the upper "
        "bound. **c**, Bee-swarm + violin of onshore Δθ50 by country (top 6 countries by "
        "sample size); country medians are similar (≈15–20°). **d**, Bee-swarm + violin of "
        "Δθ80 by group (29°/25°/30°). **e**, Grouped bars of the fraction of farms crossing "
        "each threshold (≥50%/≥80% within 20°/30°); the majority of farms recover more than "
        "half of the benefit within 20°."
    ),
    '5': (
        "**Figure 5 | Energy–exposure trade-off delivers most of the ecological gain within "
        "1% AEP.** **a**, Large Pareto scatter of exposure reduction versus AEP loss for "
        "4,246 farms × 4 budgets (0.5/1/2/5%); small points are individual farm × budget "
        "outcomes coloured by group; large markers show per-budget medians per group; the "
        "shaded band is the ≤1% AEP corridor. **b–c**, Ridgeline distributions of exposure "
        "reduction by AEP budget for onshore (b) and offshore (c) — the distribution "
        "compresses toward 100% as the budget rises. **d**, Onshore hex-bin choropleth of "
        "median exposure reduction at 1% AEP. **e–f**, Offshore VPTS (e) and Bauer (f) "
        "farm-level RR at 1% AEP over the North-Sea zoom. **g**, Median exposure reduction "
        "versus AEP budget for the three groups; benefit saturates past 1%. **h**, Farm-level "
        "saturation heatmap for onshore farms (rows sorted by Δθ50 from 1° at the top to 45° "
        "at the bottom, columns = four AEP budgets, colour = RR); farms with small Δθ50 "
        "saturate to ≥90% RR by the 1% budget; hard-case farms (large Δθ50) need higher "
        "budgets. **i**, Compact heatmap of median RR (%) across the four budgets × three "
        "groups — a visual replacement for the previous Table 2."
    ),
}

# 中文版对应 caption（关键描述与英文一致）
NEW_CAPTIONS_ZH = {
    '1': (
        "**Figure 1 | 朝向驱动暴露的机制与区域证据。** "
        "**a**，阵列几何俯视示意：平行迁徙方向 → 低暴露；垂直 → 高暴露。"
        "**b**，暴露曲面 E(θ, φ) = sin²(θ − φ) 全景（含等值线）；以迁徙轴 φ = 100° 为例，"
        "AEP 最优与生态最优朝向相差 ≈55°。**c**，罗盘环视图：三组 AEP-opt 与 eco-opt 中位方向 "
        "对比（Δ 为分组偏移量）。**d**，几何模型示意与公式 E ∝ sin²(θ − φ)。"
        "**e**，三组的暴露削减-旋转捕获曲线（中位 ± IQR 带）；前 20° 已捕获大部分收益。"
        "**f**，能源–暴露 Pareto 前沿：每个小点为 farm × 预算，按组着色；大标记为分组分预算中位；"
        "阴影带为 ≤1% AEP 走廊。**g**，研究区总览：叠加春季迁徙方向箭矢（色 = 方向集中度）与 "
        "风电场位置（陆上 4,191、VPTS 海上 29、Bauer 海上 26、雷达站 6；红框为 5e,f 面板的北海放大区）。"
        "**h**，陆上 Δθ50 六边形聚合空间图（捕获半数收益所需的旋转角中位），并叠加海上场址。"
    ),
    '2': (
        "**Figure 2 | 迁徙的方向结构与跨源稳健的朝向敏感性。** **a**，2,025 个 Bauer 网格的春（橙）"
        "秋（蓝）迁徙方向玫瑰，箭矢为向量均值。**b**，三组代表场址归一化 E(θ) 曲线的岭线图，"
        "表明 sin² 曲线家族并非模型特有。**c**，逐场址方向敏感性 rel = (E_max − E_min)/E_max 的 "
        "小提琴 + 抖点：每一场址均 >85%，分组中位分别为 99.9%/97.6%/93.4%。**d**，累积经验分布：给定 "
        "rel 阈值下的场址占比，每组带自助 95% CI 带。**e**，陆上 rel 中位的六边形聚合空间图，"
        "叠加海上场址；敏感性空间分布均匀。**f**，机制散点：陆上 rel 对春季方向集中度的关系，带分箱 "
        "中位±IQR 带；集中度越高，敏感性相应越高，与 sin² 几何一致。"
    ),
    '3': (
        "**Figure 3 | AEP 最优与生态最优朝向的系统性错位。** **a**，4,246 个风电场 θ_min 对 θ_econ "
        "的六边形密度散点（对数密度），红对角为对齐参考；每组中位以大标记显示。**b**，上：错位角 "
        "d_full 的边缘密度直方；下：三组中位方向哑铃图（本快照下每组偏移分别为 25°/20°/5°；见正文完整 IQR）。"
        "**c**，E(θ_econ)/E_min 对数柱（中位±IQR 误差棒，标注样本量）：三组分别约 600×、19×、10×。"
        "**d**，代表陆上场址极坐标叠加：春秋迁徙方向瓣与 AEP 最优阵列朝向。**e**，个案研究：错位最差 "
        "（d ≈ 90°）与最好对齐（d ≈ 0°）的两个陆上场址，双轴同框展示归一化 E(θ)（实线）与归一化 "
        "AEP(θ)（虚线），可直观看到能源与生态最优在错位场上明显错开、在对齐场上重合。"
    ),
    '4': (
        "**Figure 4 | 有限旋转即可捕获大部分可削减暴露。** "
        "**a**，三组捕获曲线（中位 ± IQR），在各组 Δθ50（17°/14°/18°）处纵向标注、50%/80% 水平参考；"
        "内嵌图展示单位角度的边际增益（%/°），在 AEP-opt 附近最大、超过 20° 后迅速衰减。"
        "**b**，Δθ50 对 d_full 的散点：陆上点色为 rel、点面积为机组数；海上以大标记；1:1 对角线为上界。"
        "**c**，陆上 Δθ50 按国家的蜂群 + 小提琴（按样本量前 6 国）；国别中位差异有限（≈15–20°）。"
        "**d**，三组 Δθ80 蜂群 + 小提琴（29°/25°/30°）。"
        "**e**，各阈值达成率分组柱（≥50%/≥80% within 20°/30°）：多数场址在 20° 内即可回收超过一半收益。"
    ),
    '5': (
        "**Figure 5 | 能源–暴露权衡：1% AEP 内即可获得大部分生态收益。** "
        "**a**，大幅 Pareto 散点：暴露削减对 AEP 损失，4,246 场址 × 4 预算（0.5/1/2/5%）；小点为个体，"
        "按组着色；大标记为分组分预算中位；阴影带为 ≤1% AEP 走廊。**b–c**，按预算分层的暴露削减 "
        "岭线图，(b) 陆上、(c) 海上：预算上升时分布向 100% 压缩。**d**，1% 预算下陆上 RR 的六边形 "
        "聚合专题图。**e–f**，1% 预算下海上 VPTS (e) 与 Bauer (f) 场址级 RR（北海放大）。"
        "**g**，三组 RR 中位对预算：>1% 后收益饱和。**h**，逐场址饱和热图：行按 Δθ50 排序（顶 1° → "
        "底 45°），列为四个预算，色为 RR；Δθ50 小的场址在 1% 预算即饱和至 ≥90% RR，Δθ50 大的疑难 "
        "场址需要更高预算。**i**，紧凑热图：三组 × 四预算的 RR (%) 中位——用作原 Table 2 的可视化替代。"
    ),
}

# Table 2 现由 Fig 5i 替代 —— 从 md 里删除相关行
TABLE2_HEADER_RE = re.compile(r'^\*\*Table 2 \|', re.IGNORECASE)


def _set_font(run, size=10.5, bold=False, italic=False, color=None):
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def _add_inline_runs(par, text, base_size=10.5, base_bold=False):
    tok = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|<sup>[^<]+</sup>)', text)
    for t in tok:
        if not t:
            continue
        if t.startswith('**') and t.endswith('**'):
            r = par.add_run(t[2:-2]); _set_font(r, base_size, bold=True)
        elif t.startswith('*') and t.endswith('*'):
            r = par.add_run(t[1:-1]); _set_font(r, base_size, bold=base_bold, italic=True)
        elif t.startswith('`') and t.endswith('`'):
            r = par.add_run(t[1:-1]); _set_font(r, base_size, bold=base_bold)
            r.font.name = 'Consolas'
        elif t.startswith('<sup>') and t.endswith('</sup>'):
            r = par.add_run(t[5:-6]); _set_font(r, base_size, bold=base_bold)
            r.font.superscript = True
        else:
            r = par.add_run(t); _set_font(r, base_size, bold=base_bold)


def _insert_figure(doc, fig_key):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fp = FIG_FILES.get(fig_key)
    if fp and os.path.exists(fp):
        run.add_picture(fp, width=FIG_WIDTH.get(fig_key, Inches(6.5)))
    else:
        run.text = f'[Figure {fig_key} missing]'


def _add_table(doc, tbl_lines):
    rows = []
    for ln in tbl_lines:
        ln = ln.strip()
        if not ln.startswith('|'):
            continue
        cells = [c.strip() for c in ln.strip('|').split('|')]
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=ncol)
    tbl.style = 'Light Grid Accent 1'
    for ri, r in enumerate(rows):
        for ci in range(ncol):
            txt = r[ci] if ci < len(r) else ''
            cell = tbl.cell(ri, ci)
            cell.text = ''
            p = cell.paragraphs[0]
            _add_inline_runs(p, txt, base_size=9.5, base_bold=(ri == 0))
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def _remove_table2_block(lines):
    """去掉 Table 2 段落 + 后续表格（改为 Fig 5i heatmap 替代）。"""
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if TABLE2_HEADER_RE.match(line.lstrip()):
            # 跳过：Table 2 caption 段
            i += 1
            # 跳过：空行
            while i < n and lines[i].strip() == '':
                i += 1
            # 跳过：表格（连续 | 开头的行）
            while i < n and lines[i].lstrip().startswith('|'):
                i += 1
            # 跳过：紧随其后的空行
            while i < n and lines[i].strip() == '':
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def build(md_path, out_path, captions):
    doc = Document()
    for s in doc.sections:
        s.top_margin = Cm(2.0); s.bottom_margin = Cm(2.0)
        s.left_margin = Cm(2.0); s.right_margin = Cm(2.0)
    doc.styles['Normal'].font.name = 'Arial'
    doc.styles['Normal'].font.size = Pt(10.5)

    lines = open(md_path, encoding='utf-8').read().splitlines()
    lines = _remove_table2_block(lines)

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip()

        if re.match(r'^-{3,}\s*$', stripped):
            i += 1; continue

        if stripped.startswith('# '):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(stripped[2:].strip()); _set_font(r, 14, bold=True)
            i += 1; continue
        if stripped.startswith('### '):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(stripped[4:].strip()); _set_font(r, 12, italic=True)
            i += 1; continue
        if stripped.startswith('## '):
            p = doc.add_paragraph()
            r = p.add_run(stripped[3:].strip()); _set_font(r, 12, bold=True)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            i += 1; continue

        if stripped.startswith('|'):
            tbl_lines = []
            while i < n and lines[i].lstrip().startswith('|'):
                tbl_lines.append(lines[i]); i += 1
            _add_table(doc, tbl_lines)
            continue

        if stripped == '':
            i += 1; continue

        # 图/附图 caption：替换主图 caption 为 v7 版
        m = re.match(r'^\*\*(Figure|Supplementary Fig\.)\s+(S?\d+)\s*\|', stripped)
        if m:
            fig_key = m.group(2)
            _insert_figure(doc, fig_key)
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(12)
            cap_text = captions.get(fig_key, stripped)
            _add_inline_runs(p, cap_text, base_size=9.5)
            i += 1; continue

        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        _add_inline_runs(p, stripped, base_size=10.5)
        i += 1

    doc.save(out_path)
    print('wrote:', out_path)


if __name__ == '__main__':
    # 英文
    build(MD_PATH, OUT_PATH, NEW_CAPTIONS_EN)
    # 中文
    zh_md = os.path.join(REPO, 'outputs', 'reports', 'manuscript_zh.md')
    zh_out = os.path.join(REPO, 'outputs', 'reports',
                          '风电场阵列朝向显著塑造候鸟碰撞暴露（西欧海上与陆上风电场的几何筛查）v7.docx')
    if os.path.exists(zh_md):
        build(zh_md, zh_out, NEW_CAPTIONS_ZH)
