# Global migratory-bird–wind-farm ecology

本仓库专门研究全球候鸟迁徙与风场的空间重合、迁徙方向、鸟类行为和生态友好朝向。它与工程仓库 `wind-direction-to-electricity-transition` 严格分离。

## Repository boundary

本仓库负责四件事。第一，建立全球风场与迁徙鸟的空间和季节重合母体。第二，从雷达、GPS或其他迁徙资料提取鸟类方向签名。第三，估计阵列方向对应的几何暴露、进入、避让、碰撞和屏障风险。第四，在需要时读取工程仓库已经计算完成的 AEP 方向曲线，形成能源—生态交换率。

本仓库不下载ERA5用于发电回测，不运行FLORIS，不计算尾流，也不重新搜索经济最优方向。所有发电计算继续保留在 `wind-direction-to-electricity-transition`。两个仓库只通过标准CSV接口交换结果。

```text
wind-direction-to-electricity-transition
    └─ exports/aep_orientation_curve.csv
                     │ read-only
                     ▼
global-migratory-bird-windfarm-ecology
    ├─ stage0_screening   全球风场—迁徙受体重合与数据就绪度
    ├─ bird_analysis      方向签名、通量、高度与行为
    ├─ ecology_analysis   暴露曲线与生态最优角
    └─ integration        读取外部AEP曲线并计算交换率
```

## Standard interfaces

鸟类分析的核心产物是 `bird_direction_signature.csv`。每行表示一个风场、物种或功能群、季节和方向证据，至少包含 `farm_id`、`receptor_id`、`season`、`direction_deg`、`concentration`、`flux`、`rotor_height_fraction` 与 `evidence_level`。

工程仓库的唯一必需输入是 `aep_orientation_curve.csv`。每行包含 `farm_id`、轴向角 `theta_deg` 和 `aep_gwh`。本仓库把它视为只读外部产物，不包含任何AEP计算实现。

生态分析输出 `ecology_orientation_curve.csv`，包含 `farm_id`、`theta_deg` 和 `risk_score`。集成层按 `farm_id + theta_deg` 对齐两条曲线，在给定AEP预算下选择生态友好方向。

## Quick start

```powershell
python -m pip install -e ".[dev]"
pytest
python -m bird_wind_ecology --help
```

使用示例数据计算1% AEP预算下的生态友好方向：

```powershell
python -m bird_wind_ecology tradeoff `
  --aep examples/aep_orientation_curve.csv `
  --ecology examples/ecology_orientation_curve.csv `
  --budget 0.01
```

## Project map

- `src/bird_wind_ecology/stage0_screening/`：全球研究母体与U0–U4就绪度分级。
- `src/bird_wind_ecology/bird_analysis/`：鸟类方向签名计算。
- `src/bird_wind_ecology/ecology_analysis/`：方向性生态暴露曲线。
- `src/bird_wind_ecology/integration/`：只读取外部AEP曲线并计算让步价。
- `schemas/`：跨仓库数据契约。
- `knowledge-base/`：研究方案、数据卡和文献证据。
- `tests/`：轴向角、方向签名、样本分级和交换率测试。

## Scientific guardrails

几何风险函数只是可替换的初筛模型。未经雷达、GPS或死亡数据验证，输出只能称为方向性暴露代理，不能称为真实死亡下降或生物多样性提升。全球生态重合图和数据就绪度图必须分别报告，因为没有公开追踪数据不等于没有迁徙风险。

