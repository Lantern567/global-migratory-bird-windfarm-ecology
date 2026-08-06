# ERA5风向作为候鸟飞行方向代理：方法学与不确定性分析

日期：2026-07-30
用途：为东亚88个（及其他共134个）缺失雷达/GPS方向数据的风场提供U2级别方向估计

---

## 1. 问题

构建生态暴露模型B(theta)需要场级鸟类飞行方向分布p(phi)。37个欧洲近海风场有VPTS雷达实测方向数据，但其余134个风场（含中国66个、日本3个、韩国2个等）缺乏任何鸟类方向观测数据。Movebank交互式地图已确认东亚近海无公开GPS追踪轨迹。AVISTEP仅覆盖9个发展中国家，不含中国。

因此需要一种替代方法，在不依赖鸟类观测的条件下，为这些风场提供可用的方向估计。

## 2. 方法：ERA5再分析风向作为飞行方向代理

### 2.1 基本原理

候鸟在高空迁徙时倾向于选择性顺风飞行（selective wind drift），以降低飞行能耗（Liechti 2006, J. Ornithol.; Kemp et al. 2010, J. Avian Biol.）。在长距离跨海迁徙中，这一倾向更强——海上缺乏地形参照物，候鸟更依赖风向导航。

既然候鸟倾向于顺风飞行，那么风场当地的迁徙季主导风向可以作为候鸟飞行方向的估算代理。注意此代理给出的是**方向轴**（axial），因为：(a) 春秋风向可能相反（如东亚冬季北风、夏季南风），对应春北迁秋南迁的候鸟规律；(b) 风场朝向本身也是轴向数据（0-180°）。

### 2.2 数据来源

工程仓库（wind-direction-to-electricity-transition）的task1_wind_metrics.csv包含171场×11年（2014-2024）的逐年ERA5风资源指标。待提取的变数：

- `theta_energy_yearly` — 逐年能量加权风向（已有，171场）
- `theta_freq_hist` — 历史频率加权风向（已有，72场）
- `theta_energy_hist` — 历史能量加权风向（已有，72场）

对缺失`theta_freq_hist`的99场，可从原始ERA5 .nc文件（offshore-task2/data/）重新计算。171场均已有ERA5链接文件（coverage_flag=1），不依赖新数据下载。

### 2.3 从风向到飞行方向：季节性映射

| 迁徙季 | 月份 | ERA5主导风向 → 候鸟飞行方向映射 |
|--------|------|----------------------------------|
| 春季北迁 | 3-5月 | ERA5南风分量 → 推断候鸟向北飞 |
| 秋季南迁 | 9-11月 | ERA5北风分量 → 推断候鸟向南飞 |

具体计算：取迁徙季期间的ERA5逐时风向，计算能量加权圆平均方向及其圆浓度（concentration = 方向一致度）。方向的季节性调转（春秋相反）通过分别计算春秋两季方向来捕捉。

### 2.4 方向不确定性量化

与雷达实测（VPTS）相比，ERA5代理引入以下不确定性源：

| 来源 | 影响的变量 | 估计量级 | 处理方式 |
|------|-----------|---------|---------|
| 候鸟不总是顺风 | direction_deg | ±30° | 圆浓度降低：用concentration参数表达 |
| 海岸地形干扰 | direction_deg | ±15° | 近岸风场加额外偏差 |
| ERA5再分析误差 | direction_deg | ±5° | 可忽略（相对其他源） |
| 迁徙季定义偏差 | season | 窗口±10天 | 敏感性分析：窗口±15天 |
| 候鸟不完全在风机高度 | rotor_height_fraction | ×0.5-1.0 | 使用Bauer et al. (2026)的50%默认值 |

**不确定性传播策略**：不给出单一方向角，而是给出方向分布p(phi) = Von Mises分布，均值=ERA5主导风向，浓度参数kappa = 1/σ²（σ为上述来源的合成标准差≈30-40°）。在暴露曲线计算中使用此分布卷积sin²核。

## 3. 科学依据与文献支持

### 直接支持的文献

- **Kemp MU, Shamoun-Baranes J, Van Gasteren H, Bouten W, Van Loon EE (2010)**. Can wind help explain seasonal differences in avian migration speed? *Journal of Avian Biology* 41: 672-677. — 论证候鸟迁徙速度与顺逆风高度相关，间接支持顺风飞行假设。此文献已在学长研究方案knowledge-base的04-数据源与可获得性.md中引用。

- **Liechti F (2006)**. Birds: blowin' by the wind? *Journal of Ornithology* 147: 202-211. — 综述候鸟对风的利用策略，结论为多数夜间迁徙候鸟倾向于选择性顺风飞行。

- **Bauer S et al. (2026)** *Nature Sustainability* — 研究使用了ERA5风场数据（Method节明确说明风速来自ERA5），虽未直接将风向用作鸟向代理，但风-鸟的耦合是方法学基础。

### 学术风险与应对

| 风险 | 应对 | 在论文中的表述 |
|------|------|---------------|
| 顺风假设在个别物种/地区不成立 | 使用圆浓度kappa表达不确定性而非硬方向 | "ERA5 wind direction is used as a coarse-flyway-level directional proxy with explicit low circular concentration (kappa ~ 2-3), which is categorically distinct from radar-measured field-level direction" |
| 侧风补偿（side-wind drift） | 侧风时鸟可能不完全顺风 | 在Discussion中透明讨论，参考Alerstam (1979)和Klaassen et al. (2011)关于侧风漂移的争议 |
| 海上-陆上差异 | 海上鸟更多依赖风 | 此代理对海上风场（我们的全部171场）优于陆上 |

### 不建议使用的场景（明确排除）

1. 地形复杂的山区风场 — 候鸟利用地形上升气流而非顺风
2. 海峡/ funneling区域 — 候鸟方向受地形约束>风约束
3. 昼间迁徙物种 — 昼间候鸟更多依赖视觉导航和热力上升

## 4. 实现方案

### Step 1: 提取ERA5迁徙季风向

对缺失theta_freq_hist的99场，从原有ERA5 .nc文件提取3-5月（春季）和9-11月（秋季）的逐时轮毂高度风向。使用能量加权（每个风向bin的累计风速三次方加权）计算圆平均方向。

### Step 2: 构建替代方向签名

对每个风场输出：

```
farm_id: int
season: spring | autumn
direction_deg: float (ERA5能量加权风向经季节性映射)
concentration: float (0.15-0.30 range, 远低于雷达0.29-0.75)
flux: 50.0 (placeholder, relative)
rotor_height_fraction: 0.5 (Bauer et al. 2026 default)
evidence_level: 'coarse-flyway'
conservation_weight: 1.0 (lower confidence)
source: 'ERA5_30yr_reanalysis_wind_direction'
```

### Step 3: 验证

对已有雷达数据的37场，用ERA5风向跑同样的暴露曲线，与雷达结果对比——如果两者的最优朝向偏差<30°，则认为ERA5代理可作为初筛工具。如果偏差>60°，不应以ERA5代理结果发表。

## 5. 不确定性声明（论文模板）

> Following Kemp et al. (2010) and Liechti (2006), we use the ERA5 reanalysis wind direction during the migration season (March-May for spring, September-November for autumn) as a coarse directional proxy for bird flight bearing. This proxy assumes facultative downwind flight, which is well-documented in long-distance nocturnal migrants but carries substantial uncertainty (circular standard deviation ~30-40deg). The resulting ecological exposure estimates serve as a coarse-reference directional guidance, distinctly lower in evidence level than radar-measured direction signatures. Farms analyzed with this proxy carry the evidence level tag "coarse-flyway" and are excluded from quantitative scientific claims involving collision risk reduction.

## 6. 结论

ERA5风向作为候鸟飞行方向的替代代理在方法学上有文献依据，但不确定度显著高于雷达实测（圆浓度0.15-0.30 vs 0.29-0.75）。此方法仅能在"粗飞线推测"证据等级下使用，不能等同于雷达或GPS的观测结果。对37个雷达覆盖的风场，应优先使用VPTS方向数据。ERA5代理仅用于134个缺数据的风场，且需在论文中透明报告上述全部假设和局限。

---

参考文献:
- Kemp MU et al. (2010) J. Avian Biol. 41: 672-677
- Liechti F (2006) J. Ornithol. 147: 202-211
- Bauer S et al. (2026) Nat. Sustain., doi:10.1038/s41893-026-01853-4
- Alerstam T (1979) J. Theor. Biol. 79: 341-353
- Klaassen RHG et al. (2011) Proc. R. Soc. B 278: 2114-2121
