---
title: Mercker et al 2026-红鸢微观与中观避让率
year: 2026
authors: Mercker; Škrábal; Blew; Liesenjohann; Raab; et al.
journal: Scientific Reports
doi: "10.1038/s41598-026-45894-3"
url: "https://doi.org/10.1038/s41598-026-45894-3"
status: verified
type: source
evidence_role: avoidance-parameterization
evidence_level: B
full_text: open-access-full-text
tags:
  - ecological-concession-price/source
  - red-kite
  - gps-tracking
  - collision-risk-model
---

# Mercker et al. (2026)

## 一句话结论

高分辨率 GPS、逐机运行数据和模拟误差校正显示，红鸢对旋翼扫掠空间的微观避让约为 80%，对整台风机的中观避让约为 87%–94%，组合避让概率约为 98%，且避让随天气和既往风机经验而变化。

## 数据与方法

- 将鸟类 GPS 点与具体风机的转速、转子朝向、轮毂高度、叶轮直径、风速和风向进行时空匹配。
- 区分旋翼风险空间的微观避让和对整台风机的中观避让。
- 通过模拟校正 GPS 空间误差对小尺度避让率的偏差。

## 可支持的论断

- 碰撞暴露不能只由几何截面决定，必须乘以尺度与物种特异的避让概率。
- 风和个体经验会改变避让，因此 $E(\theta)$ 应尽可能扩展为条件风险 $E(\theta\mid wind,season,species)$。
- 转子瞬时朝向与机排轴方向是不同变量，模型中必须明确区分。

## 不能支持的论断

- 研究没有检验规则阵列整体旋转的效果。
- 红鸢避让率不能直接套用于夜迁鸣禽、海鸟或其他猛禽。
- 高避让率不等于零种群影响，也不排除屏障效应和栖息地回避。

## 对本研究的用法

用于把简单暴露函数修正为：

$$
R_s(\theta)=E_s(\theta)P(h_s\in H_{rotor})[1-P_{avoid,s}]
$$

其中避让率应按物种、天气和飞行经验设定敏感性范围，而不是采用单一常数。

## 原文链接

[Nature Portfolio 开放全文](https://www.nature.com/articles/s41598-026-45894-3)

