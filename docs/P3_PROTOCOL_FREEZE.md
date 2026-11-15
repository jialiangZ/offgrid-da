# P3 协议冻结：single_da_12_aifs_background（2026-11-15）

按 2026-11-15 条件放行指令冻结。**formal 训练前本文件不改动；任何偏离须另立版本并记录。**

## 实验问题

背景质量（ArchesWeather-250k → AIFS-single-1.1，T2M −2.9%、风/气压 −15~−17%）
对 DA 收益的影响——与 S4 噪声剂量曲线对偶的"背景质量剂量"。

## 匹配原则（唯一差异 = 背景来源）

| 维度 | 冻结值 |
|---|---|
| 结构 | da_09 原版：`CrossAttentionAnalysisModel(channels=82, dim=128, heads=8, depth=4, injection=layered, radius_km=1000, max_neighbors=80, query_chunk=256)`。**不引入 da_11 的 radius/neighbors 调度**（隔离背景变量；da_11 交叉留待后续）|
| 种子 | 42 / 43（与 da_09 同 seed 语义的配对初始化）|
| 臂 | `da`（观测 DA）+ `neural_correction`（NC，无观测对照）× 2 种子 = **4 个 formal** |
| 预算 | epochs 10（43,840 步）、batch 1、lr 3e-4、warmup 200、patience 4（与 da_09 逐字段相同）|
| 损失尺度 | **沿用 `single_da_09/loss_statistics.json`（sha 钉死 `bb22d606…`）**——balanced loss 变量权重与 physical_std 不随背景重拟合；这是跨实验可比的口径选择，牺牲 AIFS 背景下的最优加权 |
| 数据 | 同一 `da_sample_cache` / 同一 METAR 观测 token / 同一 val 2021 选择集（1460 周期，含 2021-01-01T00 头背景，由 2020-12-31T18 尾部输入生成）|
| 背景 | AIFS-single-1.1（ckpt sha `6acedcd2…`），N320→0.25°→7×7 盒平均→1.5°；manifest sha 入 preparation |
| 输入归一化 | 实验内重算（`prepare_full_state_input.py --experiment`，train-only 82ch 统计）；legacy T2M 守卫沿用 v2（见 NORMS_V2_REPIN_NOTE）|
| 选择规则 | best full-2021-validation，与 da_09 相同（无 test 接触）|

## 共同评价指标（全部 Σcosφ·M 分母——2026-11-15 冻结的统一口径）

1. **主指标**：T2M 全局 cos-lat 加权 RMSE（K），val 2021 全年 1460 周期
2. 82ch error-balanced score（da_09 权重）
3. 距离分层 T2M（<300km 特征桶）+ 站点（训练/留出）
4. u10 / v10 / mslp RMSE
5. **双报告轴**：
   - 整体收益 = DA − 背景（对照 da_09：T2M −0.0065 K / −1.45%）
   - 观测边际 = DA − NC（对照 da_09：T2M −0.0034 K / ≈−0.8pp）
6. 评估路径：`eval_independent_2022.py` 同族（锚点自检先行：先在 2021 复现
   da_09 0.4122 / 背景 0.4207 ±0.001 再采信 AIFS 臂数字）+ step5 分层口径

## 小规模检查（formal 前置，按指令⑤）

1. `assemble_aifs_backgrounds.py`：5844 背景 complete、通道契约逐位一致
2. `prepare_full_state_input.py`：normalization 守卫通过（train_cycles=4384）
3. 4 臂 × trial 256 步：loss 下降、无 NaN、val 载入、配对初始化 sha 一致
4. 单批前向 vs 手工归一化一致性抽查
5. **trial 结果交用户过目后放行 formal**

## 预注册判读

- DA 整体收益收窄（< −0.004 K vs 背景）→ 背景改善挤压观测边际价值（SNR 挤压方向）
- 持平 / 放大 → 桥接价值上升（与 da_11 交叉定结构方向）
- NC 自身变化亦有信息量：背景变好 → 无观测订正的可压缩空间缩小多少
- 背景×噪声二维收缩曲面为后续旗舰交付，**本轮不做**
- 2022 独立时段确认复用 step6 协议（formal 后、收官前）
