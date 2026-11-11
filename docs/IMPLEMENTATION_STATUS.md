# Implementation Status

**更新：2026-11-10**（原 Round-1 快照已被取代；本文件现维护当前能力地图）。

## 当前能力（按数据流）

| 环节 | 状态 | 关键模块 / 位置 |
|---|---|---|
| 数据源接入（ERA5/METAR/Pangu 审计） | ✅ | `src/offgrid_da/data/{metar,observations,qc}.py`；DATA_AUDIT.md |
| 预报模型（ArchesWeather-M 6h，320k 步含 AR 微调） | ✅ | `experiments/archesweather-reproduction/`；F1–F7 |
| 预报冻结契约（sha256 钉死） | ✅ | `configs/forecast/archesweather_6h_contract.json`、FORECAST_INTERFACE.md |
| 82 通道背景生成 + 归一化 | ✅ | `scripts/generate_full_state_backgrounds.py`、`outputs/bg_t2m_*` |
| 五变量 METAR 观测接入 | ✅ | `src/offgrid_da/data/metar_multivar.py`（single_da_06） |
| 单周期神经 DA（dim128 交叉注意力，82 进 82 出，误差配权损失） | ✅ | `scripts/{run_error_balanced_pipeline,train_error_balanced}.py`（single_da_09） |
| Cycling DA（t2m 线 + full82 线，free/NC/DA 对照） | ✅ | `outputs/cycling_*`；24×5d、24×15d |
| 观测诊断（创新量分解 + 合成观测四臂对照） | ✅ | `outputs/obs_diagnostics_2021`、`outputs/step2_*`（2026-11-08~09） |
| 合成观测 DA 匹配训练 | ✅ | `scripts/train_step2_synthetic.py` |
| 2023 独立确认 | ✅ DA −1.3% vs free（2.280/2.282 vs 2.311，双种子/留出站/区域一致） | `outputs/cycling_t2m_06_confirm_2023` |
| 观测替换诊断（零创新/打乱/注意力关）+ 空间分层 | ✅ 收益全走注意力、位置净贡献 0、集中 <300km 桶 | `outputs/step3_*`（方案 §五/§六） |
| 脉冲响应探针 | ✅ 稠密区 400km / 稀疏区 1200km 足迹 | `outputs/step3_impulse` |
| 延长训练预算检查（§四）| ✅ 双臂收回缺口 5%/9.5%，步数非主因 | `outputs/single_da_10_budget_extension` |
| 噪声剂量-响应（§八扩展①）| ✅ k≈0.95 闭合两臂差距，H2 机制闭合 | `outputs/step4_noise_response` |
| 预报模型升级（AIFS 质量上限线） | ⬜ 调研完成待实施 | EXPERIMENT_REGISTRY.md 附录 3 |

## 实验线索引

- 实验目录索引（含命名规范、权重依赖）：`EXPERIMENTS.md`
- 动机/结论叙事（全部实验）：`EXPERIMENT_REGISTRY.md`
- Round-1 原始日志（2026-10-26/27，历史）：`EXPERIMENT_LOG.md`

## 测试

- Round 1：80/80 通过（数据管线、指标、归一化、无泄漏）。
- 各实验自带验收（frozen 指纹、配对初始化、无观测恒等、观测敏感性检查），
  见各实验目录 `acceptance.json` / `status.json`。

## 核心数字速查

| 量 | 数值 | 出处 |
|---|---:|---|
| 2021 背景基线 T2M RMSE | 0.441409 K | single_da_01 |
| 真实观测 DA（seed42/43）| 0.43494/0.43516 K | single_da_09 |
| 合成观测 DA | 0.42928/0.42969 K | step2_synthetic_da |
| 合成+局地基线（理想天花板）| 0.41720 K | step2_local_baseline |
| 创新量噪声占比 | 94.8% | obs_diagnostics_2021 |
| 零创新观测（仅位置）| ≈背景（位置净贡献 0）| step3_obs_content |
| 收益空间分布 | <300km 桶 −3.1%/−5.6%（占 25.5% 格点）| step3_spatial |
| 延长训练收回缺口 | real 5% / syn 9.5%（步数非主因）| single_da_10 |
| 噪声剂量闭合点 | k≈0.95 时合成臂=真实臂水平 | step4_noise_response |
| 2023 确认 cycling | DA 2.280/2.282 vs free 2.311（−1.3%）| cycling_t2m_06 |
