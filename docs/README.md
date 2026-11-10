# docs 索引

更新：2026-11-10。命名规范：**实验文档与其 outputs 目录同名**（`single_da_NN_*` /
`cycling_NN_*` 系列，2026-11-03 起）；**主题文档 UPPER_SNAKE**；历史对照映射见
`renames.json`（outputs 目录）与各文档头部的历史口径注记。

## 入口（按需取用）

| 想知道… | 看这里 |
|---|---|
| 做了哪些实验、为什么做、结论是什么 | `EXPERIMENT_REGISTRY.md`（叙事总册）|
| 实验目录/命名/权重依赖速查 | `EXPERIMENTS.md`（表格索引）|
| 当前实现到哪一步了 | `IMPLEMENTATION_STATUS.md`（能力地图 + 核心数字）|
| 已知的坑 | `KNOWN_ISSUES.md`（含观测 ID 映射、MSLP 长尾等新条目）|

## 主题文档

| 文档 | 内容 | 状态 |
|---|---|---|
| `DATA_AUDIT.md` | 三数据源结构/网格/QC 审计（2026-10-26）| 有效（含历史口径注记）|
| `DATA_SPEC.md` | 样本管线规范、网格与指标约定 | 部分历史（有效部分已标注）|
| `VARIABLE_MAPPING.md` | 变量/网格映射 | 有效 |
| `BASELINE_PROTOCOL.md` | Baseline 0 定义与指标口径 | 已更新（背景=冻结 ArchesWeather）|
| `FORECAST_INTERFACE.md` | 预报契约接口（DA 管线消费预报的方式）| 有效 |

## 实验文档（与 outputs 目录一一对应）

| 文档 | 实验 |
|---|---|
| `single_da_00_model_check_crossattn.md` | 交叉注意力工程验证 |
| `single_da_01_preflight_smoke.md` | 预检 + Baseline 0 全年化 |
| `single_da_02_t2m_dim32_main.md` | T2M 主实验（dim32）|
| `single_da_04_t2m_controls_ablation_oi.md` | NC 对照/消融/首个 OI |
| `single_da_05_input82_ablation.md` | 82 通道输入消融 |
| `single_da_06_obs_metar5_prep.md` | 五变量 METAR 接入 |
| `single_da_09_full82_loss_balanced.md` | 误差配权世代（当前 DA 主线）|
| `cycling_full82_01_small_superseded.md` | 全状态循环·小（被取代）|
| `cycling_full82_02_large.md` | 全状态循环·大（当前多变量循环主结果）|

> single_da_03/07/08、cycling_t2m_01–05 与观测诊断系列无独立文档，
> 结论收录于 `EXPERIMENT_REGISTRY.md` 与各目录内 report/summary。

## 历史文档

- `EXPERIMENT_LOG.md` — Round 1 原始日志（2026-10-26/27），仅作存档。
- `renames.json` — outputs 目录新旧名映射（2026-11-03）。
