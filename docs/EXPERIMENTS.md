# Experiment Index

命名规范(2026-11-03 起生效):`<系列>_<更新范围>_<序号>_<目的>`。

- **系列**:`single_da`(单周期同化,每时刻独立从 ERA5 生成背景)/ `cycling`(循环同化,分析回传)/ `report`(总结报告)/ `bg`(背景场数据)。
- **更新范围**:`t2m` = 仅更新 T2M;`full82` = 更新全部 82 通道。指**更新**了什么,不是输入了什么(完整输入 ≠ 完整更新)。
- **序号**:实验在各自序列中的时间顺序,`ls` 即按时间线排列。

旧名 → 新名的机器可读映射见 `docs/renames.json`。**注意:各实验目录内部的历史
JSON(protocol/frozen/status)仍嵌有改名前的绝对路径,反查请用该映射表。**
配置文件名未改:`configs/da/single_cycle_v1.json` ↔ `single_da_02`,
`configs/da/full_state_input_v2.json` ↔ `single_da_05`。

## Single DA 系列

| 目录(新名) | 旧名 | 内容 | 状态 | 文档 |
|---|---|---|---|---|
| `single_da_00_model_check_crossattn` | cross_attention_validation | 交叉注意力组件工程验证(entry/layered 注入、padding、无观测恒等) | 完成 | single_da_00_model_check_crossattn.md |
| `single_da_01_preflight_smoke` | single_cycle_preflight | 训练链路 smoke、断点续训、Baseline 0 验收 | 完成,被 02 取代 | single_da_01_preflight_smoke.md |
| `single_da_02_t2m_dim32_main` | single_cycle_v1 | T2M 主实验:DA vs Neural Correction,dim32,2018–2020 训练 / 2021 验证 | 完成 | single_da_02_t2m_dim32_main.md |
| `single_da_03_t2m_test_2022` | single_cycle_test_2022 | 02 的选定模型在 2022 全年独立测试 | 完成 | — |
| `single_da_04_t2m_controls_ablation_oi` | single_cycle_controls_v2 | NC 对照(seed42/43/44)、观测屏蔽/打乱消融、局地 OI 基线;NC 权重被 t2m 系 Cycling 引用 | 完成 | single_da_04_t2m_controls_ablation_oi.md |
| `single_da_05_input82_ablation` | full_state_input_v2 | 82 通道输入 vs T2M 掩码输入(均只更新 T2M,参数匹配);DA 权重被 t2m 系 Cycling 引用 | 完成 | single_da_05_input82_ablation.md |
| `single_da_06_obs_metar5_prep` | metar_multivar_v1 | 五变量 METAR 观测接入(T/U/V/Td/MSLP,数据层,未训练) | 完成 | single_da_06_obs_metar5_prep.md |
| `single_da_07_full82_small_stopped` | full_update_v1 | 82 通道全状态更新训练·小模型(dim32),因模型放大中止 | 中止 | — |
| `single_da_08_full82_large` | full_update_large_v1 | 82 通道全状态更新训练·大模型(dim128/depth4/heads8,seed42/43);权重被 cycling_full82_02 冻结引用 | 完成 | — |
| `single_da_09_full82_loss_balanced` | error_balanced_v1 | 误差配权损失(10 组各占 0.1)重训 DA/NC + 旧冻结模型消融诊断 | 完成 | single_da_09_full82_loss_balanced.md |
| `report_single_da_20261102` | single_da_report_20261102 | Single DA 系列总结报告(图 + 证据包) | 完成 | 报告在目录内 |

## Cycling DA 系列

### t2m 线(仅更新 T2M;权重来自 single_da_04/05)

| 目录(新名) | 旧名 | 内容 | 状态 |
|---|---|---|---|
| `cycling_t2m_01_pilot_4win_5d` | cycling_pilot_v1 | 四季 4×5 天试点(free/NC/DA/masked_DA);目录内含《Cycling_DA结果分析.md》 | 完成,结论被 03/04 扩展 |
| `cycling_t2m_02_alpha_scaling` | cycling_scaling_v1 | 增量 α=0/0.25/0.5/1 缩放:T2M 收益 vs 其他 81 通道退化权衡 | 完成 |
| `cycling_t2m_03_24win_raw` | cycling_24windows_v1 | 2021 年 24×5 天原始循环记录 | 完成 |
| `cycling_t2m_04_24win_eval` | cycling_evaluation_v1 | 24 窗口扩展评估(区域/留出站点/月份分组) | 完成 |
| `cycling_t2m_05_15day` | cycling_15days_v1 | 24×15 天(60 周期)长循环,延伸至 2022-01 | 完成 |
| `cycling_t2m_06_confirm_2023` | cycling_confirmation_2023_v1 | 2023 年 24 窗口×60 周期确认实验;数据=ARCO 重网格分片(与 WB2 口径验证至噪声级);DA 优于 free/NC(双种子、全球/留出站/区域一致) | 完成 | `evaluation_report.md` 在目录内 |

### full82 线(82 通道全状态更新;权重来自 single_da_08)

| 目录(新名) | 旧名 | 内容 | 状态 |
|---|---|---|---|
| `cycling_full82_01_small_superseded` | full_update_cycling_v1 | 全状态循环·小模型 | 被大模型取代 |
| `cycling_full82_02_large` | full_update_large_cycling_v1 | 全状态循环·大模型,24×5 天,十分组平衡 RMSE 主指标;当前多变量循环主结果 | 完成 | cycling_full82_02_large.md |

## 观测诊断系列（2026-11-08 起，合成观测方案）

目的：区分"网络不会用观测"(H1)与"观测与参考场不一致限制收益"(H2)。
完整动机/结论见 `EXPERIMENT_REGISTRY.md` 阶段 4。

| 目录 | 内容 | 状态 |
|---|---|---|
| `obs_diagnostics_2021` | 第一步:创新量分解 d/r/相关性,分月/纬度/高程/时间偏移分组,日块 bootstrap | 完成 |
| `step2_local_baseline` | 第二步:局地高斯创新订正基线(真实/合成双臂,训练集调优半径×增益) | 完成 |
| `step2_synthetic_da` | 第二步:合成观测 DA 匹配训练(seed42/43)+ 四臂对照 `four_arm_summary.md` | 完成 |
| `step3_obs_content` | 第三步:观测内容替换诊断(零创新/打乱/注意力关,1460 周期)+ 归因 | 完成 |
| `step3_impulse` | 第三步:单站脉冲响应探针(稠密区 400km vs 稀疏区 1200km 足迹) | 完成 |
| `step3_spatial` | 第三步:空间分层评估(距观测距离分桶 + 留出站分层,收益集中 <300km 桶) | 完成 |

结论速览：H1、H2 同时成立且 H1 为主——完美观测下神经 DA(−2.7%)只有局地基线(−5.5%)的一半；
真实观测下 DA(−1.45%)>局地基线(~0%),跨变量利用是网络优势面。

## 背景场数据

| 目录(新名) | 旧名 | 内容 |
|---|---|---|
| `bg_t2m_train_2018_2020` | arches_backgrounds_train_2018_2020 | ArchesWeather T2M 背景,训练期 |
| `bg_t2m_val_2021` | arches_backgrounds_val_2021 | 同上,2021 验证期 |
| `bg_t2m_test_2022` | arches_backgrounds_test_2022 | 同上,2022 测试期 |

82 通道完整状态背景(5844 个状态,45G)在 `single_da_05_input82_ablation/backgrounds/`。
归一化统计在 `outputs/normalization/`(被全部数据集引用,勿动)。

## 权重依赖(清理前必查)

- t2m 系 Cycling 的 DA 权重 ← `single_da_05_input82_ablation/seed42,43/full82/best.pt`
- t2m 系 Cycling 的 NC 权重 ← `single_da_04_t2m_controls_ablation_oi/seed42,43/full82/formal/best.pt`
- `cycling_full82_02_large` ← `single_da_08_full82_large/seed42,43/{da,neural_correction}/formal/{best.pt,config.json,result.json}`(frozen.json 哈希校验)
- `single_da_03_t2m_test_2022` ← `single_da_02_t2m_dim32_main` 等的 best.pt

## 2023 确认实验重启注记

改名时(2026-11-03)kill 了等待进程 `wait_and_run_confirmation_2023.py`(原 PID 528203,
最后状态 `waiting_for_2023_data`,缺 1224 个时次)。脚本路径已更新为新目录名。
**2026-11-10/11 已完成**:2023 ERA5 分片由 ARCO v3 经一阶保守重网格生成(WB2 桶
无 2023 全年;2022 口径验证至数值噪声级,见 `experiments/.../data/era5_240/prepare_2023_report.json`),
协议重冻结(路径更新+5 脚本新哈希+分片哈希)后由等待脚本自动执行。
执行注记:geoarches/.venv 需补装 pyarrow;双种子并发下载 METAR 有缓存竞态,
seed42 因竞态失败后单进程补跑成功。结果:`evaluation_report.md`——
DA 2.280/2.282 K 优于 free 2.311 与 NC 2.296/2.303(−1.3% vs free),留出站、
北半球温带/欧洲/北美区域一致改善,热带微小(观测少),预声明协议无重选。
