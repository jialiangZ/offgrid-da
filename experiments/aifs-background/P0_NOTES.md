# P0 环境与冒烟记录（2026-11-11，完成）

## 环境
- venv: `experiments/aifs-background/.venv`（uv sync --extra inference，python 3.12）
- 关键版本（官方 uv.lock 钉死）：torch 2.7.0+cu126 / anemoi-inference 0.8.3 /
  anemoi-models 0.9.3 / anemoi-graphs 0.6.4 / anemoi-transform 0.1.16.post2 /
  anemoi-plugins-ecmwf-inference 0.2.1 / flash_attn 2.8.3（预编译 wheel，零编译）
- ckpt: `aifs-single-mse-2.0.ckpt`（948.3 MiB，ecmwf/aifs-single-2.0，CC BY 4.0）

## 本地热修（必须记录；uv sync 会覆盖）
anemoi-plugins-ecmwf-inference 0.2.1 与 anemoi-transform 0.1.16.post2 的接口 bug：
`InferenceOrography.optional_inputs` 用键名 orography/geopotential，基类访问 self.z。
修复：`.../anemoi/plugins/ecmwf/inference/opendata/opendata.py:308` 的
OrographyProcessor 调用加 `z="z"` 参数。
（备选未采纳：升 transform 0.4.2 会破坏 inference 0.8.3 的模块路径。）

## 冒烟结果（CUDA_VISIBLE_DEVICES=1, RTX 4090）
```
anemoi-inference run inference.yaml lead_time=6h
```
- Open Data 初值链路全通（起报 2026-11-10 06Z）
- 输入张量 (2, 106, 542080)：2 时刻 × 106 通道 × **N320 高斯格点（542,080 ≈ 31km）**
  —— 模型内部跑 N320，输出插值 0.25° 规则网格
- **单步 6h 推理 0.9 秒**（P2 成本模型：5844 时次 ≈ 88 分钟纯推理，完全可行）
- 输出 output.grib：215 场 / 124 通道；82ch 所需（u/v/t/q/z/w×13 层 + 2t/10u/10v/msl）
  全部在内
- ckpt 加载 9 秒；图构建正常；flash-attn 生效

## 对 P1/P2 的输入
- 成本：纯推理可忽略，瓶颈将是 ARCO 初值构造与读取（~600GB 分片缓存）
- P1 注意：输出为 GRIB（earthkit 读取）→ 重网格管线入口需支持 GRIB 读数
  （或配置 output 为其他格式）；输入侧走 ARCO 自定义 initial state，
  opendata 仅冒烟用

## Wave 谱带填充预实验（2026-11-11/12，P1 门控）

ARCO v3 缺 6 个半周期波高带通道（h1012-h2530）。单次起报（2026-11-10 00Z）
对照实验：完整初值 vs h* 置零，6h 预报在 82ch 上的差：

| 通道 | 置零噪声 | vs ArchesWeather 6h 误差 | vs AIFS 预期改善量 |
|---|---:|---:|---|
| 2t | 0.072 K | 16% | **~100%（吞掉全部改善）** |
| 10u | 0.119 m/s | 32% | ~150% |
| 10v | 0.075 m/s | 20% | ~100% |
| msl | 5.4 Pa | 21% | ~110% |
| t_850 | 0.051 K | 17% | 临界 |

**结论：零/气候态填充不可接受**（GNN 全局注意力把海面扰动传入大气通道，
远超"wave 只影响海浪"的直觉）。执行方案 v1.0 预案：**切换 AIFS-single-1.1**
（v2 之前无 wave 组件，输入清单预期全在 ARCO 覆盖内，无填充近似、契约干净；
1.1 vs 2.0 的质量差距 << AIFS vs ArchesWeather 差距，不改变实验科学性）。
产物：wave_fill_probe.{json,log,py}。

## P1 工程记录（2026-11-12）

1. **管线打通**：prepare_aifs_inputs.py（ARCO→90ch npy 分片，2021 抽取中）+
   run_aifs_background.py（npy→N320→AIFS-1.1 6h→N320→1.5°→npz，物理值，
   与 single_da_05 背景格式逐位兼容）。
2. **关键坑与解**：
   - anemoi-models 0.9.3 无法运行 1.1 的 pickle（层结构重构 lin_qkv→lin_q 等）
     → 独立 venv（anemoi-inference 0.6.3 + models 0.5.0 + torch 2.4.1 +
     PyG 2.4.0 + flash-attn 2.6.3，对齐官方 notebook）
   - OOM 24G：0.5.0 推理默认不分块 → `ANEMOI_INFERENCE_NUM_CHUNKS=32`
   - 日期语义：输入 [t-6h,t] 的 6h 预报 valid 于 t+6h（x_b(t+6h)）——
     首版曾错位 6h（验证 RMSE 2.4K 暴露）
   - 纬度约定：背景 npz 北起（90→-90）vs era5_240 分片南起——验证脚本必须翻转
   - 通道映射：数值反推法确认 npz level 行序 [z,u,v,t,q,w]（4 个锚点变量锁定）
3. **验证**（2021-01-02T00 单时次，8 通道）：量级全部合理（2t 0.505 /
   msl 29.5 / t_850 0.324 / z_500 19.3），与 ArchesWeather 同量级。
   全年 P2 验收待抽取完成后执行。
4. **初步信号**：AIFS-1.1 对 ERA5 参考口径不优于 ArchesWeather——
   预注册判据的"停查/部分兑现"分支可能触发（全年数字为准）。

## P2 中期裁决 + fp16 存储研究（2026-11-12）

### P2 中期验收（STOP 分支触发）
- 覆盖：2021-01-01 ~ 05-11，522 时次（全年 36%），评估脚本 eval_p2.py
- **T2M RMSE mean 0.5318 K**（median 0.5346，p10 0.5013，p90 0.5571），bias +0.028 K
- 锚点 ArchesWeather 2021 = 0.441409；判据门 0.43 → **STOP-AND-RECORD**
- 翻盘算术：剩余 938 时次需平均 ≤0.373 K 才能全年 <0.43——比已观测分布的
  最好十分位（0.50）还低 30%，季节效应不可能弥补
- 次级通道一致偏差：10u 0.435（vs 0.377）、msl 27.5 Pa（vs 25.9）
- 决定（用户已委托）：**提前终止 2021 抽取**（day~130/365 处 kill，省 ~19h + ~340 GiB）。
  已抽 Jan–May npy 与 522 个背景 npz 保留为负结果证据。
- 科学结论（预注册负结果分支）：更强业务 FM ≠ 更好 DA 背景——
  AIFS-1.1 对 ERA5 参考口径 6h 误差比 ArchesWeather-250k 差 ~20%，
  系统差（训练目标/初值路径）抵消模型优势。AIFS 线收官，DA 转观测侧。

### fp16 输入存储研究（control_precision.py）
- 裸 fp16：9/90 通道溢出（msl/sp ~105k、z_50~z_400 至 203k > 65504）——不可用
- 缩放 fp16 输入侧噪声：2t 0.049K / msl 16Pa / z_500 9 m²s⁻²
- 端到端（经 regrid+AIFS 6h 平滑后，C vs B）：
  2t 0.024K（门控数字 +0.0005，可忽略）/ msl 6.7Pa(+2.9%) / z_500 4.0(+2.1%)
- 执行精度无关性：bf16-autocast vs fp32 forward 输出差 ≤0.003 K（A vs B）
- zstd-7 无损压缩 1.79×（199 MiB/时次）≈ fp16 的省幅但零精度影响
- 教训：test_fp16_inputs.py 的 (240,) vs (121,240) 广播 bug 制造了"21K 崩坏"假警报
  ——比较两场前必须断言 shape 相同。该脚本已删除，以 control_precision.py 为准。

## ⚠️ P2 判决反转：降尺度约定伪影（2026-11-12，用户假设验证）

用户质疑"多走的重网格是否是 deficit 主因"——分解实验（decompose_regrid.py，24 时次）证实，
且根因比预想更深：**era5_240 是面积平均约定的 0.25° ERA5 产品，而我们的输出降尺度
（N320→1.5 线性点取样）用了错误的空间滤波器对照它**：

| 2t (K) | 数值 | 含义 |
|---|---:|---|
| 真值走线性链 vs era5_240 | 0.435 | **完美预报也要白付的接口代价** |
| 真值跨6点取样 vs era5_240 | 0.532 | 点取样约定失配更重 |
| 真值 7×7 盒平均 vs era5_240 | **0.08** | 来源假设证实（era5_240≈boxavg(ARCO)）|
| AIFS 线性降尺度 vs era5_240 | 0.524 | 旧管线（=中期 STOP 的数字来源）|
| **AIFS 盒平均降尺度 vs era5_240** | **0.341** | **强过 ArchesWeather 锚点 0.4414 达 23%** |
| AIFS@0.25° vs ARCO 真值 | 0.431 | 原生分辨率真实误差（小尺度更难）|

- **中期 STOP 判决作废**：0.532 主要是约定失配伪影；修正后落入预注册"强通过"区（<0.38）
- z500 等平滑场不受约定影响（线性 vs 盒平均差 <0.3）——失配集中在粗糙场（2t/msl/t850）
- 已修正 run_aifs_background.py 输出路径为 N320→0.25→box7→1.5；
  旧 522 背景存档于 bg_aifs_2021_linearconv；522 重生成 + 全年续抽已重启
- v3 教训：0.25° 全链路迁移**不必要**——正确降尺度下 AIFS 在 1.5° 上已胜出，
  且原生 0.25° 误差（0.43）反而高于平滑后（0.34），高分辨率 DA 将面对更高背景误差 + ~36× 算力

## 压缩方案基准（bench_compress.py，2026-11-12，用户要求先小批量测试）

样本：3 个输入分片（0.348 GiB/片，1/2/4 月）+ 1 个背景 npz。结果高度跨样本一致（±0.002）。

| codec | 比率 | 编码 s/片 | 解码 GiB/s |
|---|---:|---:|---:|
| lz4 | 0.885 | 0.6 | 1.64 |
| zstd-1 / 3 / 7 / 12 | 0.789 / 0.621 / **0.558** / 0.546 | 0.7 / 2.1 / 5.2 / 10.8 | ~0.65 |
| blosc-zstd7（含 byteshuf）| 0.588 | 2.2 | **4.5** |
| blosc-zstd9-256M | **0.491** | 95 | 0.48 |
| zstd-19（±long27）| 0.494 | 90 | 0.36 |
| xz-6 | 0.428 | 18 (8T) | 0.05 ❌管线不可用 |

- 字节/位混洗对输入分片**无益**（空间平滑性主导；混洗破坏局部性），对背景 npz 反而有益 −5%（bf16 尾数噪声）但不值得换容器
- **决策**：输入分片 zstd-7（.npy.zst，CLI↔numcodecs 位级互操作已验证）；
  背景维持 zlib npz（实测 7.9 MiB 优于 zstd 容器 8.06 MiB——中途试过的 .npz.zst 已回退）
- 存量 553 片 + 常量已转换：**191 GiB → 108 GiB（−43%）**；全年投影 ~284 GiB（原 509）
- 解码 0.65 GiB/s → 生成循环 +~1.1s/时次（4.4→5.5s），547 片约 50 分钟，可接受
- 抽取端 zstd-7 内联压缩 +5s/97s（~5% 吞吐损失），无需两段式

## 修正版 P2 验收：STRONG PASS 确认（552 时次，2026-11-12）

- **T2M RMSE 0.3513 K**（median 0.3511，p10 0.3219，p90 0.3839，bias +0.029）
  vs 锚点 0.441409 → **−20.4%**；预注册强通过门 <0.38 ✓
- 次级通道一致领先：10u 0.302（vs 0.377，−20%）、msl 20.5 Pa（vs 25.9，−21%）
- 月度平稳：1-5月 0.361/0.364/0.350/0.342/0.332，夏季略降——全年翻门需剩余
  908 时次平均 >0.396（观测 p90 仅 0.384），几乎不可能
- 数据：data/bg_aifs_2021（盒平均约定，552 npz）；
  data/bg_aifs_2021_linearconv 保留为约定失配对照证据
- 生成实测 6.1s/时次（zstd 解码 +1.1s + 盒平均输出路径 +0.6s vs 旧 4.4s）
- 下一步：全年抽取完成（~26h）→ 补生成 May-Dec → 全年 P2 终章 → P3 解锁
