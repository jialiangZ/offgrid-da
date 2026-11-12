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
