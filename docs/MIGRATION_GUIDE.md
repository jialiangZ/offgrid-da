# 迁移指南（2026-11-13）

目标：把仓库迁到新机器，**只带代码 + 文档 + 不可再生的训练产物**；所有可再下载/
再处理的数据（ERA5 参考场、观测缓存、AIFS 输入/背景、venv）全部丢弃，新机上按
本指南重建。本文即"新机数据重备的权威流程"，所有再生命令都验证过。

---

## 1. 迁移策略总览

```
git clone（代码/文档/配置，~几十 MB）
   + rsync 白名单（不可再生资产，~7 GB，见 §2）
   + 新机数据重备（§4，按需逐项，总计 ~650 GB 可再生数据）
```

## 2. 必须手动搬运的资产（rsync 白名单，全部 gitignored）

| 路径 | 大小 | 内容 | 为什么不可再生 |
|---|---:|---|---|
| `outputs/forecast_freeze/archesweather_250k_20261031/` | 1.0 GB | **自含**：ckpt+config+norm stats+manifest（sha256 已验证 267641e1…）| 自训模型，DA 契约锚点 |
| `outputs/single_da_*/`（去掉 backgrounds）| ~2.5 GB | 各实验 best.pt + result.json + 评估报告 | 匹配训练产物（证据链 + 复评能力）|
| `outputs/step*/ outputs/cycling*/ outputs/synth*/` | ~1 GB | 评估报告、分层裁决、EVAL_BUG_NOTE | 实验证据 |
| `outputs/normalization/` | 20 KB | norm_stats_train_2018_2020.json | 训练统计（可重算但必须逐位一致，直接带）|
| `experiments/archesweather-reproduction/stats/` | ~1 MB | era5_norm_stats_1998_2017*.pt | 同上（freeze 已含快照，双保险）|
| `experiments/archesweather-reproduction/swanlogs_archive.tar.gz` | 16 MB | ArchesWeather 训练遥测曲线 | 唯一训练过程记录（16M，tar.gz）|
| `outputs/data_audit/` | 2 MB | Phase 0 审计产物 | 记录 |

**rsync 命令**（旧机执行；排除可再生的重型子目录）：

```bash
rsync -av --info=stats1 \
  --exclude 'single_da_05_input82_ablation/backgrounds/' \
  --exclude 'metar_multivar_cache/' \
  --exclude 'cache/' --exclude 'da_sample_cache/' --exclude 'da_test_sample_cache/' \
  --exclude 'bg_t2m_train_2018_2020/' --exclude 'bg_t2m_val_2021/' --exclude 'bg_t2m_test_2022/' \
  --exclude '*.log' \
  outputs/ NEW:~/Documents/offgrid-da/outputs/
rsync -av experiments/archesweather-reproduction/stats/ \
  NEW:~/Documents/offgrid-da/experiments/archesweather-reproduction/stats/
```

（`git clone` 后其余目录由 git 提供；上述两目录 rsync 补齐。）

## 3. 本机已清理（丢弃）清单

| 路径 | 原大小 | 再生方式 |
|---|---:|---|
| `experiments/archesweather-reproduction/data/era5_240/` | 432 GB | §4.1 |
| `experiments/aifs-background/data/`（aifs_inputs + bg_aifs_2021*）| 181 GB | §4.4 / §4.5 |
| `experiments/archesweather-reproduction/geoarches/`（venv/modelstore/evalstore）| 25 GB | §4.6（仅复训需要）|
| `experiments/aifs-background/{.venv,v1.1/.venv}/` | ~14 GB | §4.7 |
| `experiments/archesweather-reproduction/{data-venv,logs}/` | ~270 MB | §4.6 |
| `experiments/aifs-background/v1.1/aifs-single-mse-1.1.ckpt` | 948 MB | §4.7（HF 重下）|
| `outputs/` 各缓存与背景（见 rsync 排除项）| ~79 GB | §4.2 / §4.3 |

## 4. 新机数据重备（按需逐项，顺序即依赖序）

### 4.0 环境基座
```bash
# Python 3.12 + uv；GPU ≥24 GB（AIFS-1.1 推理需 NUM_CHUNKS=32）；磁盘预算见 §5
cd offgrid-da
uv venv .venv && uv pip install --python .venv -e ".[dev]"
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pytest -q          # 80 项测试应全绿
```

### 4.1 ERA5 参考场 era5_240（432 GB，网络 ~数小时-1天，来源=WeatherBench2 公共 zarr）
```bash
cd experiments/archesweather-reproduction/scripts
# 1998-2020：官方 WB2 240x121 conservative zarr → 分片（校验：尺寸/变量/有限值）
./prepare_data.sh                       # 主入口（分年下载+validate）
python prepare_era5_2021_2022.py        # 2021-2022（含 2023-01 头）
python prepare_era5_2023_arco.py        # 2023（ARCO v3 重网格版）
```
- **约定要点**：此数据是**面积平均（conservative/boxavg）约定**——一切与它对比或
  喂给 DA 的场必须用同约定（AIFS 侧已内置 N320→0.25°→7×7 盒平均，勿改回线性）。
- 验证锚点：104 个 `era5_240_{year}_{h}h.nc`；单点抽查 `2m_temperature` 与
  ARCO 盒平均差 ~0.08 K。

### 4.2 METAR 观测缓存（19 GB，来源=WB2 metar parquet，公共）
```bash
.venv/bin/python scripts/prepare_multivar_metar.py     # 重建 outputs/metar_multivar_cache/
```
- 验证锚点：每时次 `YYYYMMDDThh.pt`；站点角色划分 seed=42 与旧机一致
  （`norm_stats_train_2018_2020.json` 引用同一划分）。

### 4.3 DA 采样缓存（outputs/{cache,da_sample_cache,da_test_sample_cache}，~13 GB）
首次运行对应 runner 时**自动重建**（cache 目录机制），无需手动。快速预热的入口：
```bash
.venv/bin/python scripts/cycling_prefetch.py --help    # 按需选择范围
```

### 4.4 ArchesWeather 背景（single_da_05/backgrounds 45 GB + bg_t2m_* 2.7 GB）
```bash
.venv/bin/python scripts/generate_full_state_backgrounds.py   # 读 freeze ckpt 生成
```
- 前置：`outputs/forecast_freeze/…/checkpoint.ckpt`（§2 已带）。
- 验证锚点：npz 格式 surface(4,1,121,240)/level(6,13,121,240)、物理值、纬度北起。

### 4.5 AIFS 输入与背景（aifs-background 线）
```bash
cd experiments/aifs-background
# ① 输入分片（ARCO v3 → 90ch 0.25° zstd-7 .npy.zst；~40h/年，网络主导）
../../.venv/bin/python prepare_aifs_inputs.py --year 2021
# ② 背景（AIFS-1.1 6h → 盒平均 1.5° npz；~6s/时次 GPU）
CUDA_VISIBLE_DEVICES=0 ANEMOI_INFERENCE_NUM_CHUNKS=32 \
  v1.1/.venv/bin/python run_aifs_background.py \
  --inputs data/aifs_inputs/2021 --out data/bg_aifs_2021
# ③ P2 验收
../../.venv/bin/python eval_p2.py data/bg_aifs_2021
```
- **验收锚点（预注册）**：T2M **0.3513 K**（552 时次 Jan–May 实测；锚点 0.441409，
  强通过门 <0.38）。全年数字待抽取完成后补终章。

### 4.6 复训 ArchesWeather（仅当需要，一般不需要）
GeoArches 钉在 `fb2285fc…`（manifest 内含 worktree patch 快照）；
按 `experiments/archesweather-reproduction/LOCAL_REPRO_NOTES.md` +
`scripts/prepare_train_env.sh` 重建（PyTorch 2.5.1+cu124、bf16-mixed）。

### 4.7 AIFS-1.1 运行环境（v1.1 venv，版本必须精确对齐）
```bash
cd experiments/aifs-background/v1.1
# ckpt 从 HuggingFace 重下（sha256 校验见 manifest.json / P0_NOTES.md）
curl -L -o aifs-single-mse-1.1.ckpt "https://huggingface.co/ecmwf/aifs-single-1.1/resolve/main/aifs-single-mse-1.1.ckpt"
cd .. && uv venv v1.1/.venv --python 3.12
uv pip install --python v1.1/.venv "torch==2.4.1" "torch-geometric==2.4.0" \
  "anemoi-inference[huggingface]==0.6.3" "anemoi-models==0.5.0" \
  "earthkit-regrid==0.4.0" zarr gcsfs numcodecs
uv pip install --python v1.1/.venv "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```
- **三个生死开关**：① 必须用 v1.1/.venv（0.9.x 无法加载 1.1 pickle）；
  ② `ANEMOI_INFERENCE_NUM_CHUNKS=32`（默认不分块 → 24G OOM）；
  ③ 输出日期语义 = t+6h（脚本已处理）。

## 5. 新机磁盘预算（峰值）

| 项 | 峰值 |
|---|---:|
| git + 不可再生资产（§2）| ~7 GB |
| era5_240 全量 | 432 GB |
| AIFS 输入 2021（zstd-7）| ~284 GB |
| AIFS 背景 2021 | ~11 GB |
| 观测/DA 缓存 | ~33 GB |
| venvs | ~22 GB |
| **合计（全量重备）** | **~790 GB** |

分期建议：先 4.1+4.2+4.3（DA 复评能力），AIFS 线过 P3 决策后再抽训练年。

## 6. 验证清单（迁移成功的判据）

1. `pytest -q` 全绿（数据无关的 80 项）
2. `sha256sum outputs/forecast_freeze/…/checkpoint.ckpt` = `267641e1f051…`
3. era5_240 重备后：任取一年的 result.json 复评入口可跑通（`scripts/step3_spatial_stratification.py --stride 4`）
4. AIFS 背景重备后：`eval_p2.py` 输出 T2M ≈ 0.35 ± 0.01（同种子确定性，
   允许 GPU 代差带来的小抖动）
5. 对照 `docs/EXPERIMENT_REGISTRY.md` 阶段 5 的锚点数字
