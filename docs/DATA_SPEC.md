# OffGrid-DA Data Specification (Phase 1)

> 历史口径注记（2026-11-10）：本文写于 Round 1（Pangu 背景、Jan-2020 窗口时代）。
> 当前实际数据协议已演进：背景 = 冻结 ArchesWeather 预报（任意 6h 时刻，非 Pangu
> 存档匹配），训练/验证 = 2018–2020/2021 全年（4384/1460 周期），82 通道状态 +
> 五变量 METAR（见 single_da_05/06 与 FORECAST_INTERFACE.md）。本文保留的仍有效
> 部分：网格与坐标约定、样本结构、QC 与站点划分、指标定义。窗口内数字仅作
> Round-1 参照。

**Scope:** single-cycle DA sample pipeline. It creates one sample at every six-hour ERA5 valid time that can be matched to a Pangu forecast. The legacy Jan-2020 data audit used a fixed 6h lead (62 cycles); the current dataset index uses both 6h and 12h leads (124 cycles for Jan 2020). Train/validation/test intervals are 2018-2020, 2021, and 2022, all half-open: [start, end).
All numbers below were measured on real data (see `outputs/data_audit/sample_statistics.csv`).

## 1. Sample definition

| Field | Value | Source |
|---|---|---|
| Background | Pangu 6h forecast, init 00/12 UTC | `prediction_timedelta=6` |
| Reference | ERA5 analysis at valid time | 6-hourly zarr |
| Observations | MADIS METAR, `timeObs` in (t−60min, t] | timeObs-by-hour parquet |
| Valid times | Every 6h, 2020-01-01T00 .. 2020-01-31T18 | 124 samples in the current dataset index |
| Received cutoff | `timeReceived` ≤ t+30min | latency tolerance |
| Target variable | T2M (K) | single variable, Stage A |

Index columns: `sample_id, init_time, lead_time_hours, valid_time, forecast_index, analysis_index, observation_window_start, observation_window_end`.

## 2. Grid conventions (canonical, verified by tests)

- 240×121 equiangular with poles, 1.5°; Pangu and ERA5 grids identical.
- Arrays `[C, 121, 240]`: **latitude descending 90→−90**, longitude ascending [0, 360).
- Source zarrs store (…, longitude, latitude) with latitude ascending → readers transpose + flip (tested by `test_field_rows_correspond_to_latitudes`).
- METAR station longitude [−180, 180] → `mod 360`.
- Bilinear obs operator: periodic in longitude, clamped at poles, NaN-propagating; numpy and differentiable torch versions agree to 1e-10 (`test_numpy_torch_consistency_and_differentiable`).

## 3. Observation record schema

Per scalar token: `station_id, source_id=0(METAR), variable_id=0(T2M), latitude, longitude[0,360), elevation_m, valid_time(=timeObs), received_time(=timeReceived), value(K), qc_flag(DD), quality_code`.

QC chain (per-cycle counts in `sample_statistics.csv`):
1. read hourly partitions covering the window (partitions hold timeObs [H:00, H:59] — verified)
2. timeObs ∈ (t−60min, t]
3. timeReceived ≤ t+30min
4. location QC: finite lat/lon/elev, |lat|≤90, lon ∈ [−180,180], elev ∈ [−450, 9000] m
5. station split filter: **held-out stations never enter model input**
6. DD flag ∈ {S,V,C,T,B,G,Q} (reject Z missing / X implausible) + finite + T2M ∈ [180, 340] K
7. dedup per station per window: keep latest timeObs

## 4. Historical audit statistics (Jan 2020, fixed 6h lead, 62 cycles)

These values are retained from the source-data audit and do not describe the expanded 124-cycle sample index.

| Quantity | Value |
|---|---|
| stations in window | 5457 total → 4397 context / 1060 held-out (80/20, deterministic) |
| obs per cycle after QC | min 3715 / mean 3834 / max 3934 (T2M context only) |
| innovation obs−bg (K) | mean +0.071, std 2.493, rmse 2.494 |
| innovation obs−ERA5 (K) | rmse ≈ 2.475 (sample 0) |
| Pangu 6h vs ERA5 grid RMSE (area-weighted) | mean 0.463 K (range 0.43–0.50) |
| background T2M stats | mean 277.261 K, std 20.044 K (n=1.80M cells) |
| reference T2M stats | mean 277.256 K, std 20.054 K |

**Key finding — representativeness dominates the station signal:** grid forecast error is
0.46 K while obs-vs-grid innovation is 2.49 K and obs-vs-ERA5 innovation is about 2.5 K. The
station innovation is therefore mostly point-vs-1.5°-grid representativeness error + obs
error, not Pangu forecast error. Consequences for modelling:

- DA supervision against ERA5 grid remains clean (background ≠ reference),
- station-space evaluation must be interpreted against the ~2.5 K representativeness floor,
- the Innovation feature (Phase 4) must not be treated as pure forecast error.

## 5. Normalization

outputs/normalization/norm_stats_train_2018_2020.json contains train-only statistics over 4,383 six-hour samples: Pangu background 278.865 ± 21.141 K and ERA5 reference 278.851 ± 21.186 K, each from 127,282,320 finite grid points. Metadata records the half-open training interval and confirms validation/test were excluded.

## 6. Caching

- METAR parquet: content-addressed under `outputs/cache/metar/` (124 hourly partitions for
  the minimal window, ≈ 37 MB).
- zarr fields: read directly via gcsfs on demand (each T2M field ≈ 117 KB compressed).
- Per-process `lru_cache` on hourly partitions.

## 7. Leakage guarantees (tested)

- `test_no_future_observation_leakage`: obs_time_offsets ≤ 0 and > −60 min.
- `test_obs_within_window_and_cutoff`: timeObs and timeReceived bounds on real records.
- `test_time_alignment`: valid_time = init + lead exactly; ERA5 contains every valid time.
- Held-out stations: `test_heldout_never_in_model_input`.
- Test split (2022) never touched in this round; no normalization stats from val/test.


## Implemented training sample contract

The Jan-2020 dataset window contains 124 samples, while the legacy fixed-6h source audit retained 62 cycles; the configured train, validation, and test splits are [2018-01-01, 2021-01-01), [2021-01-01, 2022-01-01), and [2022-01-01, 2023-01-01). The dataset indexes every six-hour valid cycle and reads one sample at a time. At a valid time, it chooses the shortest available Pangu lead from init hours 00/12 UTC (6h or 12h). It does not materialize duplicate grid files.

Each item includes Pangu background and same-time ERA5 reference grids with shape [C,121,240], context METAR point records from (t−60 min,t] with the received-time cutoff, grid coordinates, and sample/time identifiers. Physical-unit tensors remain available for metrics. When the configured train-only normalizer is present, background_normalized, reference_normalized, obs_values_normalized, and normalized observation errors are also returned. Context observations use background statistics; the ERA5 reference uses its own statistics.

The obs_* fields contain context stations only. heldout_obs_* fields and the separate heldout_obs_mask carry held-out station records for independent diagnostics and must not be passed to the model. Context and held-out records are padded independently by single_cycle_collate.

Observation uncertainty fields expose instrument and representativeness components separately. The effective standard deviation is sqrt(sigma_instrument^2 + sigma_representativeness^2); for T2M the configured values are 1.0 K and 2.5 K, giving 2.69 K. ERA5 is consistently described as a reference, not error-free truth.
