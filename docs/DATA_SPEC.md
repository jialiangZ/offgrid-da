# OffGrid-DA Data Specification (Phase 1)

**Scope:** minimal single-cycle DA pipeline over 2020-01-01 .. 2020-01-31 (60 analysis cycles).
All numbers below were measured on real data (see `outputs/data_audit/sample_statistics.csv`).

## 1. Sample definition

| Field | Value | Source |
|---|---|---|
| Background | Pangu 6h forecast, init 00/12 UTC | `prediction_timedelta=6` |
| Reference | ERA5 analysis at valid time | 6-hourly zarr |
| Observations | MADIS METAR, `timeObs` in (t−60min, t] | timeObs-by-hour parquet |
| Valid times | 06/18 UTC daily, 2020-01-01T06 .. 2020-01-30T18 | 60 samples |
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

## 4. Measured statistics (Jan 2020, 60 cycles)

| Quantity | Value |
|---|---|
| stations in window | 5455 total → 4374 context / 1081 held-out (80/20, deterministic) |
| obs per cycle after QC | min 3687 / mean 3812 / max 3923 (T2M context only) |
| innovation obs−bg (K) | mean +0.078, std 2.423, rmse 2.424 |
| innovation obs−ERA5 (K) | rmse ≈ 2.475 (sample 0) |
| Pangu 6h vs ERA5 grid RMSE (area-weighted) | mean 0.462 K (range 0.43–0.50) |
| background T2M stats | mean 277.283 K, std 20.001 K (n=1.74M cells) |
| reference T2M stats | mean 277.280 K, std 20.010 K |

**Key finding — representativeness dominates the station signal:** grid forecast error is
0.46 K while obs-vs-grid innovation is 2.42 K and obs-vs-ERA5 innovation is 2.48 K. The
station innovation is therefore mostly point-vs-1.5°-grid representativeness error + obs
error, not Pangu forecast error. Consequences for modelling:

- DA supervision against ERA5 grid remains clean (background ≠ reference),
- station-space evaluation must be interpreted against the ~2.4 K representativeness floor,
- the Innovation feature (Phase 4) must not be treated as pure forecast error.

## 5. Normalization

`outputs/cache/norm_stats_jan2020.json` holds background/reference mean/std for the minimal
window. **These are demo-window statistics**: before full training, statistics must be
recomputed over 2018-2020 train years only (plan section 3.9); `NormStats` supports this.

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
