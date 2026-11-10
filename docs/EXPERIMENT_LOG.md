# Experiment Log

> 定位（2026-11-10）：Round 1（2026-10-26/27）的原始日志，历史文档。
> 后续实验的动机与结论见 `EXPERIMENT_REGISTRY.md`；目录与权重索引见 `EXPERIMENTS.md`。

## 2026-10-26/27 — Round 1: M0 + M1 minimal pipeline

**Goal:** data audit + minimal real-data single-cycle samples (no model training).

### E1: Data audit (Phase 0)

- Script: `scripts/audit_data.py` (live reads)
- Findings:
  - Pangu: `prediction_timedelta` 6..240h step 6h → **6h lead exists**; 3652 inits (00/12Z,
    2018-01-01..2022-12-31T12); 11 variables; 240×121, poles included; no units attrs.
  - ERA5: 6-hourly 00/06/12/18Z through 2023-01-10T18; 62 vars; identical grid;
    T2M units K; 11-variable intersection with Pangu (all Pangu vars exist in ERA5).
  - Aligned samples: Jan-2020 window 62; train 2192 / val 730 / test 730 (date intervals are half-open) (2018–2022).
  - METAR: 43 columns verified; temperature K; longitude [−180,180]; partitions hold
    timeObs [H:00,H:59]; DD flags S/V/C/T/B/G/Q/X/Z with only Z valueless; X implausible.
- Deliverables: DATA_AUDIT.md, VARIABLE_MAPPING.md, coverage.png, metar_schema.csv,
  sample_statistics.csv, audit_raw.json.

### E2: Minimal dataset build (Phase 1)

- Script: `scripts/prepare_data.py`, window [2020-01-01, 2020-02-01), T2M.
- Results (62 cycles):
  - obs/cycle: 3715 / 3834 / 3934 (min/mean/max) after QC (context stations only)
  - innovation obs−bg: mean +0.071 K, std 2.493 K
  - area-weighted Pangu-6h vs ERA5 T2M RMSE: 0.463 K mean
  - norm stats (window): bg 277.261±20.044 K; ref 277.256±20.054 K
- **Scientific note:** obs−grid innovation (2.49 K) ≫ grid forecast error (0.46 K) and
  ≈ obs−ERA5 innovation (about 2.5 K) → station innovation dominated by representativeness
  error (airport point vs 1.5° cell), consistent with plan §13 warning 4. Station-space
  metrics must be reported against this ~2.5 K floor; grid-space DA supervision remains
  well-posed (bg ≠ ref).
- Bug found & fixed during E2: area-weighted RMSE divided by row weights instead of
  broadcast cell weights (7.16 K → 0.46 K). Regression test added
  (`test_evaluation_metrics.py`).

### E3: Visualization

- `scripts/visualize.py`: cycles 20200103T18, 20200121T06, 20200124T18 (seed 42);
  6-panel QA figures, station-split map, innovation histograms.

### E4: Baseline 0 and train normalization

- scripts/prepare_train_norms.py: 2,192 cycles from [2018-01-01, 2021-01-01); validation and test excluded.
  - Pangu T2M: mean 278.8796 K, sample std 21.1460 K.
  - ERA5 reference T2M: mean 278.8662 K, sample std 21.1624 K.
  - 63,655,680 finite grid points per field. Saved JSON and provenance metadata under outputs/normalization/.
- scripts/evaluate_baseline0.py --split minimal: exact identity analysis = Pangu background; ERA5 reference.
  - Pooled area-weighted grid RMSE 0.46283 K; MAE 0.27215 K; mean per-cycle RMSE 0.46265 K.
  - Context obs-background RMSE 2.494 K; context obs-ERA5-reference RMSE 2.444 K over 237,726 reports. These station diagnostics include the roughly 2.5 K representativeness floor and are not pure forecast-error scores.
  - Outputs: outputs/baselines/baseline0_minimal.json and per-cycle CSV.

### Test suite

- 74 passed / 0 failed (tests/). Coverage: schema, variable mapping, time alignment,
  lead availability, obs window + cutoff, no-leakage, QC filters, station split,
  lat/lon mapping incl. pole-row equivalence to source zarr, interpolation analytic
  tests incl. seam/periodicity, collate padding/mask, DataLoader integration, metrics,
  normalization.
