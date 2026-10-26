# Implementation Status

**Project:** OffGrid-DA — Background-Aware Off-Grid Observation Assimilation
**Round:** 1 (M0 + M1 minimal pipeline) — **Date:** 2026-10-27

## Milestone summary

| Milestone | Status | Evidence |
|---|---|---|
| M0 environment / data audit | ✅ done | `docs/DATA_AUDIT.md`, `docs/VARIABLE_MAPPING.md`, `outputs/data_audit/` |
| M1 minimal dataset + tests | ✅ done | 63/63 tests pass, real samples, `docs/DATA_SPEC.md` |
| M2 minimal Off-Grid DA model | ⬜ next | — |

## Implemented

- `src/offgrid_da/data/forecast.py` — Pangu reader (`prediction_timedelta` coordinate,
  canonical grid conversion [C,121,240] lat-descending)
- `src/offgrid_da/data/analysis.py` — ERA5 reader (same canonicalization)
- `src/offgrid_da/data/metar.py` — HTTPS+cache parquet reader, window reads
- `src/offgrid_da/data/variables.py` — variable registry, DD flag semantics + quality codes
- `src/offgrid_da/data/qc.py` — window / received-cutoff / location / DD / range / dedup
  filters with per-step counts
- `src/offgrid_da/data/observations.py` — `ObservationRecord` / `ObservationSet`,
  METAR → scalar tokens
- `src/offgrid_da/data/alignment.py` — sample index (init+lead→valid), bilinear obs
  operator (numpy + differentiable torch), longitude periodicity, pole clamping,
  spherical distances
- `src/offgrid_da/data/stations.py` — deterministic 80/20 context/held-out split
- `src/offgrid_da/data/dataset.py` — `SingleCycleDADataset` (plan §3.8 output contract)
- `src/offgrid_da/data/collate.py` — padded batches with token mask
- `src/offgrid_da/utils/` — config (YAML), normalization (Welford, train-only),
  io (cached GCS fetch), reproducibility
- `scripts/audit_data.py`, `scripts/prepare_data.py`, `scripts/visualize.py`
- 63 automated tests (12 files) incl. real-data schema/window/leakage tests

## Verified

- `pytest tests/` → **63 passed** (first network run cached ~4 min, warm run ~4 s)
- `scripts/audit_data.py` → full Phase-0 audit on live data
- `scripts/prepare_data.py` → 60/60 cycles, statistics CSV + norm stats
- `scripts/visualize.py` → 3 random cycles × 6 panels, station-split map, histograms
- `DataLoader(batch_size=2, collate_fn=single_cycle_collate)` produces training-ready
  batches of real samples

## Data used

- Pangu: `gs://weatherbench2/datasets/pangu/2018-2022_0012_...zarr`, lead 6h, Jan-2020 inits
- ERA5: `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_...zarr`, T2M
- METAR: `metar-timeObs-by-hour_...`, 124 hourly partitions (Jan 2020)

## Configuration

`configs/data/wb2_metar.yaml` (paths, window 60 min, latency 30 min, splits, station split).

## Known deviations from the plan

1. Pangu has **no `lead_time` coordinate** — the real name is `prediction_timedelta`
   (int64 hours). Code and docs use the real name.
2. METAR QC in the parquet export carries only DD descriptors (no numeric MADIS QC
   codes). DD semantics beyond the documented S/T/V/C/Z were characterized empirically
   (B/G/Q/X; only Z has no values; X rows implausible) — see DATA_AUDIT §4.
3. Hourly partitions contain timeObs [H:00, H:59] (rebucketed), not the raw
   [H−15, H+44] MADIS window — window reader adapted accordingly.

## Next (Round 2 proposal)

1. M2: Background encoder + observation encoder + sparse spherical-kNN cross-attention +
   zero-init residual decoder; plan §4.7 tests A–G.
2. Cache monthly METAR as consolidated arrays to speed dataset init (~190 s cold scan →
   seconds).
3. Compute 2018–2020 train normalization statistics (plan §3.9) and station split over
   the full train window.
4. Baseline 0 (background identity) evaluation harness on the minimal window.
