# OffGrid-DA

Background-Aware Off-Grid Observation Assimilation with WeatherBench 2
(Pangu forecasts + ERA5 analyses + MADIS METAR station observations).

Residual single-cycle DA: `analysis = background + Δθ(background, off-grid observations)`.

## Status (Round 1: M0 + M1 minimal pipeline)

- ✅ Phase 0 audit of the three live data sources (`docs/DATA_AUDIT.md`)
- ✅ Minimal single-cycle dataset over Jan 2020, T2M (`docs/DATA_SPEC.md`)
- ✅ 63 automated tests passing, incl. real-data leakage/QC/alignment checks
- ⬜ Phase 2 model (background encoder + off-grid cross-attention + decoder)

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# or with the existing environment:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Usage

```bash
# Phase 0: audit the three WeatherBench2 sources (live reads, cached)
python scripts/audit_data.py

# Phase 1: build minimal-window samples, statistics and normalization
python scripts/prepare_data.py

# Phase 1: figures (6 panels for 3 random cycles, station split, histograms)
python scripts/visualize.py 3

# Tests (network required for real-data tests; ~4 min first run, ~4 s warm)
pytest tests/
```

```python
from offgrid_da.data.dataset import SingleCycleDADataset
from offgrid_da.data.collate import single_cycle_collate

dataset = SingleCycleDADataset(split="minimal")   # Jan 2020, T2M, lead 6h
sample = dataset[0]      # background/reference [1,121,240], ~3.8k obs tokens
batch = single_cycle_collate([dataset[0], dataset[1]])
```

## Layout

```
configs/     data / model / train / experiment YAML
src/offgrid_da/
  data/      forecast, analysis, metar, qc, observations, alignment,
             stations, dataset, collate, variables
  models/    (Phase 2) encoders, sparse cross-attention, decoder
  ...        baselines / losses / evaluation / utils
scripts/     audit_data, prepare_data, visualize, train (Phase 3), evaluate
tests/       63 tests incl. real-data schema, time alignment, QC, leakage
docs/        DATA_AUDIT, DATA_SPEC, VARIABLE_MAPPING, status & logs
outputs/     data_audit figures & CSVs, cache
```

## Data

All public GCS (anonymous): `gs://weatherbench2/datasets/{pangu,era5,metar}/…`
— exact paths in `configs/data/wb2_metar.yaml`.
