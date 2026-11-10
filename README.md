# OffGrid-DA

Background-Aware Off-Grid Observation Assimilation with WeatherBench 2
(Pangu forecasts + ERA5 analyses + MADIS METAR station observations).

Residual single-cycle DA: `analysis = background + Δθ(background, off-grid observations)`.

## Status (Round 1: M0 + M1 minimal pipeline)

- ✅ Phase 0 audit of the three live data sources (`docs/DATA_AUDIT.md`)
- ✅ Single-cycle train/validation/test samples with train-only normalization (`docs/DATA_SPEC.md`)
- ✅ 80 automated tests passing, incl. real-data leakage/QC/alignment checks
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

# Train-only normalization and the identity baseline
python scripts/prepare_train_norms.py
python scripts/evaluate_baseline0.py --split minimal

# Tests (network required for real-data tests)
pytest tests/
```

```python
from offgrid_da.data.dataset import SingleCycleDADataset
from offgrid_da.data.collate import single_cycle_collate

dataset = SingleCycleDADataset(split="train", require_normalization=True)
sample = dataset[0]      # physical and normalized grids [1,121,240], sparse station records
batch = single_cycle_collate([dataset[0], dataset[1]])
```

## Layout

```
configs/     data / model / train / experiment YAML
src/offgrid_da/
  data/      forecast, analysis, metar, qc, observations, alignment,
             stations, dataset, collate, variables
  models/    (Phase 2) encoders, sparse cross-attention, decoder
  evaluation/ Baseline 0 and fixed area-weighted grid metrics
  ...        baselines / losses / evaluation / utils
scripts/     audit_data, prepare_data, visualize, train (Phase 3), evaluate
tests/       74 tests incl. real-data schema, time alignment, QC, leakage
docs/        DATA_AUDIT, DATA_SPEC, VARIABLE_MAPPING, status & logs
outputs/     data_audit figures & CSVs, cache
```

## Data

All public GCS (anonymous): `gs://weatherbench2/datasets/{pangu,era5,metar}/…`
— exact paths in `configs/data/wb2_metar.yaml`.


## Training sample pipeline

The configured 2018-2020 interval yields 4,383 samples, one at every available six-hour ERA5 valid time. For each valid time, the index selects the shortest available Pangu forecast from a 00/12 UTC initialization (6h or 12h lead). The dataset indexes cycles and reads each sample on demand; it does not copy all source grids into a second serialized store. Each sample keeps physical-unit background and ERA5 reference fields, and adds normalized copies using the frozen training statistics. Context METAR records are model inputs; held-out records are collated separately for station-space checks. The effective T2M observation error combines the 1.0 K instrument assumption and 2.5 K representativeness floor in quadrature (2.69 K total).
