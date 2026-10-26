# offgrid-da

Background-Aware Off-Grid Observation Assimilation (WeatherBench2 Pangu + ERA5 + MADIS METAR).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Audit raw data
python scripts/audit_data.py

# Prepare training data
python scripts/prepare_data.py
```

## Testing

```bash
pytest
```

## Project layout

- `src/offgrid_da/` — core library (data loading, models, losses, evaluation, utils)
- `configs/` — experiment and data configuration files
- `scripts/` — runnable pipelines (data audit, preparation, ...)
- `tests/` — test suite
- `docs/` — design notes and data documentation
