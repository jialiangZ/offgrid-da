# ArchesWeather reproduction workspace

This workspace prepares a single deterministic ArchesWeather-M run on the 1.5-degree ERA5 grid.

## Reproduction target

- Model: deterministic ArchesWeather-M, adapted to a 6-hour forecast lead.
- Grid: WeatherBench 2 ERA5, 240 longitude by 121 latitude points (1.5 degrees).
- State: six upper-air variables at 13 pressure levels plus four surface variables.
- Source cadence and forecast lead: 6-hourly ERA5; each step maps (t-6h, t) to t+6h.
- Split: train 1998-2017 (20 years), validation 2019, test 2020. 2018 supplies the validation previous-state boundary; recent-data fine-tuning uses 2007-2017.
- Recipe: 300,000 optimization steps, including recent-period fine-tuning after step 250,000. The paper also reports a 20,000-step autoregressive fine-tuning stage.

This changes the temporal protocol from the paper's 24-hour lead; the model architecture remains ArchesWeather-M.

## Directory layout

- `scripts/` — all runnable scripts: data preparation (`prepare_data.sh`, `validate_data.py`, `prepare_era5_2021_2022.py`), normalization statistics (`compute_train_norms*.py`), environment setup (`prepare_train_env.sh`), and training launchers (`run_training_20y.sh`, `monitor_and_start_training.sh`, `continue_training_150k.sh`, `prepare_continuation.py`). All scripts resolve the workspace root relative to their own location.
- `data/era5_240/full/` — one NetCDF file per year and UTC cycle. Referenced by absolute path from the main repository; do not relocate.
- `stats/` — normalization statistics (`era5_norm_stats_1998_2017_lead6h.pt` is referenced by absolute path from the main repository; do not relocate).
- `geoarches/` — pinned upstream checkout (v0.2.2) with its own `.venv`, `modelstore/` (checkpoints), `swanlogs/`, and `evalstore/`. Referenced by absolute path from the main repository; do not relocate.
- `data-venv/` — isolated environment used for data downloads.
- `patches/` — code changes relative to the pinned upstream release.
- `logs/` — runtime logs grouped per phase: `data_prep/`, `run_20y/`, `continue150k/`.

## Pinned implementation

- Upstream: https://github.com/INRIA/geoarches
- Release: v0.2.2, commit fb2285fc960fca9f11b2b8d0126deba810f87895
- Paper configuration: geoarches/paper/configs/archesweather-m-seed0.yaml
- The combined 20-year and 6-hour code changes are preserved in patches/archesweather-20year-6h.patch; the earlier 24-hour patch remains available as patches/archesweather-20year.patch.

## Data and environments

- Data: data/era5_240/full/, one NetCDF file per year and UTC cycle, from the WeatherBench 2 6-hourly 240x121 ERA5 Zarr archive.
- Download range: 1998-2020 inclusive, four UTC shards per year; 92 files.
- Download complete: 421,725,637,516 bytes. The structural validator passed file counts, dimensions, timestamps, and UTC hours.
- Expected size: about 18.3 GB per year, around 420 GB total.
- Data environment: data-venv/. Training environment: geoarches/.venv. Recreate the GPU-compatible environment with scripts/prepare_train_env.sh. Training metrics are recorded with SwanLab in offline mode under geoarches/swanlogs/.
- Training statistics: stats/era5_norm_stats_1998_2017_lead6h.pt, with input moments from 1998-2017 and delta standard deviations from adjacent 6-hour pairs.

## Preflight state

- Train loader: 29,218 valid samples. Validation loader: 1,460 samples.
- Test loader: 1,462 valid 6-hourly starts from 2020-01-01 00Z through 2020-12-31 06Z. The final two 6-hourly starts cannot complete a two-step rollout because data ends at 2020-12-31 18Z and no 2021 boundary shard is present.
- The 84.6M-parameter model completed a four-GPU short diagnostic with the 6-hour statistics and two-step validation; losses were finite.
- The run uses built-in backbone gradient checkpointing and AdamW foreach=False to fit 24 GB RTX 4090 cards.
- Full training has not started.

## Start full training

After reviewing the boundary limitation, run:

```bash
bash scripts/run_training_20y.sh
```

This launches the 300,000-step deterministic model run; it does not include a separate 20,000-step autoregressive fine-tuning stage.
