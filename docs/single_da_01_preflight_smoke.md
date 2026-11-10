# Single-cycle DA preflight

Forecast backgrounds: frozen ArchesWeather 250k, six-hour single-step forecasts initialized from ERA5. Training: 2018–2020; validation: 2021; test: 2022.

All normalized background, reference target, observations and predicted analysis use the training-background mean/std from `outputs/normalization/arches_train_2018_2020.json`. Physical-unit fields remain available. Reference-specific statistics are retained in the statistics file for audit, but are not used as target coordinates in ArchesSingleCycleDADataset.

Entry point: `scripts/single_cycle_acceptance.py`. Default training scope is a small preflight, not the full experiment. The script supports configurable date ends, sample limits, batch size, validation frequency, checkpoint frequency and optimizer-state resume. It uses cosine-latitude-weighted training/validation MSE, reports validation RMSE in Kelvin, and writes SwanLab offline logs. The model configuration currently used is dim=8, heads=2, depth=1, radius=1000 km, at most 8 neighbors; a full experiment recipe still needs to be selected.

## Verified commands

```bash
.venv/bin/python scripts/single_cycle_acceptance.py --mode baseline --output outputs/single_da_01_preflight_smoke
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/single_cycle_acceptance.py --mode train --steps 10 --output outputs/single_da_01_preflight_smoke/smoke_weighted
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/single_cycle_acceptance.py --mode train --steps 15 --resume outputs/single_da_01_preflight_smoke/smoke_weighted/checkpoint_10.pt --output outputs/single_da_01_preflight_smoke/resume_weighted
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/single_cycle_acceptance.py --mode train --steps 20 --resume outputs/single_da_01_preflight_smoke/resume_weighted/checkpoint_15.pt --output outputs/single_da_01_preflight_smoke/final_smoke
```

To include the complete training/validation date windows, pass `--train-end 2021-01-01 --val-end 2022-01-01 --sample-limit 0`. This implementation preloads selected samples in memory; the full-data mode has not been exercised. Smoke metrics use two training cycles and two validation cycles; they are engineering checks, not full-year model skill. The 2022 test set is not used by the preflight training or model validation.

## Environment

Training preflight uses project `.venv`; SwanLab 0.10.3 was installed with uv. Dependency resolution changed protobuf from 7.36.2 to 6.33.6. Forecast inference continues to use the separate GeoArches environment.

## Results

Regression tests: 29 passed. Weighted validation RMSE on two cycles: step 10 0.4540515 K; resumed step 15 0.4538889 K; resumed step 20 0.4539344 K. Checkpoint model reload matched exactly. Station roles were disjoint both within samples and across the selected training/validation samples. Full-year background-only score is written to `outputs/single_da_01_preflight_smoke/baseline0.json` upon completion.

Full 2021 validation Baseline 0 completed: 1460 cycles, pooled area-weighted RMSE 0.441409050 K, pooled MAE 0.270702538 K.
