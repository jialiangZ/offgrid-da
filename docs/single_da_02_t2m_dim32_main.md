# Single-cycle experiment v1

Configuration: `configs/da/single_cycle_v1.json`. Frozen execution copy and source snapshots: `outputs/single_da_02_t2m_dim32_main/config.json`, `implementation.json`, and `source_snapshot/`.

Forecast: fixed ArchesWeather 250k, independent six-hour forecasts. Train: 2018–2020 (4384 cycles), validation: 2021 (1460 cycles). The 2022 test set is excluded from the experiment runner, cache preparation and model selection.

## Recipe

DA model: two residual convolution blocks with layered spherical local cross-attention, dimension 32, four heads, 1000 km radius, nearest 16 valid observations. Analysis increment decoder starts at zero; globally empty observations produce exact background identity. Innovation remains disabled. The background-only Neural Correction uses exactly the same initial convolutional branch and output decoder, with no observation input. It has fewer parameters; the comparison alone does not isolate attention capacity from observation information.

Batch size 1. AdamW, learning rate 3e-4, weight decay .01; 200 warmup updates then cosine decay to .05 of peak learning rate. Maximum 10 epochs (43840 updates); evaluate all 1460 validation cycles each epoch; stop after four epochs without improvement. Save every 1000 steps, best validation checkpoint, and final state. All data order is shuffled deterministically by seed+epoch; both models use seed 42 and the same ordering and initial shared background weights. Each model selects its best checkpoint on validation. A 256-step trial precedes formal training, with a frozen 64-cycle validation subset spanning the year and validation every 64 steps. Trial warmup is 64 steps because of its shorter budget. Formal runs restart from the common initialization, not the trial checkpoint.

All normalized inputs and targets use the training-background mean/std. Training loss is cosine-latitude-weighted MSE; model selection uses pooled weighted grid RMSE in Kelvin, aggregating squared error and weights before square root. ERA5 is a reference field. METAR station observations retain instrument/representativeness errors in cached samples; those errors are not currently used as network inputs or loss weights. No station-space skill claim is made.

## Data and memory

`LazyArchesDataset` constructs an index without scanning METAR. Hash-based station membership remains fixed across time windows. Reading one sample validates physical/normalized fields, time offsets, station isolation and writes an atomic versioned disk cache. Cache signatures include forecast manifest, training normalization, data configuration and data-transform source hashes. Training and validation load cached samples on demand; no list of full tensor samples is built in RAM. Full-cache preparation visits every training and validation index before any formal training is allowed. Eight preparation workers and two training loader workers are used.

The local attention gather was changed from an expanded grid-query by all-observations tensor to direct indexed token gathering, preserving the operation and reducing backward memory.

## Gates and restart

Runner: `.venv/bin/python scripts/run_single_cycle_experiment.py`. It uses physical GPU 3 through CUDA_VISIBLE_DEVICES=3, which maps to cuda:0 inside child training processes. It holds an exclusive pipeline lock and refuses changes in the frozen recipe or implementation. Stage order: complete data traversal -> 256-step trial -> DA training -> Neural Correction training -> comparison.json. Failure in a stage stops subsequent stages and is recorded in status.json.

Each training checkpoint contains model, optimizer, learning-rate scheduler, global step, data/config signatures and Python/NumPy/Torch/CUDA RNG states. Epoch permutation and within-epoch position are reconstructed from the saved step. Restarting the runner reuses valid disk samples and resumes from last.pt when present. It may repeat work since the last saved checkpoint after an interruption.

## Verified engineering checks

32 focused and regression tests passed. An additional interrupted-versus-continuous GPU training test passed, comparing model, optimizer and scheduler states after resume. The selected dim=32 model completed a real-batch two-step GPU diagnostic in approximately .83 seconds with peak allocated memory approximately 500 MiB; this short measurement is not an end-to-end throughput estimate. Trained observation perturbations changed output; empty observations preserved exact identity. The Neural Correction output was unchanged under observation perturbation.

Pipeline progress: outputs/single_da_02_t2m_dim32_main/status.json; per-stage preparation.log, trial.log, da.log, neural_correction.log. After completion, comparison.json contains the two selected validation RMSEs and their common Background-only Baseline 0. This is a single-seed experiment; comparison metrics do not establish statistical significance or cycling stability.
