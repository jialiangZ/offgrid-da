# Baseline 0 Protocol

> 更新注记（2026-11-10）：本协议的**背景定义已变更**。Round 1（2026-10）背景为
> Pangu 存档预报（00/12Z 起报的最短 lead）；自 single_da_01 起，背景统一为
> **冻结的 ArchesWeather-250k 预报**（`configs/forecast/archesweather_6h_contract.json`
> 钉住的 checkpoint，任意 6h 时刻从 ERA5 t−12h/t−6h 状态生成）。下文"Pangu"字样
> 仅为 Round-1 历史口径；指标定义、站点诊断口径与归一化原则不变，继续有效。

## Definition

Baseline 0 makes no observation-driven update:

    analysis_estimate = background

The background is the frozen ArchesWeather forecast at the valid time (historically, in Round 1, the shortest available Pangu forecast from a 00/12 UTC initialization), and the reference field is the ERA5 analysis at the same valid time. The sample index includes every matchable six-hour ERA5 valid time. ERA5 is a reference for evaluation, not error-free truth.

## Fixed grid metrics

For each valid grid cell at latitude phi, use area weight cos(phi) and exclude cells where either field is non-finite.

- Area-weighted RMSE pooled across cycles: sqrt(sum(w * (analysis - reference)^2) / sum(w)).
- Area-weighted MAE pooled across cycles: sum(w * abs(analysis - reference)) / sum(w).
- Also report the arithmetic mean of per-cycle area-weighted RMSE so temporal variability is visible.

The primary baseline comparison is the pooled area-weighted grid RMSE. The Baseline 0 identity is exact: its analysis estimate equals the Pangu background array.

## Station diagnostics

The evaluator reports context-METAR RMSE against bilinearly sampled Pangu and ERA5 fields. These are diagnostics, not the primary grid score. Reports come from the preceding 60-minute observation window and include point-to-1.5-degree-grid representativeness error, observation error, and timing mismatch. The observed January 2020 obs-minus-ERA5 residual is about 2.5 K; treat that as a representativeness floor, not pure forecast error. Do not describe ERA5 as truth.

## Training normalization

scripts/prepare_train_norms.py reads only the configured training interval [2018-01-01, 2021-01-01), at every configured six-hour valid time, using the shortest available lead (6h or 12h). It calculates finite-grid-point means and sample standard deviations separately for Pangu background and ERA5 reference variables. Validation (2021) and test (2022) are excluded. The small JSON output is accompanied by metadata recording the interval and sample count.

## Reproduce

    PYTHONPATH=src python scripts/prepare_train_norms.py
    PYTHONPATH=src python scripts/evaluate_baseline0.py --split minimal
    pytest tests/

The current January 2020 run uses 124 valid times (62 at 6h lead and 62 at 12h lead). It produced pooled grid RMSE 0.50606 K, pooled grid MAE 0.29926 K, and mean per-cycle grid RMSE 0.50417 K. Context station diagnostics over 475,766 reports were 2.486 K against Pangu and 2.402 K against ERA5. The older 62-cycle fixed-6h source audit is not the current baseline result.

The evaluator writes one JSON summary and a per-cycle CSV under outputs/baselines/.
