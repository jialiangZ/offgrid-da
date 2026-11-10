"""Evaluate Baseline 0 (analysis estimate equals the background field)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402
from offgrid_da.data.dataset import SingleCycleDADataset  # noqa: E402
from offgrid_da.evaluation import area_weighted_mae, area_weighted_rmse, baseline0  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("minimal", "train", "val", "test"), default="minimal")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baselines"
    )
    args = parser.parse_args()

    dataset = SingleCycleDADataset(split=args.split)
    grid = dataset.pangu.grid()
    latitude = grid["latitude"]
    per_cycle: list[dict] = []
    pooled_obs_bg_sq = 0.0
    pooled_obs_ref_sq = 0.0
    pooled_obs_count = 0
    all_bg_sq = 0.0
    all_abs = 0.0
    all_area = 0.0
    cycle_rmse: list[float] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        background = sample["background"].numpy()
        reference = sample["reference"].numpy()
        analysis = baseline0(background)
        grid_rmse = area_weighted_rmse(analysis, reference, latitude)
        grid_mae = area_weighted_mae(analysis, reference, latitude)
        cycle_rmse.append(grid_rmse)

        valid = np.isfinite(analysis) & np.isfinite(reference)
        weights = np.broadcast_to(
            np.maximum(np.cos(np.deg2rad(latitude)), 0.0)[None, :, None], analysis.shape
        )
        error = np.where(valid, analysis.astype(np.float64) - reference, 0.0)
        all_bg_sq += float((weights * error * error).sum(dtype=np.float64))
        all_abs += float((weights * np.abs(error)).sum(dtype=np.float64))
        all_area += float(np.where(valid, weights, 0.0).sum(dtype=np.float64))

        n_obs = int(sample["obs_values"].shape[0])
        obs_bg_rmse = np.nan
        obs_ref_rmse = np.nan
        if n_obs:
            obs_lat = sample["obs_locations"][:, 0].numpy()
            obs_lon = sample["obs_locations"][:, 1].numpy()
            obs_values = sample["obs_values"].numpy().astype(np.float64)
            bg_at_obs = sample_background_at_obs(
                background[0], grid["latitude"], grid["longitude"], obs_lat, obs_lon
            )
            ref_at_obs = sample_background_at_obs(
                reference[0], grid["latitude"], grid["longitude"], obs_lat, obs_lon
            )
            bg_error = obs_values - bg_at_obs
            ref_error = obs_values - ref_at_obs
            finite = np.isfinite(bg_error) & np.isfinite(ref_error)
            if finite.any():
                pooled_obs_bg_sq += float(np.square(bg_error[finite]).sum())
                pooled_obs_ref_sq += float(np.square(ref_error[finite]).sum())
                pooled_obs_count += int(finite.sum())
                obs_bg_rmse = float(np.sqrt(np.mean(np.square(bg_error[finite]))))
                obs_ref_rmse = float(np.sqrt(np.mean(np.square(ref_error[finite]))))

        per_cycle.append({
            "sample_id": sample["sample_id"],
            "valid_time": sample["valid_time"].isoformat(),
            "lead_time_hours": int(dataset.index.iloc[i]["lead_time_hours"]),
            "n_context_observations": n_obs,
            "area_weighted_grid_rmse": grid_rmse,
            "area_weighted_grid_mae": grid_mae,
            "context_obs_background_rmse": obs_bg_rmse,
            "context_obs_era5_reference_rmse": obs_ref_rmse,
        })
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"evaluated {i + 1}/{len(dataset)} cycles", flush=True)

    summary = {
        "baseline": "Baseline 0: analysis = background",
        "split": args.split,
        "interval": {
            "start_inclusive": str(dataset.index["valid_time"].min()),
            "last_valid_time": str(dataset.index["valid_time"].max()),
        },
        "sample_count": len(dataset),
        "forecast_lead_hours": {
            str(int(lead)): int(count)
            for lead, count in dataset.index["lead_time_hours"].value_counts().sort_index().items()
        },
        "reference_field": "ERA5 analysis (reference, not error-free truth)",
        "grid_metrics": {
            "area_weighted_rmse_all_cycles": float(np.sqrt(all_bg_sq / all_area)),
            "area_weighted_mae_all_cycles": float(all_abs / all_area),
            "mean_per_cycle_area_weighted_rmse": float(np.mean(cycle_rmse)),
        },
        "station_diagnostics": {
            "observation_count": pooled_obs_count,
            "context_obs_vs_background_rmse": (
                float(np.sqrt(pooled_obs_bg_sq / pooled_obs_count)) if pooled_obs_count else None
            ),
            "context_obs_vs_era5_reference_rmse": (
                float(np.sqrt(pooled_obs_ref_sq / pooled_obs_count)) if pooled_obs_count else None
            ),
            "interpretation": (
                "Diagnostic only: station-grid residuals include observation error, "
                "hour-window timing mismatch, and roughly 2.5 K point-to-grid representativeness. "
                "Do not interpret this as pure forecast error or as ground-truth verification."
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"baseline0_{args.split}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(per_cycle).to_csv(args.output_dir / f"{stem}_by_cycle.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
