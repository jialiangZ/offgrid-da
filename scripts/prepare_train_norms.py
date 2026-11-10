"""Compute normalization statistics from the configured training interval only."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.alignment import build_sample_index  # noqa: E402
from offgrid_da.data.analysis import ERA5Reader  # noqa: E402
from offgrid_da.data.forecast import PanguReader  # noqa: E402
from offgrid_da.utils.config import load_config  # noqa: E402
from offgrid_da.utils.normalization import NormStats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "normalization" / "norm_stats_train_2018_2020.json"
_WORKER_PANGU = None
_WORKER_ERA5 = None
_WORKER_BACKGROUND_VARIABLES = None
_WORKER_REFERENCE_VARIABLES = None


def _initialize_worker(pangu_url, era5_url, background_variables, reference_variables):
    global _WORKER_PANGU, _WORKER_ERA5
    global _WORKER_BACKGROUND_VARIABLES, _WORKER_REFERENCE_VARIABLES
    _WORKER_PANGU = PanguReader(pangu_url)
    _WORKER_ERA5 = ERA5Reader(era5_url)
    _WORKER_BACKGROUND_VARIABLES = background_variables
    _WORKER_REFERENCE_VARIABLES = reference_variables


def _compute_batch(batch):
    stats = NormStats()
    for init_time, lead_hours, valid_time in batch:
        background = _WORKER_PANGU.read_background(
            init_time, lead_hours, _WORKER_BACKGROUND_VARIABLES
        )
        reference = _WORKER_ERA5.read_analysis(valid_time, _WORKER_REFERENCE_VARIABLES)
        for channel, variable in enumerate(_WORKER_BACKGROUND_VARIABLES):
            stats.update("background", variable, background[channel])
        for channel, variable in enumerate(_WORKER_REFERENCE_VARIABLES):
            stats.update("reference", variable, reference[channel])
    return stats.stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")

    cfg = load_config()
    train = cfg["splits"]["train"]
    pangu = PanguReader(cfg["data"]["pangu"]["url"])
    era5 = ERA5Reader(cfg["data"]["era5"]["url"])
    index = build_sample_index(
        pangu, era5, train["start"], train["end"],
        lead_hours=cfg["time"].get("lead_time_hours"),
        valid_interval_hours=int(cfg["time"]["analysis_interval_hours"]),
    )
    background_variables = list(cfg["variables"]["background_variables"])
    reference_variables = list(cfg["variables"]["era5_reference_variables"])
    if len(background_variables) != len(reference_variables):
        raise ValueError("background and ERA5 reference variable lists must have the same length")

    batches = []
    for start in range(0, len(index), args.batch_size):
        rows = index.iloc[start:start + args.batch_size]
        batches.append([
            (str(row["init_time"]), int(row["lead_time_hours"]), str(row["valid_time"]))
            for _, row in rows.iterrows()
        ])

    stats = NormStats()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(
            cfg["data"]["pangu"]["url"],
            cfg["data"]["era5"]["url"],
            background_variables,
            reference_variables,
        ),
    ) as executor:
        future_sizes = {
            executor.submit(_compute_batch, batch): len(batch)
            for batch in batches
        }
        for future in as_completed(future_sizes):
            batch_stats = NormStats()
            batch_stats.stats = future.result()
            stats.merge(batch_stats)
            completed += future_sizes[future]
            print(f"completed {completed}/{len(index)} training cycles", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.save(args.output)
    metadata = {
        "split": "train",
        "interval": {"start_inclusive": train["start"], "end_exclusive": train["end"]},
        "sample_count": int(len(index)),
        "lead_time_policy": ("shortest_available" if cfg["time"].get("lead_time_hours") is None else "fixed"),
        "lead_time_hours_used": sorted(int(x) for x in index["lead_time_hours"].unique()),
        "valid_interval_hours": int(cfg["time"]["analysis_interval_hours"]),
        "background_variables": background_variables,
        "reference_variables": reference_variables,
        "grid_shape": [121, 240],
        "statistics": "unweighted finite grid points; sample std uses ddof=1",
        "validation_and_test_used": False,
        "statistics_file": args.output.name,
        "workers": args.workers,
        "batch_size": args.batch_size,
    }
    metadata_path = args.output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"statistics: {args.output}")
    print(f"metadata:   {metadata_path}")
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
