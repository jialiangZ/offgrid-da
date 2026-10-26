"""Phase 1 data preparation: iterate the minimal window, aggregate statistics.

Outputs:
    outputs/data_audit/sample_statistics.csv   (per-cycle obs counts + innovations)
    outputs/cache/norm_stats_jan2020.json      (window normalization stats)
    outputs/cache/station_split_jan2020.json   (context / held-out station pools)
    outputs/data_audit/innovation_histograms/  (global innovation histogram)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.dataset import SingleCycleDADataset  # noqa: E402
from offgrid_da.utils.normalization import NormStats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "innovation_histograms").mkdir(parents=True, exist_ok=True)

    ds = SingleCycleDADataset(split="minimal")
    print(f"samples: {len(ds)}, stations context={len(ds.context_stations)} heldout={len(ds.heldout_stations)}")

    stats = NormStats()
    rows = []
    innovations_all = []
    lat_weights = np.cos(np.deg2rad(ds.pangu.grid()["latitude"]))

    for i in range(len(ds)):
        s = ds[i]
        bg = s["background"][0].numpy().astype(np.float64)
        ref = s["reference"][0].numpy().astype(np.float64)
        # area-weighted background RMSE vs ERA5 for this cycle
        err2 = (bg - ref) ** 2
        w = lat_weights[:, None]
        bg_rmse = float(np.sqrt(np.nansum(w * err2) / np.nansum(w)))

        stats.update("background", "2m_temperature", bg)
        stats.update("reference", "2m_temperature", ref)

        inn = ds.innovations(i)
        innovations_all.append(inn[np.isfinite(inn)])
        qc = s.get("qc_report", {})
        rows.append(
            {
                "sample_id": s["sample_id"],
                "valid_time": s["valid_time"],
                "n_obs": int(s["obs_values"].shape[0]),
                "qc_raw_reports": qc.get("n_raw_reports", np.nan),
                "qc_after_window": qc.get("n_after_time_window", np.nan),
                "qc_after_received": qc.get("n_after_received_cutoff", np.nan),
                "qc_after_location": qc.get("n_after_location", np.nan),
                "qc_after_dd": qc.get("n_after_dd_flag", np.nan),
                "qc_after_dedup": qc.get("n_after_dedup", np.nan),
                "innov_mean_k": float(np.nanmean(inn)) if len(inn) else np.nan,
                "innov_std_k": float(np.nanstd(inn)) if len(inn) else np.nan,
                "innov_rmse_k": float(np.sqrt(np.nanmean(inn**2))) if len(inn) else np.nan,
                "bg_era5_rmse_k": bg_rmse,
                "obs_dt_min_mean": float(s["obs_time_offsets"].mean()) if len(s["obs_time_offsets"]) else np.nan,
                "obs_dt_min_max": float(s["obs_time_offsets"].max()) if len(s["obs_time_offsets"]) else np.nan,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(ds)}")

    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "sample_statistics.csv", index=False)
    stats.save(CACHE_DIR / "norm_stats_jan2020.json")
    with open(CACHE_DIR / "station_split_jan2020.json", "w") as f:
        json.dump(
            {
                "split_name": ds.station_split_name,
                "context_stations": sorted(ds.context_stations),
                "heldout_stations": sorted(ds.heldout_stations),
            },
            f,
        )

    inn_all = np.concatenate(innovations_all)
    np.save(CACHE_DIR / "innovations_jan2020.npy", inn_all)

    print("\n== summary ==")
    print(f"n_obs per cycle: min={df.n_obs.min()} mean={df.n_obs.mean():.0f} max={df.n_obs.max()}")
    print(f"innovation (obs-bg) K: mean={inn_all.mean():.3f} std={inn_all.std():.3f} rmse={np.sqrt((inn_all**2).mean()):.3f}")
    print(f"area-weighted Pangu 6h vs ERA5 T2M RMSE: mean={df.bg_era5_rmse_k.mean():.3f} K (range {df.bg_era5_rmse_k.min():.2f}..{df.bg_era5_rmse_k.max():.2f})")
    print(f"norm stats: {json.dumps(stats.to_dict())}")
    print(f"written: {AUDIT_DIR/'sample_statistics.csv'}")


if __name__ == "__main__":
    main()
