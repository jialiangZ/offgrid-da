"""Phase 1 visualization (plan section 12, Task 4).

For at least 3 randomly chosen analysis times, produce a 6-panel figure:
  1. Pangu background T2M         4. Pangu - ERA5 difference
  2. ERA5 reference T2M           5. METAR - interpolated background (innovation)
  3. METAR station observations   6. innovation histogram

Plus per-time station coverage maps (context vs held-out split) under
outputs/data_audit/observation_maps/ and a global innovation histogram under
outputs/data_audit/innovation_histograms/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.dataset import SingleCycleDADataset  # noqa: E402
from offgrid_da.utils.reproducibility import seed_everything  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"


def plot_one_time(ds: SingleCycleDADataset, idx: int) -> None:
    s = ds[idx]
    bg = s["background"][0].numpy()
    ref = s["reference"][0].numpy()
    lat = s["grid_latitude"].numpy()
    lon = s["grid_longitude"].numpy()
    obs = s["obs_locations"].numpy()
    vals = s["obs_values"].numpy()
    inn = ds.innovations(idx)
    vt = s["valid_time"]

    LON, LAT = np.meshgrid(lon, lat)
    vmin, vmax = 230, 310
    dmax = float(np.nanmax(np.abs(bg - ref)))

    fig, axes = plt.subplots(2, 3, figsize=(19, 8))
    ax = axes[0, 0]
    im = ax.pcolormesh(LON, LAT, bg, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(f"(1) Pangu 6h background T2M [K]\nvalid {vt}")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.pcolormesh(LON, LAT, ref, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title("(2) ERA5 reference T2M [K]")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    sc = ax.scatter(obs[:, 1], obs[:, 0], c=vals, s=2, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"(3) METAR station T2M [K] (n={len(vals)})")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(sc, ax=ax)

    ax = axes[1, 0]
    im = ax.pcolormesh(LON, LAT, bg - ref, cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="auto")
    ax.set_title("(4) Background - ERA5 [K]")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    vmax_i = float(np.nanpercentile(np.abs(inn), 99))
    sc = ax.scatter(obs[:, 1], obs[:, 0], c=inn, s=2, cmap="RdBu_r", vmin=-vmax_i, vmax=vmax_i)
    ax.set_title(f"(5) METAR - interpolated background [K]\ninnovation mean={np.nanmean(inn):+.2f}, std={np.nanstd(inn):.2f}")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(sc, ax=ax)

    ax = axes[1, 2]
    ax.hist(inn[np.isfinite(inn)], bins=61, range=(-15, 15), color="steelblue", edgecolor="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("(6) Innovation histogram [K]")
    ax.set_xlabel("obs - background [K]")

    fig.suptitle(f"OffGrid-DA single-cycle sample {s['sample_id']}", fontsize=13)
    fig.tight_layout()
    out = AUDIT_DIR / "observation_maps" / f"sample_{s['sample_id']}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")

    # innovation histogram per time
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(inn[np.isfinite(inn)], bins=61, range=(-15, 15), color="steelblue", edgecolor="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Innovation (obs - background) {s['sample_id']}")
    ax.set_xlabel("K")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "innovation_histograms" / f"innov_{s['sample_id']}.png", dpi=130)
    plt.close(fig)


def plot_station_split(ds: SingleCycleDADataset) -> None:
    s = ds[0]
    obs = s["obs_locations"].numpy()
    roles = s["obs_station_role"].numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    m = roles == 0
    ax.scatter(obs[m, 1], obs[m, 0], s=3, c="tab:blue", label=f"context (assimilated, n={int(m.sum())})")
    ax.scatter(obs[~m, 1], obs[~m, 0], s=8, c="tab:red", marker="x",
               label=f"held-out (independent eval, n={int((~m).sum())})")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(f"Station split at {s['sample_id']} (deterministic 80/20 by station ID)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = AUDIT_DIR / "observation_maps" / "station_split.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def main(n_times: int = 3) -> None:
    seed_everything(42)
    (AUDIT_DIR / "observation_maps").mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "innovation_histograms").mkdir(parents=True, exist_ok=True)

    ds = SingleCycleDADataset(split="minimal")
    rng = np.random.default_rng(42)
    picks = sorted(rng.choice(len(ds), size=n_times, replace=False).tolist())
    print(f"visualizing samples: {[ds.index['sample_id'].iloc[i] for i in picks]}")
    for i in picks:
        plot_one_time(ds, i)
    plot_station_split(ds)

    # global innovation histogram over the whole window
    cache = PROJECT_ROOT / "outputs" / "cache" / "innovations_jan2020.npy"
    if cache.exists():
        inn = np.load(cache)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(inn, bins=81, range=(-20, 20), color="steelblue", edgecolor="none")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"All Jan-2020 innovations (n={len(inn)}): mean={inn.mean():+.3f} K, std={inn.std():.3f} K")
        ax.set_xlabel("obs - background [K]")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / "innovation_histograms" / "innov_all_jan2020.png", dpi=130)
        plt.close(fig)
        print("  wrote global innovation histogram")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(n)
