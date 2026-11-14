#!/usr/bin/env python3
"""Independent-period (2022 test year) confirmation for single_da_09 vs single_da_11.

Never touched by any training/selection (train 2018-20, selection val 2021).
Groups: background / da_09 s42,s43 / da_11 s42,s43, real METAR observations,
stride 4 -> 365 cycles, matched protocol with step3/step4/step5 conventions:

  - global cos-lat-weighted RMSE (82ch error-balanced score + surface channels)
  - distance-to-nearest-observation bins [0,300) [300,600) [600,1000) [1000,inf) km
  - station-position stratification (train vs heldout role)

Test-year data path: full-state 2022 backgrounds (outputs/fullstate_backgrounds_test_2022)
+ local era5_240 2022 shards as reference + on-demand multivar METAR tokens
(same cache namespace as train/val). Models load via step3_obs_content_diag.load_model
(structure hyper-params recovered from sibling config.json — EVAL_BUG_NOTE fix).

Anchor mode (--year 2021) re-runs the same code over the validation slice to
reproduce the known step4/step5 anchors (da_09 s42 T2M ~0.4122, da_11 s42 ~0.4075)
before trusting 2022 numbers.

Outputs: outputs/step6_independent_2022/{results.json, report.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import hashlib

from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402
from offgrid_da.data.full_state_data import CHANNELS, SURFACE, LEVEL, PRESSURE, read_state, digest  # noqa: E402
from offgrid_da.data.metar import metar_hour_url  # noqa: E402
from offgrid_da.data.metar_multivar import build_records, pack_records  # noqa: E402
from offgrid_da.data.multivar_dataset import MultivarMetarDataset, multivar_collate  # noqa: E402
from offgrid_da.utils.io import gs_to_https  # noqa: E402
from step3_obs_content_diag import load_model, OBS_KEYS  # noqa: E402
from error_balanced import metrics  # noqa: E402

CH = {"t2m": 2, "u10": 0, "mslp": 3}
BINS = (0., 300., 600., 1000., np.inf)
R_EARTH = 6371.0
GROUPS = ["background", "da09_s42", "da09_s43", "da11_s42", "da11_s43"]
CKPTS = {
    "da09_s42": ROOT / "outputs/single_da_09_full82_loss_balanced/seed42/da/formal/best.pt",
    "da09_s43": ROOT / "outputs/single_da_09_full82_loss_balanced/seed43/da/formal/best.pt",
    "da11_s42": ROOT / "outputs/single_da_11_multiscale/seed42/da/formal/best.pt",
    "da11_s43": ROOT / "outputs/single_da_11_multiscale/seed43/da/formal/best.pt",
}
FULLSTATE = {
    2021: ROOT / "outputs/single_da_05_input82_ablation/backgrounds/manifest.json",
    2022: ROOT / "outputs/fullstate_backgrounds_test_2022/manifest.json",
}


def unit_xyz(lat, lon):
    la = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


class TestYearDataset(torch.utils.data.Dataset):
    """2022 (or 2021 anchor) full-state samples assembled without train/val lazy caches."""

    def __init__(self, year: int):
        self.year = year
        self.mm = MultivarMetarDataset(ROOT, "val")  # obs machinery: scales/reader/token signature/cache
        stats = json.loads((ROOT / "outputs/single_da_05_input82_ablation/normalization.json").read_text())
        self.mean = torch.tensor(stats["input_mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(stats["input_std"], dtype=torch.float32)[:, None, None]
        manifest = json.loads(FULLSTATE[year].read_text())
        if manifest["channel_names"] != CHANNELS:
            raise ValueError("full-state channel order mismatch")
        self.background_dir = FULLSTATE[year].parent
        rows = [r for r in manifest["samples"] if r["valid_utc"].startswith(str(year))]
        self.rows = sorted(rows, key=lambda r: r["valid_utc"])
        self._nc: dict[int, xr.Dataset] = {}
        self.lat = torch.linspace(90, -90, 121)
        self.lon = torch.arange(240, dtype=torch.float64) * 1.5

    def __len__(self):
        return len(self.rows)

    def _era5(self, valid: pd.Timestamp):
        hour = valid.hour
        if hour not in self._nc:
            path = ROOT / f"experiments/archesweather-reproduction/data/era5_240/full/era5_240_{self.year}_{hour}h.nc"
            self._nc[hour] = xr.open_dataset(str(path))
        ds = self._nc[hour].sel(time=np.datetime64(valid)).sortby("latitude", ascending=False).sortby("longitude")
        a = [ds[v].transpose("latitude", "longitude").values for v in SURFACE]
        for v in LEVEL:
            for p in PRESSURE:
                a.append(ds[v].sel(level=p).transpose("latitude", "longitude").values)
        target = torch.tensor(np.stack(a), dtype=torch.float32)
        if not torch.isfinite(target).all():
            raise ValueError(f"nonfinite reference at {valid}")
        return target

    def observation_sample(self, valid: pd.Timestamp):
        """Replica of MultivarMetarDataset.observation_sample for an explicit valid time."""
        path = self.mm.cache / (valid.strftime("%Y%m%dT%H") + ".pt")
        stamps = pd.date_range((valid - pd.Timedelta(minutes=105)).floor("h"), valid.floor("h"), freq="h")
        sources = [ROOT / "outputs/cache" / hashlib.sha1(
            gs_to_https(metar_hour_url(self.mm.reader.base_url, self.mm.reader.layout, t)).encode()).hexdigest()
            for t in stamps]

        def check_sources(sample):
            if len(sample["raw_sha256"]) != len(sources):
                return False
            return all(p.exists() and digest(p) == exp for p, exp in zip(sources, sample["raw_sha256"]))

        if path.exists():
            sample = torch.load(path, map_location="cpu", weights_only=False)
            if sample["token_signature"] != self.mm.token_signature or pd.Timestamp(sample["valid_time"]) != valid:
                raise ValueError("Token provenance mismatch")
            if not check_sources(sample):
                raise ValueError("Raw observation source changed")
            return sample
        import os
        import tempfile
        raw = self.mm.reader.read_window(valid - pd.Timedelta(minutes=60), valid)
        records, report = build_records(raw, valid)
        sample = pack_records(records, valid, self.mm.scales)
        sample.update(qc_report=report, valid_time=valid, token_signature=self.mm.token_signature,
                      raw_sha256=[digest(p) for p in sources])
        for prefix in ("obs", "heldout_obs"):
            for suffix in ("values", "values_normalized", "locations", "time_offsets", "error", "error_normalized"):
                if not torch.isfinite(sample[prefix + "_" + suffix]).all():
                    raise ValueError("Nonfinite observation")
        fd, name = tempfile.mkstemp(prefix=path.name, dir=self.mm.cache)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(sample, stream)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return sample

    def __getitem__(self, index: int):
        row = self.rows[index]
        valid = pd.Timestamp(row["valid_utc"]).tz_convert("UTC").tz_localize(None)
        state82 = torch.from_numpy(read_state(self.background_dir / row["file"], row["sha256"]))
        grid_state = (state82 - self.mean) / self.std
        ref82 = self._era5(valid)
        sample = {
            "background": state82.clone(),
            "reference": ref82,
            "background_normalized": grid_state.clone(),
            "reference_normalized": (ref82 - self.mean) / self.std,
            "grid_state": grid_state.clone(),
            "grid_latitude": self.lat.clone(),
            "grid_longitude": self.lon.clone(),
            "valid_time": valid,
            "sample_id": row["file"],
        }
        sample.update(self.observation_sample(valid))
        return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022, choices=(2021, 2022))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="outputs/step6_independent_2022")
    ap.add_argument("--groups", default="all", help="'all' or comma list from " + ",".join(GROUPS))
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    stats = json.loads((ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json").read_text())
    loss_w, phys_std = stats["loss_weights"], np.array(stats["physical_std"])
    group_names = GROUPS if args.groups == "all" else args.groups.split(",")
    models = {g: load_model(CKPTS[g], device) for g in group_names if g != "background"}
    n_groups = len(group_names)

    ds = TestYearDataset(args.year)
    idxs = list(range(0, len(ds), args.stride))
    print(f"year {args.year}: {len(ds)} cycles, stride {args.stride} -> {len(idxs)} evaluated", flush=True)

    latg, long = np.linspace(90, -90, 121), np.arange(240) * 1.5
    sse_global = np.zeros((n_groups, 82), dtype=np.float64)
    den_global = 0.0
    dist_sse = np.zeros((n_groups, len(BINS) - 1, len(CH)))
    dist_den = np.zeros((len(BINS) - 1, len(CH)))
    st_sse = np.zeros((n_groups, 2, len(CH)))
    st_n = np.zeros(2, dtype=np.int64)

    with torch.inference_mode():
        for n, i in enumerate(idxs):
            sample = ds[i]
            batch = multivar_collate([sample])
            x = batch["background_normalized"].to(device)
            y = batch["reference_normalized"].to(device)
            obs = {k: batch[k].to(device) for k in OBS_KEYS}
            preds = [x] + [models[g](x, obs)["analysis"] for g in group_names if g != "background"]
            preds = torch.stack(preds)
            err_n = (preds.double() - y.double())
            latw = torch.cos(torch.deg2rad(torch.linspace(90, -90, 121, device=device))).clamp_min(0)
            sse_global += (err_n.square() * latw[None, None, None, :, None]).sum((1, 3, 4)).cpu().numpy()
            den_global += float(latw.sum()) * x.shape[-1]
            err = err_n * torch.tensor(phys_std, device=device).double()[None, None, :, None, None]

            locs = batch["obs_locations"][0].numpy().astype(np.float64)
            locs = locs[batch["observation_mask"][0].numpy().astype(bool)]
            tree = cKDTree(unit_xyz(locs[:, 0], locs[:, 1]))
            glat, glon = np.meshgrid(latg, long, indexing="ij")
            chord, _ = tree.query(unit_xyz(glat.ravel(), glon.ravel()), k=1)
            dist_km = 2 * R_EARTH * np.arcsin(np.clip(chord / 2, 0, 1)).reshape(glat.shape)
            bidx = np.digitize(dist_km.ravel(), BINS) - 1
            cosw = np.repeat(np.cos(np.deg2rad(latg))[:, None], len(long), 1).ravel()
            for ci, ch in enumerate(CH.values()):
                e = err[:, 0, ch].reshape(n_groups, -1).cpu().numpy()
                for b in range(len(BINS) - 1):
                    m = bidx == b
                    if m.any():
                        w = cosw[m]
                        dist_sse[:, b, ci] += (e[:, m] ** 2 * w).sum(1)
                        dist_den[b, ci] += w.sum()
            role_locs = [batch["obs_locations"][0].numpy().astype(np.float64),
                         batch["heldout_obs_locations"][0].numpy().astype(np.float64)]
            for r_id, loc_r in enumerate(role_locs):
                if len(loc_r) == 0:
                    continue
                rl, rn = loc_r[:, 0], loc_r[:, 1]
                st_n[r_id] += len(loc_r)
                for ci, ch in enumerate(CH.values()):
                    e = err[:, 0, ch].cpu().numpy()
                    for g in range(n_groups):
                        d = sample_background_at_obs(e[g], latg, long, rl, rn)
                        ok = np.isfinite(d)
                        st_sse[g, r_id, ci] += (d[ok] ** 2).sum()
            if (n + 1) % 40 == 0:
                print(f"{n + 1}/{len(idxs)} cycles", flush=True)

    rmse_global = np.sqrt(sse_global / den_global) * phys_std
    results = {"year": args.year, "cycles": len(idxs), "stride": args.stride, "groups": group_names,
               "channels": list(CH), "bins_km": [list(BINS[:-1]) + ["inf"]],
               "global_surface_rmse": {g: {n: float(rmse_global[gi, c]) for n, c in CH.items()} for gi, g in enumerate(group_names)},
               "global_score": {g: float(metrics(sse_global[gi], den_global, loss_w, phys_std, CHANNELS)["score"])
                                for gi, g in enumerate(group_names)},
               "dist_rmse": {name: np.sqrt(dist_sse[:, :, k] / dist_den[None, :, k]).round(5).tolist()
                             for k, name in enumerate(CH)},
               "station_rmse": {name: {r: np.sqrt(st_sse[:, r, k] / max(st_n[r], 1)).round(5).tolist()
                                       for r, rname in enumerate(("train", "heldout"))} for k, name in enumerate(CH)},
               "station_n": st_n.tolist()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{args.year}.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results["global_surface_rmse"], indent=1), flush=True)


if __name__ == "__main__":
    main()
