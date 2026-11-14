#!/usr/bin/env python3
"""Assemble the single_da_12 (AIFS background) experiment directory.

Hardlinks AIFS background npz shards (train 2018-20 + val 2021, produced by
run_aifs_background.py) into outputs/single_da_12_aifs_background/backgrounds/
and writes a manifest in the full-state format that FullStateInputDataset
consumes (channel_names/order identical to single_da_05). Per-file sha256 is
recorded; existing complete manifests are revalidated instead of rewritten.

Usage: .venv/bin/python scripts/assemble_aifs_backgrounds.py
Prerequisite: bg_aifs_train_{2018,2019,2020} and bg_aifs_2021 state_*.npz dirs.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offgrid_da.data.full_state_data import CHANNELS, SURFACE, LEVEL, PRESSURE  # noqa: E402

AIFS = ROOT / "experiments/aifs-background/data"
EXP = ROOT / "outputs/single_da_12_aifs_background/backgrounds"
TRAIN_INTERVAL = ("2018-01-01T00:00:00Z", "2021-01-01T00:00:00Z")
VAL_INTERVAL = ("2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    dirs = sorted(AIFS.glob("bg_aifs_train_*")) + [AIFS / "bg_aifs_2021"]
    dirs = [d for d in dirs if d.is_dir()]
    assert dirs, "no AIFS background source dirs found"
    EXP.mkdir(parents=True, exist_ok=True)
    manifest_path = EXP / "manifest.json"
    metadata = {"checkpoint_sha256": "6acedcd2baee0e4a9ac6e8a9193b594247305801a26932aca5a3dee49ac3da8c",
                "config_sha256": None, "statistics_sha256": None,
                "scope": "AIFS-single-1.1 6h forecasts as full-state DA backgrounds (P3)",
                "downscale": "N320 -> 0.25 deg -> 7x7 box average -> 240x121 (boxavg convention)",
                "channel_names": CHANNELS, "surface_variables": SURFACE, "level_variables": LEVEL,
                "pressure_hpa": PRESSURE,
                "surface_units": ["m/s", "m/s", "K", "Pa"],
                "level_units": ["m2/s2", "m/s", "m/s", "K", "kg/kg", "Pa/s"],
                "valid_interval": [TRAIN_INTERVAL[0], VAL_INTERVAL[1]],
                "lead_hours": 6, "init_history": "ARCO t-6h,t -> AIFS -> valid t+6h",
                "latitude": "90 to -90", "longitude": "0 to 358.5", "precision": "float32",
                "expected_cycles": 5844, "source_dirs": [str(d) for d in dirs]}
    done = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        done = {s["valid_utc"]: s for s in old.get("samples", [])}
        print(f"existing manifest: {len(done)} samples")
    samples = []
    files = sorted([f for d in dirs for f in d.glob("state_*.npz")])
    assert files, "no state_*.npz in sources"
    for n, f in enumerate(files):
        stamp = f.name.split(".")[0].replace("state_", "")
        valid = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:00:00Z"
        dest = EXP / f"background_{stamp}.npz"
        if valid in done and dest.exists():
            rec = done[valid]
            if sha(dest) != rec["sha256"]:
                raise ValueError(f"changed file for {valid}")
            samples.append(rec)
            continue
        if not dest.exists():
            os.link(f, dest)
        init = (np.datetime64(stamp[:4] + "-" + stamp[4:6] + "-" + stamp[6:8] + "T" + stamp[9:11], "s")
                - np.timedelta64(6, "h"))
        rec = {"valid_utc": valid, "init_utc": np.datetime_as_string(init, unit="s") + "Z",
               "file": dest.name, "sha256": sha(dest),
               "device": "aifs", "precision": "float32"}
        samples.append(rec)
        if (n + 1) % 500 == 0:
            print(f"{n + 1}/{len(files)}", flush=True)
    samples.sort(key=lambda s: s["valid_utc"])
    times = [np.datetime64(s["valid_utc"].replace("Z", ""), "s") for s in samples]
    expected = np.arange(np.datetime64(TRAIN_INTERVAL[0].replace("Z", ""), "s"),
                         np.datetime64(VAL_INTERVAL[1].replace("Z", ""), "s"),
                         np.timedelta64(6, "h"))
    missing = [str(t) for t in set(expected.astype("datetime64[s]")) - set(times)]
    metadata["samples"] = samples
    metadata["complete"] = not missing
    metadata["missing_cycles"] = missing[:20]
    metadata["n_samples"] = len(samples)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(tmp, manifest_path)
    print(f"assembled {len(samples)} backgrounds | complete={not missing} | missing={len(missing)}")
    if missing:
        print("first missing:", missing[:8])


if __name__ == "__main__":
    main()
