#!/usr/bin/env python3
"""Warm-up pass: materialize the lazy DA sample cache (outputs/da_sample_cache).

Iterates LazyArchesDataset over train(2018-20) + val(2021) so every per-cycle
.pt is built once (reads ArchesWeather backgrounds + WB2 era5/metar from the
cloud, cached under outputs/cache). Required before prepare_multivar_metar.py
(FullStateInputDataset demands a complete cache) and before any DA training.

Run: .venv/bin/python scripts/warmup_da_cache.py [--workers N]
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offgrid_da.data.arches_training import LazyArchesDataset  # noqa: E402


def identity(v):
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(2)

    for split in ("train", "val"):
        t0 = time.time()
        ds = LazyArchesDataset(ROOT, split)
        n = len(ds)
        done = sum(1 for p in ds.cache_paths if p.is_file())
        print(f"{split}: {n} samples, {done} already cached, sig={ds.signature[:12]}", flush=True)
        if done == n:
            print(f"{split}: complete, skipping", flush=True)
            continue
        loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers,
                            collate_fn=identity,
                            multiprocessing_context="spawn" if args.workers else None,
                            persistent_workers=bool(args.workers), prefetch_factor=4 if args.workers else None)
        for i, _ in enumerate(loader):
            if (i + 1) % 200 == 0:
                r = (i + 1 - done) / (time.time() - t0)
                print(f"{split}: {i+1}/{n} (+{r*3600:.0f}/h fresh)", flush=True)
        final = sum(1 for p in ds.cache_paths if p.is_file())
        print(f"{split}: DONE {final}/{n} in {(time.time()-t0)/60:.1f} min", flush=True)
        assert final == n, f"{split} cache incomplete: {final}/{n}"
    print("WARMUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
