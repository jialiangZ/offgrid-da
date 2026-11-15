#!/usr/bin/env python3
"""Verify an AIFS input shard: shape, finiteness, and value ranges cross-checked
against a same-month 2021 shard (sanity, not identity). Exit 1 on failure."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zstd_io

REF = Path(__file__).resolve().parent / "data/aifs_inputs/2021/2021011500.npy.zst"


def main():
    ref = zstd_io.load_array(REF)
    failures = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        try:
            a = zstd_io.load_array(p)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{p.name}: unreadable ({e})")
            continue
        if a.shape != (90, 721, 1440) or a.dtype != np.float32:
            failures.append(f"{p.name}: bad shape/dtype {a.shape} {a.dtype}")
            continue
        if not np.isfinite(a).all():
            failures.append(f"{p.name}: nonfinite values")
            continue
        # channelwise std must be within a factor of ~3 of the same-season reference
        s_new, s_ref = a.std(axis=(1, 2)), ref.std(axis=(1, 2))
        ratio = s_new / np.maximum(s_ref, 1e-9)
        if not ((ratio > 0.3) & (ratio < 3.5)).all():
            bad = int((~((ratio > 0.3) & (ratio < 3.5))).sum())
            failures.append(f"{p.name}: {bad}/90 channels out of range vs 2021 reference")
            continue
        print(f"OK {p.name}: std-ratio range [{ratio.min():.2f}, {ratio.max():.2f}]")
    if failures:
        print("FAILURES:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
