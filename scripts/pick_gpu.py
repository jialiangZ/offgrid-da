#!/usr/bin/env python3
"""Pick a GPU with enough free memory and print its index (for CUDA_VISIBLE_DEVICES).

Shared-machine helper: other users' jobs appear and grow at any time, so the
caller should re-check before long runs. Usage:

  GPU=$(pick_gpu.py --min-free-mib 6000) || exit 1
  CUDA_VISIBLE_DEVICES=$GPU ...
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-mib", type=int, default=6000)
    ap.add_argument("--prefer", type=int, default=None, help="tie-break preference (still requires the free-memory floor)")
    a = ap.parse_args()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"pick_gpu: nvidia-smi failed: {e}", file=sys.stderr)
        return 1
    gpus = []
    for line in out.strip().splitlines():
        idx, free = (int(x.strip()) for x in line.split(","))
        gpus.append((idx, free))
    ok = [(i, f) for i, f in gpus if f >= a.min_free_mib]
    if not ok:
        print("pick_gpu: no GPU with >= %d MiB free; state: %s" % (a.min_free_mib, gpus), file=sys.stderr)
        return 1
    ok.sort(key=lambda x: (-x[1], 0 if x[0] == a.prefer else 1, x[0]))
    print(ok[0][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
