#!/usr/bin/env python3
"""single_da_11: multi-scale neighbor schedule (structure experiment).

Motivation (step-3 diagnostics): the effective receptive field is
min(radius_km, distance to the 80th-nearest station); in dense regions the
neighbor cap truncates the field to ~400 km, leaving the 300-1000 km band
without far-station information — exactly where the local Gaussian baseline
(300 km kernel with tails) still earns gain and the network earns none
(step3_spatial: 300-600 km real −1.4%/syn −1.4%, >600 km ≈ 0).

Intervention: per-layer radius/neighbor schedules —
  layer 0:  400 km / 80 neighbors  (local, fine)
  layer 1:  700 km / 48
  layer 2: 1000 km / 32
  layer 3: 1000 km / 24            (bridging, coarse)
Search-time change only: parameter shapes identical to single_da_09, so the
same seed yields the same paired initialization (verified in tests).

Matched training via the standard trainer CLI (train_error_balanced.py),
same data/budget/loss as single_da_09. Trial (256 steps) then formal.

Usage: run_single_da_11.py [--seed 42] [--formal] [--gpu N]
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

RADIUS_SCHEDULE = [400.0, 700.0, 1000.0, 1000.0]
NEIGHBORS_SCHEDULE = [80, 48, 32, 24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--formal", action="store_true", help="run formal training (else trial)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    out_root = ROOT / "outputs/single_da_11_multiscale"
    base_cfg = json.loads((ROOT / "outputs/single_da_09_full82_loss_balanced/seed42/da/formal/config.json").read_text())

    kind = "formal" if args.formal else "trial"
    target = out_root / f"seed{args.seed}" / "da" / kind
    target.mkdir(parents=True, exist_ok=True)
    if (target / "result.json").exists():
        print("already complete:", target)
        return

    cfg = dict(base_cfg)
    cfg["seed"] = args.seed
    cfg["name"] = f"multiscale_{args.seed}_da"
    cfg["model"] = dict(base_cfg["model"], radius_schedule=RADIUS_SCHEDULE,
                        neighbors_schedule=NEIGHBORS_SCHEDULE)
    (target / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")

    # preparation with recomputed signatures (source files were touched by the
    # 2026-11-03 renames; same evidence chain as step2/single_da_10)
    from run_full_update_v1 import Data
    from offgrid_da.data.full_state_data import digest
    stats_path = ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json"
    prep = {"complete": True,
            "signatures": {s: Data(ROOT, s).signature for s in ("train", "val")},
            "loss_statistics_sha256": digest(stats_path),
            "base_preparation": str(ROOT / "outputs/single_da_09_full82_loss_balanced/preparation.json"),
            "note": "Signatures recomputed for single_da_11 (multi-scale schedule); code differs "
                    "from the single_da_09 snapshot only by outputs-directory renames (see "
                    "outputs/step2_synthetic_da/preparation.json) and the additive, "
                    "parameter-shape-neutral schedule options in analysis.py."}
    (out_root / "preparation.json").write_text(json.dumps(prep, indent=1) + "\n")

    cmd = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/train_error_balanced.py"),
           "--config", str(target / "config.json"),
           "--preparation", str(out_root / "preparation.json"),
           "--output", str(target), "--kind", "da"] + (["--trial"] if kind == "trial" else [])
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc == 0:
        result = json.loads((target / "result.json").read_text())
        result.update(radius_schedule=RADIUS_SCHEDULE, neighbors_schedule=NEIGHBORS_SCHEDULE)
        (target / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
