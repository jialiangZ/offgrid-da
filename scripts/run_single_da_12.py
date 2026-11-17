#!/usr/bin/env python3
"""single_da_12: AIFS-1.1 backgrounds, matched DA / NC (neural_correction) training.

P3 of the AIFS background-replacement plan. Backgrounds: AIFS-single-1.1 6h
forecasts (boxavg downscale convention); everything else matched to
single_da_09 (loss statistics, budget, data pipeline, paired initialization).

Two reporting axes (per 2026-11-15 directive):
  1. overall gain: DA vs AIFS background (compare against da_09's -1.45% T2M)
  2. observation marginal: DA - NC (compare against da_09's ~-0.8pp T2M)

Usage: run_single_da_12.py --kind {da,neural_correction} --seed {42,43} [--formal] [--gpu N]
Prerequisites: assemble_aifs_backgrounds.py complete (5844) and
prepare_full_state_input.py --experiment outputs/single_da_12_aifs_background
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

EXP = ROOT / "outputs/single_da_12_aifs_background"
AIFS_CKPT_SHA = "6acedcd2baee0e4a9ac6e8a9193b594247305801a26932aca5a3dee49ac3da8c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("da", "neural_correction"), default="da")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None,
                    help="pass-through to the trainer: resume from a checkpoint of this arm")
    args = ap.parse_args()

    bg_manifest = json.loads((EXP / "backgrounds/manifest.json").read_text())
    if not bg_manifest.get("complete"):
        raise SystemExit(f"backgrounds incomplete: {bg_manifest.get('n_samples')} samples")
    if not (EXP / "normalization.json").exists():
        raise SystemExit("run prepare_full_state_input.py --experiment first")

    out_root = EXP
    kind = "formal" if args.formal else "trial"
    target = out_root / f"seed{args.seed}" / args.kind / kind
    target.mkdir(parents=True, exist_ok=True)
    if (target / "result.json").exists():
        print("already complete:", target)
        return

    base_cfg = json.loads((ROOT / "outputs/single_da_09_full82_loss_balanced/seed42/da/formal/config.json").read_text())
    cfg = dict(base_cfg)
    cfg["seed"] = args.seed
    cfg["name"] = f"aifs_bg_{args.seed}_{args.kind}"
    cfg["forecast_checkpoint_sha256"] = AIFS_CKPT_SHA
    cfg["background_source"] = "aifs-single-1.1 (boxavg downscale); see backgrounds/manifest.json"
    (target / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")

    # preparation signatures recomputed with the AIFS experiment dataset
    import run_full_update_v1 as rfu
    from offgrid_da.data.multivar_dataset import MultivarMetarDataset

    class ExpData(rfu.Data):
        def __init__(self, root, split):
            MultivarMetarDataset.__init__(self, root, split, experiment=EXP)
            self.signature = hashlib.sha256((self.signature + Path(rfu.__file__).read_text()).encode()).hexdigest()

    stats_path = ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json"
    prep = {"complete": True,
            "signatures": {s: ExpData(ROOT, s).signature for s in ("train", "val")},
            "loss_statistics_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "base_preparation": str(ROOT / "outputs/single_da_09_full82_loss_balanced/preparation.json"),
            "backgrounds_manifest_sha256": hashlib.sha256((EXP / "backgrounds/manifest.json").read_bytes()).hexdigest(),
            "note": "single_da_12: identical trainer/loss/budget to single_da_09; backgrounds and "
                    "per-channel input statistics from AIFS-single-1.1 (experiment-local normalization)."}
    (out_root / "preparation.json").write_text(json.dumps(prep, indent=1) + "\n")

    cmd = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/train_error_balanced.py"),
           "--config", str(target / "config.json"),
           "--preparation", str(out_root / "preparation.json"),
           "--output", str(target), "--kind", args.kind,
           "--experiment", str(EXP)] + (["--trial"] if kind == "trial" else [])
    if args.resume is not None:
        cmd += ["--resume", str(args.resume)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    print("running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
