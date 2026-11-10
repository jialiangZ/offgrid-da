"""Run data acceptance, trial and the matched DA/control experiments, failing at gates."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offgrid_da.data.arches_training import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/da/single_cycle_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_da_02_t2m_dim32_main")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = args.output / "config.json"
    if frozen.exists() and frozen.read_bytes() != args.config.read_bytes():
        raise RuntimeError("Experiment configuration is frozen; choose a different output for a new recipe")
    if not frozen.exists():
        frozen.write_bytes(args.config.read_bytes())
    cfg = json.loads(frozen.read_text())
    paths = [ROOT / "src/offgrid_da/data/arches_training.py", ROOT / "src/offgrid_da/data/arches_dataset.py",
             ROOT / "src/offgrid_da/data/dataset.py", ROOT / "src/offgrid_da/models/analysis.py",
             ROOT / "src/offgrid_da/models/observation_encoder.py", ROOT / "scripts/train_single_cycle.py",
             ROOT / "scripts/prepare_single_cycle.py", Path(__file__)]
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    fingerprint_path = args.output / "implementation.json"
    if fingerprint_path.exists() and json.loads(fingerprint_path.read_text()) != provenance:
        raise RuntimeError("Experiment implementation changed; use a new experiment name")
    atomic_json(fingerprint_path, provenance)
    snapshots = args.output / "source_snapshot"
    for path in paths:
        dest = snapshots / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
    for suffix in ("train_2018_2020", "val_2021"):
        metadata = json.loads((ROOT / f"outputs/arches_backgrounds_{suffix}/manifest.json").read_text())
        if metadata["checkpoint_sha256"] != cfg["forecast_checkpoint_sha256"]:
            raise RuntimeError("Unexpected forecast checkpoint")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["PYTHONUNBUFFERED"] = "1"
    status = {"started_unix": time.time(), "stages": {}, "test_set_used": False}

    def run(stage, command):
        for path in paths:
            if digest(path) != provenance[str(path.relative_to(ROOT))]:
                raise RuntimeError("Implementation changed between stages")
        status["current_stage"] = stage
        status["stages"][stage] = "running"
        atomic_json(args.output / "status.json", status)
        print("START", stage, flush=True)
        with (args.output / f"{stage}.log").open("ab", buffering=0) as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status["stages"][stage] = "complete"
        atomic_json(args.output / "status.json", status)

    python = sys.executable
    try:
        run("preparation", [python, str(ROOT / "scripts/prepare_single_cycle.py"),
                            "--workers", str(cfg["prepare_workers"]), "--output", str(args.output)])
        prep = args.output / "preparation.json"
        report = json.loads(prep.read_text())
        if not report["complete"]:
            raise RuntimeError("Incomplete preparation")
        common = [python, str(ROOT / "scripts/train_single_cycle.py"), "--config", str(frozen),
                  "--preparation", str(prep)]
        trial_path = args.output / "trial"
        trial_cmd = common + ["--output", str(trial_path), "--trial", "--kind", "da"]
        if (trial_path / "last.pt").exists() and not (trial_path / "result.json").exists():
            trial_cmd += ["--resume", str(trial_path / "last.pt")]
        if not (trial_path / "result.json").exists():
            run("trial", trial_cmd)
        trial = json.loads((trial_path / "result.json").read_text())
        if not trial["checks"]["empty_exact_identity"] or not trial["checks"]["checkpoint_reload_exact"]:
            raise RuntimeError("Trial did not pass the engineering gates")
        results = {}
        for kind in ("da", "neural_correction"):
            out = args.output / kind
            command = common + ["--output", str(out), "--kind", kind]
            if (out / "last.pt").exists() and not (out / "result.json").exists():
                command += ["--resume", str(out / "last.pt")]
            if not (out / "result.json").exists():
                run(kind, command)
            results[kind] = json.loads((out / "result.json").read_text())
            if results[kind]["selected_validation"]["cycles"] != 1460:
                raise RuntimeError("Formal evaluation must cover all validation cycles")
        baseline = results["da"]["selected_validation"]["baseline0_rmse_K"]
        if abs(baseline - results["neural_correction"]["selected_validation"]["baseline0_rmse_K"]) > 1e-10:
            raise RuntimeError("Controls did not use identical background/reference pairs")
        comparison = {"validation_year": 2021, "validation_cycles": 1460, "test_set_used": False,
                      "baseline0_rmse_K": baseline,
                      "DA_rmse_K": results["da"]["selected_validation"]["rmse_K"],
                      "neural_correction_rmse_K": results["neural_correction"]["selected_validation"]["rmse_K"],
                      "recipe": str(frozen), "notes": "Best checkpoints selected on validation; single seed, no significance claim."}
        atomic_json(args.output / "comparison.json", comparison)
        status["current_stage"] = "complete"
        status["finished_unix"] = time.time()
        atomic_json(args.output / "status.json", status)
        print("ALL COMPLETE", json.dumps(comparison), flush=True)
    except Exception as error:
        status["failed"] = repr(error)
        atomic_json(args.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()

