"""Traverse train/validation once, validating and caching samples without preloading RAM."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offgrid_da.data.arches_training import LazyArchesDataset, atomic_json
from offgrid_da.data.collate import single_cycle_collate

DATASETS = {}


def initialize(root):
    import torch
    torch.set_num_threads(1)
    for split in ("train", "val"):
        DATASETS[split] = LazyArchesDataset(root, split)


def visit(task):
    split, index = task
    dataset = DATASETS[split]
    sample = dataset[index]
    dataset.validate_sample(sample)
    batch = single_cycle_collate([sample])
    for prefix, key in (("obs", "observation_mask"), ("heldout_obs", "heldout_obs_mask")):
        if int(batch[key].sum()) != len(sample[prefix + "_values"]):
            raise ValueError("Wrong observation mask")
    return {"split": split, "index": index, "sample_id": sample["sample_id"],
            "context": len(sample["obs_values"]), "heldout": len(sample["heldout_obs_values"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {s: LazyArchesDataset(ROOT, s) for s in ("train", "val")}
    lengths = {s: len(d) for s, d in datasets.items()}
    tasks = [(s, i) for s, d in datasets.items() for i in range(min(args.limit, len(d)) if args.limit else len(d))]
    results, failures = [], []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(str(ROOT),)) as pool:
        pending = {pool.submit(visit, task): task for task in tasks}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"task": pending[future], "error": repr(error)})
            if (len(results) + len(failures)) % 50 == 0:
                progress = {"success": len(results), "failed": len(failures), "total": len(tasks),
                            "seconds": time.time() - started}
                atomic_json(args.output / "preparation_progress.json", progress)
                print(progress, flush=True)
    report = {"expected": lengths, "visited": len(results), "failures": failures,
              "complete": not args.limit and not failures and len(results) == sum(lengths.values()),
              "signatures": {s: d.signature for s, d in datasets.items()},
              "seconds": time.time() - started, "samples": sorted(results, key=lambda x: (x["split"], x["index"]))}
    atomic_json(args.output / "preparation.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} samples failed; see preparation.json")
    print("PREPARATION COMPLETE", len(results), flush=True)


if __name__ == "__main__":
    main()

