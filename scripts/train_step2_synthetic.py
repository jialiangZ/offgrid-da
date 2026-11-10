"""Step-2 matched arm: noiseless synthetic observation DA training (plan §4).

Reproduces the exact training glue of single_da_09's error-balanced pipeline
(train_error_balanced.py) with corrected artifact paths, and replaces the four
surface observation values (T2M/U10/V10/MSLP) by H(x_ref): the bilinear grid-
to-station sampling of the reference analysis at the SAME station locations,
times, masks and error fields as the real METAR arm. Dewpoint observations
(var id 3) are left real in both arms, so the matched contrast isolates the
four plan variables' values. Noise is zero. Model, seeds, initialization,
schedule, loss and evaluation are identical to the real arm.

Usage (mirrors error_balanced_trainer.main CLI):
  python train_step2_synthetic.py --config <cfg> --preparation <prep.json> \
      --output <dir> --kind da [--trial]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from offgrid_da.data.metar_multivar import SPECS  # noqa: E402
from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402
import run_full_update_v1 as base_runner  # noqa: E402
import error_balanced_trainer as trainer  # noqa: E402
from offgrid_da.data.multivar_dataset import multivar_collate  # noqa: E402
from offgrid_da.data.full_state_data import CHANNELS, digest  # noqa: E402
from offgrid_da.models import CrossAttentionAnalysisModel  # noqa: E402
from error_balanced import balanced_loss, latitude_weights, metrics  # noqa: E402

SYNTH_VARS = (0, 1, 2, 4)  # T2M, U10, V10, MSLP; dewpoint (3) stays real
STATS = ROOT / "outputs/single_da_09_full82_loss_balanced/loss_statistics.json"
PREP = ROOT / "outputs/single_da_09_full82_loss_balanced/preparation.json"


class SyntheticData(base_runner.Data):
    """Same samples as the real arm, with the four surface obs values replaced
    by H(reference). Locations, masks, error fields, quality flags, heldout
    observations and all state fields are untouched."""

    def __getitem__(self, index):
        s = super().__getitem__(index)
        ref = s["reference"].numpy().astype(np.float64)
        lat = s["grid_latitude"].numpy().astype(np.float64)
        lon = s["grid_longitude"].numpy().astype(np.float64)
        var_ids = s["obs_variables"].numpy()
        locs = s["obs_locations"].numpy().astype(np.float64)
        mean = self.base.mean.numpy().astype(np.float64)
        std = self.base.std.numpy().astype(np.float64)
        values = s["obs_values"].clone()
        values_n = s["obs_values_normalized"].clone()
        for v in SYNTH_VARS:
            ch = SPECS[v]["background_channel"]
            m = var_ids == v
            if not m.any():
                continue
            idx = np.where(m)[0]
            h = sample_background_at_obs(ref[ch], lat, lon, locs[m, 0], locs[m, 1])
            finite = np.isfinite(h)
            if not finite.any():
                continue
            good = np.zeros_like(m)
            good[idx[finite]] = True
            mt = torch.from_numpy(good)
            values[mt] = torch.from_numpy(h[finite].astype(np.float32))
            values_n[mt] = torch.from_numpy(
                ((h[finite] - mean[ch]) / std[ch]).astype(np.float32))
        s["obs_values"] = values
        s["obs_values_normalized"] = values_n
        return s


def main():
    stats = json.loads(STATS.read_text())
    assert stats["complete"] and stats["cycles"] == 4384
    cfg_idx = sys.argv.index("--config") + 1
    settings = json.loads(Path(sys.argv[cfg_idx]).read_text())
    assert settings["loss_statistics_sha256"] == digest(STATS)

    trainer.LazyArchesDataset = SyntheticData
    trainer.CachedSamples = lambda d: d
    trainer.single_cycle_collate = multivar_collate
    original_loader = trainer.DataLoader

    def loader(*a, **kw):
        kw.update(num_workers=2, multiprocessing_context="spawn",
                  persistent_workers=True, prefetch_factor=2)
        return original_loader(*a, **kw)

    trainer.DataLoader = loader
    initial = {}

    def factory(kind, cfg):
        torch_model = CrossAttentionAnalysisModel(channels=82, **cfg)
        model = torch_model if kind == "da" else trainer.BackgroundCorrection(torch_model)
        if not initial:
            h = hashlib.sha256()
            for prefix, module in [("input", torch_model.input),
                                   ("blocks", torch_model.blocks),
                                   ("output", torch_model.output)]:
                for name, tensor in module.state_dict().items():
                    h.update((prefix + name).encode())
                    h.update(tensor.numpy().tobytes())
            initial["convolution_sha256"] = h.hexdigest()
        return model

    trainer.make_model = factory
    trainer.training_loss = lambda p, y: balanced_loss(p, y, stats["loss_weights"])

    def evaluate(model, loader, device, std):
        if model:
            model.eval()
        sse = np.zeros(82)
        baseline = np.zeros(82)
        den = 0.0
        cycles = 0
        with torch.inference_mode():
            for batch in loader:
                x, o, y = trainer.pack(batch, device)
                prediction = x if model is None else model(x, o)["analysis"]
                if not torch.isfinite(prediction).all():
                    raise ValueError("Nonfinite evaluation")
                w = latitude_weights(x).double()
                sse += ((prediction.double() - y.double()).square() * w).sum((0, 2, 3)).cpu().numpy()
                baseline += ((x.double() - y.double()).square() * w).sum((0, 2, 3)).cpu().numpy()
                den += float(w.expand(len(x), 1, *x.shape[-2:]).sum())
                cycles += len(x)
        result = metrics(sse, den, stats["loss_weights"],
                         np.array(stats["physical_std"]), CHANNELS)
        bg = metrics(baseline, den, stats["loss_weights"],
                     np.array(stats["physical_std"]), CHANNELS)
        return {**result, "baseline0_score": bg["score"], "baseline": bg,
                "cycles": cycles,
                "selection_metric": "dimensionless training-error-balanced RMSE"}

    trainer.evaluate = evaluate
    trainer.main()
    out = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads((out / "result.json").read_text())
    result.update(initial, loss_statistics_sha256=digest(STATS),
                  synthetic_obs={"variables": list(SYNTH_VARS), "noise": 0.0,
                                 "dewpoint": "real in both arms",
                                 "construction": "bilinear H(x_ref) at identical "
                                                 "station locations/masks; error fields unchanged"})
    from offgrid_da.data.arches_training import atomic_json
    atomic_json(out / "result.json", result)


if __name__ == "__main__":
    main()
