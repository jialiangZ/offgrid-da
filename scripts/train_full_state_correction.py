"""Reuse the frozen v1 optimizer/sampler/evaluation loop for a matched input ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import train_single_cycle as trainer
from offgrid_da.data.full_state_data import FullStateInputDataset, full_state_collate
from single_cycle_controls import FullStateCorrection
from offgrid_da.data.arches_training import atomic_json


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    cfg = json.loads(args.config.read_text())
    mode, experiment = cfg['input_mode'], cfg['experiment_root']
    trainer.LazyArchesDataset = lambda root, split: FullStateInputDataset(root, split, experiment, mode)
    trainer.CachedSamples = lambda dataset: dataset  # The wrapper already reads validated samples on demand.
    trainer.single_cycle_collate = full_state_collate
    original_pack = trainer.pack

    def pack(batch, device):
        x, obs, y = original_pack(batch, device)
        obs['grid_state'] = batch['grid_state'].to(device, non_blocking=True)
        return x, obs, y

    initial_hash = None

    def factory(kind, settings):
        nonlocal initial_hash
        if kind != 'neural_correction':
            raise ValueError('This entry trains background-only correction')
        model = FullStateCorrection(**settings)
        if initial_hash is None:
            h = hashlib.sha256()
            for key, value in model.state_dict().items():
                h.update(key.encode())
                h.update(value.detach().cpu().numpy().tobytes())
            initial_hash = h.hexdigest()
        return model

    trainer.pack = pack
    trainer.make_model = factory
    trainer.main()
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result.update(input_mode=mode, input_slots=82, output_variables=['2m_temperature'],
                  initial_model_sha256=initial_hash, parameters_matched_by_design=True, observation_dependence=False, inactive_attention_parameters=True)
    atomic_json(path, result)


if __name__ == '__main__':
    main()

