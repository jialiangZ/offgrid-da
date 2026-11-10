"""GPU integration test: interrupted/resumed training matches uninterrupted training."""
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicitly assigned GPU required")
def test_interrupted_optimizer_scheduler_and_sampler_resume(monkeypatch, tmp_path):
    root = Path('/home/jlzhang/Documents/offgrid-da')
    sys.path.insert(0, str(root / 'scripts'))
    trainer = importlib.import_module('train_single_cycle')
    from offgrid_da.data.arches_training import LazyArchesDataset
    real = {s: LazyArchesDataset(root, s) for s in ('train', 'val')}
    for dataset in real.values():
        dataset[0]
        dataset[1]

    class TinyDataset:
        def __init__(self, root, split):
            self.dataset = real[split]
            self.signature = 'integration-test-only-' + self.dataset.signature
            self.norm_stats = self.dataset.norm_stats

        def __len__(self):
            return 2

        def sample_path(self, i):
            return self.dataset.sample_path(i)

    # Match the fixture signature to the already validated real cached samples.
    signatures = {s: 'integration-test-only-' + d.signature for s, d in real.items()}
    original_getitem = trainer.CachedSamples.__getitem__

    def fixture_getitem(self, i):
        original_signature = self.signature
        self.signature = self.signature.removeprefix('integration-test-only-')
        try:
            return original_getitem(self, i)
        finally:
            self.signature = original_signature

    monkeypatch.setattr(trainer, 'LazyArchesDataset', TinyDataset)
    monkeypatch.setattr(trainer.CachedSamples, '__getitem__', fixture_getitem)
    cfg = json.loads((root / 'configs/da/single_cycle_v1.json').read_text())
    cfg.update(trial_steps=4, trial_validate_every=2, trial_validation_cycles=2,
               loader_workers=0, log_every_steps=1, save_every_steps=2)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    preparation = tmp_path / 'fixture_preparation.json'
    preparation.write_text(json.dumps({'complete': True, 'signatures': signatures,
                                       'scope': 'two real cached samples per split; integration fixture only'}))

    def run(output, resume=None):
        argv = ['train_single_cycle.py', '--config', str(config), '--preparation', str(preparation),
                '--output', str(output), '--kind', 'da', '--trial']
        if resume:
            argv += ['--resume', str(resume)]
        monkeypatch.setattr(sys, 'argv', argv)
        trainer.main()

    uninterrupted = tmp_path / 'uninterrupted'
    resumed = tmp_path / 'resumed'
    run(uninterrupted)
    original_save = trainer.atomic_checkpoint

    def interrupt_after_save(path, value):
        original_save(path, value)
        if Path(path).name == 'last.pt' and value['step'] == 2:
            raise RuntimeError('simulated interruption')

    monkeypatch.setattr(trainer, 'atomic_checkpoint', interrupt_after_save)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(resumed)
    import swanlab
    swanlab.finish()
    monkeypatch.setattr(trainer, 'atomic_checkpoint', original_save)
    run(resumed, resumed / 'last.pt')
    a = torch.load(uninterrupted / 'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)
    assert a['step'] == b['step'] == 4
    assert a['scheduler'] == b['scheduler']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=1e-5, atol=1e-7)
    for key, state in a['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=1e-5, atol=1e-7)

