"""Prepare checkpoint surgery for the 250k->300k continuation and the 20k-step
autoregressive fine-tuning stage.

Stage A  continue300k: single-step training on recent2 data, 250,000 -> 300,000
         steps. Peak LR 1.1691209544357965e-5 (the peak the 150k->250k stage
         started from; the paper's 300k-step cosine sits near 2e-5 at 250k, same
         order), cosine-decaying to zero over the remaining 50,000 steps. This
         mirrors the paper recipe's recent-period fine-tuning tail.
Stage B  arft20k: autoregressive fine-tuning, 300,000 -> 320,000 steps with a
         two-step rollout (dataset multistep=2). Peak LR 2.5e-6 (~21% of the
         stage-A peak, in line with AR fine-tuning convention), cosine-decaying
         to zero over 20,000 steps. The multistep curriculum
         (increase_multistep_period) is pinned at 10000 epochs so the rollout
         stays at exactly two steps throughout.

Both stages reuse the checkpoint-surgery pattern proven by
prepare_continuation.py: optimizer/scheduler state is rewritten for a fresh
cosine horizon while model weights stay untouched. The script is idempotent:
a fully prepared stage is left unchanged, and stage B is only prepared once
its source checkpoint (stage A at 300,000 steps) exists.
"""
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'geoarches'
STATS = ROOT / 'stats/era5_norm_stats_1998_2017_lead6h.pt'

PREV = 'archesweather-m-20y-6h-seed0-continue150k'
NAME_A = 'archesweather-m-20y-6h-seed0-continue300k'
NAME_B = 'archesweather-m-20y-6h-seed0-arft20k'

PEAK_LR_A = 1.1691209544357965e-5
PEAK_LR_B = 2.5e-6


def surgery(checkpoint: dict, peak_lr: float) -> dict:
    """Reset optimizer/scheduler for a fresh cosine horizon at peak_lr."""
    for group in checkpoint['optimizer_states'][0]['param_groups']:
        group['lr'] = peak_lr
        group['initial_lr'] = peak_lr
    scheduler = checkpoint['lr_schedulers'][0]
    n_groups = len(checkpoint['optimizer_states'][0]['param_groups'])
    scheduler['base_lrs'] = [peak_lr] * n_groups
    scheduler['last_epoch'] = 0
    scheduler['_step_count'] = 1
    scheduler['_last_lr'] = [peak_lr] * n_groups
    return checkpoint


def base_config(source_dir: Path) -> 'OmegaConf':
    cfg = OmegaConf.load(source_dir / 'config.yaml')
    cfg.resume = False
    cfg.dataloader.dataset.lead_time_hours = 6
    cfg.dataloader.dataset.norm_scheme = 'train_stats'
    cfg.dataloader.dataset.norm_stats_path = str(STATS)
    cfg.dataloader.dataset.domain = 'recent2'
    cfg.dataloader.dataset.switch_recent_data_after_steps = 125000
    cfg.module.module.norm_stats_path = str(STATS)
    cfg.module.module.num_warmup_steps = 0
    cfg.module.val.rollout_iterations = 2
    cfg.module.val.metrics_kwargs.rollout_iterations = 2
    cfg.dataloader.validation_args = {'multistep': 2, 'domain': 'val'}
    cfg.dataloader.test_args = {'multistep': 2, 'domain': 'test'}
    cfg.cluster.batch_size = 1
    cfg.cluster.cpus = 3
    cfg.cluster.gpus = 2
    cfg.cluster.precision = 'bf16-mixed'
    cfg.cluster.swanlab_mode = 'offline'
    cfg.log = True
    return cfg


def prepare_stage(source_dir: Path, source_step: int, name: str, peak_lr: float,
                  num_training_steps: int, max_steps: int, save_step_frequency: int,
                  multistep: int, extra=None) -> bool:
    """Prepare one stage; returns True if newly prepared, False if already done."""
    source = source_dir / f'checkpoints/checkpoint_global_step={source_step}.ckpt'
    new = REPO / 'modelstore' / name
    dest = new / f'checkpoints/checkpoint_global_step={source_step}.ckpt'
    if not source.is_file():
        raise SystemExit(f'Missing source checkpoint: {source}')
    if dest.exists() and (new / 'config.yaml').is_file():
        print(f'{name}: already prepared; leaving unchanged')
        return False
    if dest.exists():
        raise SystemExit(f'{name}: refusing to overwrite incomplete checkpoint')

    cfg = base_config(source_dir)
    cfg.name = name
    cfg.exp_dir = 'modelstore/' + name
    cfg.max_steps = max_steps
    cfg.save_step_frequency = save_step_frequency
    cfg.val_check_interval = 1.0
    cfg.check_val_every_n_epoch = 1
    cfg.dataloader.dataset.multistep = multistep
    cfg.module.module.lr = peak_lr
    cfg.module.module.num_training_steps = num_training_steps
    if multistep > 1:
        cfg.module.train.rollout_iterations = multistep
        cfg.module.train.metrics_kwargs.rollout_iterations = multistep
    else:
        cfg.module.train.rollout_iterations = 1
        cfg.module.train.metrics_kwargs.rollout_iterations = 1
    if extra:
        for key, value in extra.items():
            OmegaConf.update(cfg, key, value)
    OmegaConf.resolve(cfg)

    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if checkpoint.get('global_step') != source_step or not checkpoint.get('optimizer_states') or not checkpoint.get('lr_schedulers'):
        raise SystemExit(f'{name}: source checkpoint lacks expected step/optimizer/scheduler state')
    checkpoint = surgery(checkpoint, peak_lr)
    new.mkdir(parents=True, exist_ok=True)
    (new / 'checkpoints').mkdir(exist_ok=True)
    OmegaConf.save(cfg, new / 'config.yaml', resolve=True)
    torch.save(checkpoint, dest)
    print(f'Prepared {dest}')
    print(f'{name}: global_step={source_step}; peak_lr={peak_lr:.12g}; '
          f'scheduler_steps={num_training_steps}; target_global_step={max_steps}; multistep={multistep}')
    return True


def main():
    if not STATS.is_file():
        raise SystemExit('Missing train normalization statistics')
    prev_dir = REPO / 'modelstore' / PREV
    a_dir = REPO / 'modelstore' / NAME_A

    # Stage A: 250k -> 300k, single-step, recent2.
    prepare_stage(
        prev_dir, 250000, NAME_A, PEAK_LR_A,
        num_training_steps=50000, max_steps=300000, save_step_frequency=25000,
        multistep=1,
    )

    # Stage B: 300k -> 320k, two-step autoregressive rollout. Requires stage A
    # to have finished; if its 300k checkpoint is absent, report and exit 0 so
    # the launcher can call this script again after stage A completes.
    src = a_dir / 'checkpoints/checkpoint_global_step=300000.ckpt'
    if not src.is_file():
        print(f'{NAME_B}: source {src} not available yet; prepare it after stage A finishes')
        return
    prepare_stage(
        a_dir, 300000, NAME_B, PEAK_LR_B,
        num_training_steps=20000, max_steps=320000, save_step_frequency=10000,
        multistep=2,
        extra={'module.module.increase_multistep_period': 10000},
    )


if __name__ == '__main__':
    main()
