from pathlib import Path
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'geoarches'
OLD = REPO / 'modelstore/archesweather-m-20y-6h-seed0'
NAME = 'archesweather-m-20y-6h-seed0-continue150k'
NEW = REPO / 'modelstore' / NAME
SOURCE = OLD / 'checkpoints/checkpoint_global_step=150000.ckpt'
DEST = NEW / 'checkpoints/checkpoint_global_step=150000.ckpt'
STATS = ROOT / 'stats/era5_norm_stats_1998_2017_lead6h.pt'
if not SOURCE.is_file() or not STATS.is_file():
    raise SystemExit('Missing source checkpoint or train normalization statistics')
if DEST.exists() and (NEW / 'config.yaml').is_file():
    print('Continuation checkpoint and config already prepared; leaving them unchanged')
    raise SystemExit(0)
if DEST.exists():
    raise SystemExit('Refusing to overwrite incomplete continuation checkpoint')

cfg = OmegaConf.load(OLD / 'config.yaml')
cfg.name = NAME
cfg.exp_dir = 'modelstore/' + NAME
cfg.resume = False
cfg.max_steps = 250000
cfg.save_step_frequency = 30000
cfg.val_check_interval = 1.0
cfg.check_val_every_n_epoch = 1
cfg.dataloader.dataset.domain = 'recent2'
cfg.dataloader.dataset.multistep = 1
cfg.dataloader.dataset.lead_time_hours = 6
cfg.dataloader.dataset.norm_scheme = 'train_stats'
cfg.dataloader.dataset.norm_stats_path = str(STATS)
cfg.dataloader.dataset.switch_recent_data_after_steps = 125000
cfg.module.module.norm_stats_path = str(STATS)
cfg.module.module.lr = 1.1691209544357965e-5
cfg.module.module.num_warmup_steps = 0
cfg.module.module.num_training_steps = 100000
cfg.module.train.rollout_iterations = 1
cfg.module.train.metrics_kwargs.rollout_iterations = 1
cfg.module.val.rollout_iterations = 2
cfg.module.val.metrics_kwargs.rollout_iterations = 2
cfg.cluster.batch_size = 1
cfg.cluster.cpus = 3
cfg.cluster.precision = 'bf16-mixed'
cfg.cluster.swanlab_mode = 'offline'
cfg.log = True

checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=False)
if checkpoint.get('global_step') != 150000 or not checkpoint.get('optimizer_states') or not checkpoint.get('lr_schedulers'):
    raise SystemExit('Checkpoint does not contain the expected step/optimizer/scheduler state')
start_lr = float(checkpoint['optimizer_states'][0]['param_groups'][0]['lr'])
if abs(start_lr - 1.1691209544357965e-5) > 1e-10:
    raise SystemExit('Unexpected checkpoint learning rate: ' + str(start_lr))
for group in checkpoint['optimizer_states'][0]['param_groups']:
    group['lr'] = start_lr
    group['initial_lr'] = start_lr
scheduler = checkpoint['lr_schedulers'][0]
scheduler['base_lrs'] = [start_lr] * len(checkpoint['optimizer_states'][0]['param_groups'])
scheduler['last_epoch'] = 0
scheduler['_step_count'] = 1
scheduler['_last_lr'] = [start_lr] * len(checkpoint['optimizer_states'][0]['param_groups'])
NEW.mkdir(parents=True, exist_ok=True)
(NEW / 'checkpoints').mkdir(exist_ok=True)
OmegaConf.save(cfg, NEW / 'config.yaml', resolve=True)
torch.save(checkpoint, DEST)
print('Prepared ' + str(DEST))
print('global_step=%s; initial_lr=%.12g; scheduler_steps=100000; target_global_step=250000' % (checkpoint.get('global_step'), start_lr))