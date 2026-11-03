#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/geoarches"
STATS="$ROOT/stats/era5_norm_stats_1998_2017_lead6h.pt"
if [[ ! -f "$STATS" ]]; then
  echo "Missing train statistics: $STATS" >&2
  exit 1
fi
cd "$REPO"
exec .venv/bin/python -m geoarches.main_hydra \
  dataloader=era5 module=archesweather module.val.rollout_iterations=2 module.val.metrics_kwargs.rollout_iterations=2 module.train.rollout_iterations=1 dataloader.dataset.multistep=1 \
  dataloader.dataset.path="$ROOT/data/era5_240/full" \
  dataloader.dataset.lead_time_hours=6 dataloader.test_args.domain=test \
  dataloader.dataset.norm_scheme=train_stats \
  "++dataloader.dataset.norm_stats_path=$STATS" \
  "++module.module.norm_stats_path=$STATS" \
  "++module.backbone.gradient_checkpointing=true" \
  name=archesweather-m-20y-6h-seed0 max_steps=171000 seed=0 log=true resume=false \
  cluster.batch_size=1 cluster.cpus=3 cluster.precision=bf16-mixed cluster.swanlab_mode=offline ++dataloader.dataset.switch_recent_data_after_steps=125000 "$@"
