#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/geoarches"
NAME="archesweather-m-20y-6h-seed0-continue150k"
LOGDIR="$ROOT/logs/continue150k"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/status.txt"
LOG="$LOGDIR/training.log"
LOCK="$LOGDIR/launcher.lock"
exec 9>"$LOCK"
if ! flock -n 9; then echo "Another continuation launcher holds the lock"; exit 1; fi
if [[ ! -s "$REPO/modelstore/$NAME/checkpoints/checkpoint_global_step=150000.ckpt" || ! -s "$REPO/modelstore/$NAME/config.yaml" ]]; then
  "$REPO/.venv/bin/python" "$ROOT/scripts/prepare_continuation.py" | tee "$LOGDIR/prepare.log"
fi
cd "$REPO"
last_pair=""
count=0
while true; do
  mapfile -t rows < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
  idle=()
  for row in "${rows[@]}"; do
    IFS=, read -r idx util mem <<< "$row"
    if [[ "$util" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ ]] && (( util < 10 && mem < 1536 )); then idle+=("$idx"); fi
  done
  if (( ${#idle[@]} >= 2 )); then pair="${idle[0]},${idle[1]}"; else pair=""; fi
  if [[ -n "$pair" && "$pair" == "$last_pair" ]]; then ((count+=1)); else count=0; last_pair="$pair"; fi
  printf '%s candidate=%s consecutive=%d/5 all_gpus=%s\n' "$(date -Is)" "${pair:-none}" "$count" "${rows[*]}" | tee -a "$STATUS"
  if (( count >= 0 )); then break; fi
  sleep 60
done
printf '%s\n' "$(date -Is) Launching on GPUs $pair" | tee -a "$STATUS"
export CUDA_VISIBLE_DEVICES="$pair"
if "$REPO/.venv/bin/python" -m geoarches.main_hydra \
  dataloader=era5 module=archesweather \
  module.train.rollout_iterations=1 module.val.rollout_iterations=2 \
  module.val.metrics_kwargs.rollout_iterations=2 module.train.metrics_kwargs.rollout_iterations=1 \
  dataloader.dataset.multistep=1 ++dataloader.dataset.domain=recent2 \
  dataloader.dataset.path="$ROOT/data/era5_240/full" \
  dataloader.dataset.lead_time_hours=6 dataloader.test_args.domain=test \
  dataloader.dataset.norm_scheme=train_stats \
  ++dataloader.dataset.norm_stats_path="$ROOT/stats/era5_norm_stats_1998_2017_lead6h.pt" \
  ++module.module.norm_stats_path="$ROOT/stats/era5_norm_stats_1998_2017_lead6h.pt" \
  ++module.backbone.gradient_checkpointing=true \
  module.module.lr=1.1691209544357965e-5 module.module.num_warmup_steps=0 \
  module.module.num_training_steps=100000 \
  name="$NAME" max_steps=250000 seed=0 log=true resume=true \
  cluster.batch_size=1 cluster.cpus=3 cluster.precision=bf16-mixed cluster.swanlab_mode=offline \
  save_step_frequency=30000 val_check_interval=1.0 check_val_every_n_epoch=1 \
  ++dataloader.dataset.switch_recent_data_after_steps=125000 \
  > "$LOG" 2>&1; then rc=0; else rc=$?; fi
printf '%s exit_code=%d\n' "$(date -Is)" "$rc" | tee -a "$STATUS"
exit "$rc"