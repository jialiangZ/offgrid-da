#!/usr/bin/env bash
# Sequentially run the remaining ArchesWeather training recipe on GPUs 0 and 1:
#   Stage A  continue300k: 250k -> 300k single-step, recent2 (paper recipe tail)
#   Stage B  arft20k:      300k -> 320k two-step autoregressive fine-tuning
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/geoarches"
PY="$REPO/.venv/bin/python"
NAME_A="archesweather-m-20y-6h-seed0-continue300k"
NAME_B="archesweather-m-20y-6h-seed0-arft20k"
LOGDIR="$ROOT/logs/continue300k_ar"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$LOGDIR"
STATUS="$LOGDIR/status.txt"
LOCK="$LOGDIR/launcher.lock"
exec 9>"$LOCK"
if ! flock -n 9; then echo "Another 300k/AR launcher holds the lock" >&2; exit 1; fi
if pgrep -af '[p]ython.*-m geoarches.main_hydra' >/dev/null; then
  echo "Refusing to start: an existing Geoarches training process was found." | tee -a "$STATUS" >&2
  exit 2
fi
# hydra resolves exp_dir=modelstore/<name> relative to the current directory;
# all training must run from the geoarches checkout.
cd "$REPO"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"; }

# NOTE: when resuming an existing experiment dir, main_hydra replaces the
# module/dataloader sections from the exp dir's config.yaml and then re-applies
# ONLY '+ '/'++'-prefixed CLI overrides. Every module/dataloader override below
# therefore uses the ++ form; plain overrides would be silently dropped.
COMMON_ARGS=(
  dataloader=era5 module=archesweather
  ++module.train.rollout_iterations=1 ++module.val.rollout_iterations=2
  ++module.val.metrics_kwargs.rollout_iterations=2 ++module.train.metrics_kwargs.rollout_iterations=1
  ++dataloader.dataset.multistep=1 ++dataloader.dataset.domain=recent2
  "++dataloader.dataset.path=$ROOT/data/era5_240/full"
  ++dataloader.dataset.lead_time_hours=6 ++dataloader.test_args.domain=test
  ++dataloader.dataset.norm_scheme=train_stats
  "++dataloader.dataset.norm_stats_path=$ROOT/stats/era5_norm_stats_1998_2017_lead6h.pt"
  "++module.module.norm_stats_path=$ROOT/stats/era5_norm_stats_1998_2017_lead6h.pt"
  "++module.backbone.gradient_checkpointing=true"
  seed=0 log=true resume=true
  cluster.batch_size=1 cluster.cpus=3 cluster.gpus=2 cluster.precision=bf16-mixed cluster.swanlab_mode=offline
  val_check_interval=1.0 check_val_every_n_epoch=1
  "++dataloader.dataset.switch_recent_data_after_steps=125000"
)

run_stage() {
  local name="$1" log="$2"; shift 2
  log "Launching $name on GPUs $CUDA_VISIBLE_DEVICES"
  if "$PY" -m geoarches.main_hydra "${COMMON_ARGS[@]}" "name=$name" "$@" \
      > "$log" 2>&1; then rc=0; else rc=$?; fi
  log "$name exit_code=$rc"
  return "$rc"
}

log "=== 250k->300k + AR fine-tune pipeline start (GPUs $CUDA_VISIBLE_DEVICES) ==="

"$PY" "$ROOT/scripts/prepare_continuation_300k_ar.py" 2>&1 | tee -a "$STATUS"

# Stage A: 250k -> 300k.
if [[ ! -s "$REPO/modelstore/$NAME_A/checkpoints/checkpoint_global_step=250000.ckpt" ]]; then
  log "Stage A checkpoint was not prepared; aborting"
  exit 3
fi
run_stage "$NAME_A" "$LOGDIR/stageA_250k_300k.log" \
  "++module.module.lr=1.1691209544357965e-5" ++module.module.num_warmup_steps=0 \
  ++module.module.num_training_steps=50000 max_steps=300000 save_step_frequency=25000
rc_a=$?
if (( rc_a != 0 )); then log "Stage A failed (rc=$rc_a); not starting AR stage"; exit "$rc_a"; fi

# Stage B: prepare (needs the fresh 300k checkpoint), then 300k -> 320k.
"$PY" "$ROOT/scripts/prepare_continuation_300k_ar.py" 2>&1 | tee -a "$STATUS"
if [[ ! -s "$REPO/modelstore/$NAME_B/checkpoints/checkpoint_global_step=300000.ckpt" ]]; then
  log "Stage B checkpoint was not prepared; aborting"
  exit 4
fi
run_stage "$NAME_B" "$LOGDIR/stageB_arft20k.log" \
  ++dataloader.dataset.multistep=2 ++module.train.rollout_iterations=2 \
  ++module.train.metrics_kwargs.rollout_iterations=2 \
  ++module.module.increase_multistep_period=10000 \
  "++module.module.lr=2.5e-6" ++module.module.num_warmup_steps=0 \
  ++module.module.num_training_steps=20000 max_steps=320000 save_step_frequency=10000
rc_b=$?
log "=== pipeline finished (stageA=0, stageB rc=$rc_b) ==="
exit "$rc_b"
