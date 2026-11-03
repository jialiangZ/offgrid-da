#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/run_20y"; mkdir -p "$LOG_DIR"
MONITOR_LOG="$LOG_DIR/gpu_wait_2cards.log"; LOCK_FILE="$LOG_DIR/.gpu_wait_2cards.lock"
TRAIN_LOG="$LOG_DIR/train_20y_6h_$(date +%Y%m%d_%H%M%S).log"
POLL_SECONDS="${POLL_SECONDS:-60}"; STABLE_POLLS="${STABLE_POLLS:-3}"
MAX_USED_MIB="${MAX_USED_MIB:-512}"; MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-5}"
CHECK_ONCE=0; if [[ "${1:-}" == "--check-once" ]]; then CHECK_ONCE=1; fi
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"; }
get_idle_gpus() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v mm="$MAX_USED_MIB" -v mu="$MAX_UTIL_PERCENT" '{ gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); if (($2+0)<=mm && ($3+0)<=mu) print $1 }'; }
if (( CHECK_ONCE )); then mapfile -t idle_gpus < <(get_idle_gpus); printf 'Idle GPU IDs (memory <= %s MiB, utilization <= %s%%):' "$MAX_USED_MIB" "$MAX_UTIL_PERCENT"; if ((${#idle_gpus[@]})); then printf ' %s' "${idle_gpus[@]}"; else printf ' none'; fi; printf '\n'; exit 0; fi
exec 9>"$LOCK_FILE"; if ! flock -n 9; then echo "A GPU monitor is already running." >&2; exit 1; fi
TARGET_DIR="$ROOT/geoarches/modelstore/archesweather-m-20y-6h-seed0"
if [[ -e "$TARGET_DIR" ]]; then log "Refusing to overwrite existing experiment directory: $TARGET_DIR"; exit 2; fi
if pgrep -af '[p]ython.*-m geoarches.main_hydra' >/dev/null; then log "Refusing to start: an existing Geoarches training process was found."; exit 2; fi
log "Monitoring every ${POLL_SECONDS}s; requiring ${STABLE_POLLS} consecutive idle checks."
log "Idle means memory <= ${MAX_USED_MIB} MiB and utilization <= ${MAX_UTIL_PERCENT}%."
candidate=""; stable_count=0
while true; do
  mapfile -t idle_gpus < <(get_idle_gpus)
  if ((${#idle_gpus[@]} >= 2)); then
    pair="${idle_gpus[0]},${idle_gpus[1]}"
    if [[ "$pair" == "$candidate" ]]; then ((stable_count += 1)); else candidate="$pair"; stable_count=1; fi
    log "Candidate GPUs: $pair; stable check ${stable_count}/${STABLE_POLLS}."
    if (( stable_count >= STABLE_POLLS )); then
      mapfile -t current_idle < <(get_idle_gpus); valid=1; IFS=',' read -r -a selected_gpus <<< "$candidate"
      for gpu in "${selected_gpus[@]}"; do if ! printf '%s\n' "${current_idle[@]}" | grep -qx "$gpu"; then valid=0; fi; done
      if (( valid )) && ! pgrep -af '[p]ython.*-m geoarches.main_hydra' >/dev/null && [[ ! -e "$TARGET_DIR" ]]; then
        log "Conditions met; launching two-GPU training on physical GPUs $candidate. Log: $TRAIN_LOG"
        set +e; CUDA_VISIBLE_DEVICES="$candidate" bash "$ROOT/scripts/run_training_20y.sh" cluster.gpus=2 >>"$TRAIN_LOG" 2>&1 &
        train_pid=$!; printf '%s\n' "$train_pid" > "$LOG_DIR/train_20y_6h.pid"; log "Training PID=$train_pid."
        wait "$train_pid"; train_status=$?; set -e
        log "Training exited with status $train_status. Monitor will stop and will not relaunch."; exit "$train_status"
      fi
      log "Pre-launch check failed (GPU state changed, training started, or target directory appeared); waiting again."
      candidate=""; stable_count=0
    fi
  else
    if [[ -n "$candidate" ]]; then log "Only ${#idle_gpus[@]} GPU(s) are idle; resetting stability count."; fi
    candidate=""; stable_count=0; log "Idle GPU count=${#idle_gpus[@]}; waiting for two."
  fi
  sleep "$POLL_SECONDS"
done
