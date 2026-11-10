#!/usr/bin/env bash
# Sequential formal synthetic-obs DA training (seeds 42, 43) on one GPU.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/step2_synthetic_da"
PY="$ROOT/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
mkdir -p "$OUT"
STATUS="$OUT/formal_status.txt"
exec 9>"$OUT/formal.lock"
if ! flock -n 9; then echo "another formal launcher is running" >&2; exit 1; fi
log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"; }
for SEED in 42 43; do
  log "launching seed $SEED formal on GPU $CUDA_VISIBLE_DEVICES"
  if "$PY" "$ROOT/scripts/train_step2_synthetic.py" \
      --config "$OUT/config_seed${SEED}.json" \
      --preparation "$OUT/preparation.json" \
      --output "$OUT/seed${SEED}/da/formal" \
      --kind da > "$OUT/formal_seed${SEED}.log" 2>&1; then
    log "seed $SEED formal exit_code=0"
  else
    log "seed $SEED formal FAILED rc=$?"; exit 1
  fi
done
log "all formal runs complete"
