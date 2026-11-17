#!/bin/bash
# P3 formal driver: 4 arms x 43840 steps (da_09-matched budget) on the
# AIFS-background experiment, at most 3 GPUs (GPU3 stays free).
#   lane A (GPU0): da s42          (~9.6 h)
#   lane B (GPU1): da s43          (~9.4 h)
#   lane C (GPU2): nc s42 -> nc s43 (~3.6 h each, sequential)
# Each arm retries with --resume from its own best.pt; run_single_da_12 exits
# early once result.json exists, so restarts of this driver are idempotent.
set -u
cd /home/jlzhang/Documents/offgrid-da
PY=./.venv/bin/python
EXP=outputs/single_da_12_aifs_background
say() { echo "$(date -Is) $*" >> "$EXP/formal_driver.status"; }

run_arm() {  # run_arm <kind> <seed> <gpu>
  local kind=$1 seed=$2 gpu=$3 out rc attempt
  out=$EXP/seed$seed/$kind/formal
  for attempt in $(seq 1 20); do
    local -a extra=()
    if [ -f "$out/best.pt" ] && [ ! -f "$out/result.json" ]; then
      extra=(--resume "$out/best.pt")
    fi
    say "formal $kind s$seed attempt $attempt gpu$gpu resume=${extra[1]:-no}"
    $PY scripts/run_single_da_12.py --kind "$kind" --seed "$seed" --formal --gpu "$gpu" \
      ${extra+"${extra[@]}"} > "$EXP/formal_${kind}_s$seed.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && { say "formal $kind s$seed DONE"; return 0; }
    say "formal $kind s$seed attempt $attempt rc=$rc; retry in 60s"
    sleep 60
  done
  say "formal $kind s$seed FAILED after 20 attempts"; return 1
}

say "formal driver start (3 GPUs; GPU3 free)"
run_arm da 42 0 & A=$!
run_arm da 43 1 & B=$!
{ run_arm neural_correction 42 2 && run_arm neural_correction 43 2; } & C=$!
fail=0
wait $A || fail=1; wait $B || fail=1; wait $C || fail=1
say "ALL FORMAL ARMS DONE fail=$fail"
exit $fail
