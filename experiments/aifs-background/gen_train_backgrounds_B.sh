#!/bin/bash
# Plan-B driver (user-approved 2026-11-16): generate each training year's AIFS
# backgrounds on GPU0 as soon as THAT year's extraction passes its gate, instead
# of waiting for all three years. Mirrors train_year_prep.sh Stage C + assembly
# exactly (same output dirs data/bg_aifs_train_$YEAR, same gates, same assembler),
# so the result is identical to the chain's own Stage C - just earlier.
#
# Pinned to GPU0 via pick_gpu --only 0 (the other cards belong to other users).
# run_aifs_background.py resumes by skipping existing outputs, so restarts are
# safe. Runs as systemd unit aifs-bg-gen.service (Restart=on-failure).
set -u
cd "$(dirname "$0")"
ROOT=/home/jlzhang/Documents/offgrid-da
VENV=$PWD/v1.1/.venv/bin/python
LOG=train_year_prep.log
GPU_MIN_FREE_MIB=21000
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

count() { ls data/aifs_inputs/$1/*.npy.zst 2>/dev/null | wc -l; }

pick0() {  # GPU0 only; poll forever, log every 10th minute
  local i=0 idx
  while :; do
    if idx=$($ROOT/.venv/bin/python "$ROOT/scripts/pick_gpu.py" --only 0 \
             --min-free-mib "$GPU_MIN_FREE_MIB" 2>>"$LOG"); then
      echo "$idx"; return 0
    fi
    i=$((i + 1)); [ $((i % 10)) -eq 0 ] && say "bg-B: GPU0 still not free enough (poll $i)"
    sleep 60
  done
}

gen_year() {  # gen_year <year> <gate_input_files> <gate_min_backgrounds>
  local Y=$1 GIN=$2 GOUT=$3 gpu rc=1 attempt NB
  say "bg-B: waiting for $Y inputs (need >= $GIN, have $(count $Y))"
  while [ "$(count $Y)" -lt "$GIN" ]; do sleep 120; done
  say "bg-B: $Y inputs complete ($(count $Y) files); generating backgrounds on GPU0"
  for attempt in $(seq 1 10); do
    gpu=$(pick0)
    say "bg-B: $Y attempt $attempt on cuda:$gpu"
    CUDA_VISIBLE_DEVICES=$gpu ANEMOI_INFERENCE_NUM_CHUNKS=32 $VENV run_aifs_background.py \
      --inputs "data/aifs_inputs/$Y" --out "data/bg_aifs_train_$Y" >> "bg_gen_$Y.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    say "bg-B: $Y attempt $attempt failed rc=$rc; retry in 120s (resumes where it left off)"
    sleep 120
  done
  if [ $rc -ne 0 ]; then say "bg-B ABORT: $Y generation failed after 10 attempts"; exit 1; fi
  NB=$(ls "data/bg_aifs_train_$Y"/*.npz 2>/dev/null | wc -l)
  say "bg-B: year $Y backgrounds: $NB (min $GOUT)"
  [ "$NB" -ge "$GOUT" ] || { say "bg-B ABORT: $Y backgrounds incomplete ($NB < $GOUT)"; exit 1; }
  [ -f "data/bg_aifs_train_$Y/state_${Y}0101T00.npz" ] \
    || { say "bg-B ABORT: $Y head background missing"; exit 1; }
  say "bg-B: year $Y DONE"
}

# input gates: own stamps + prev-year tails (2019: 1460+2; 2020: 1464+2);
# background gates: same mins as the chain (2020 is a leap year)
gen_year 2018 1460 1460
gen_year 2019 1462 1460
gen_year 2020 1466 1464

say "bg-B final gate: assembling the single_da_12 background set (union must be complete)"
$ROOT/.venv/bin/python "$ROOT/scripts/assemble_aifs_backgrounds.py" >> "$LOG" 2>&1 \
  || { say "bg-B ABORT: assembler failed"; exit 1; }
grep -q '"complete": true' "$ROOT/outputs/single_da_12_aifs_background/backgrounds/manifest.json" \
  || { say "bg-B ABORT: assembled background set incomplete"; exit 1; }
say "TRAINING-YEAR PREP COMPLETE (plan B): inputs + backgrounds for 2018-2020, assembly complete"
