#!/bin/bash
# 2022 independent-period data chain for the da_12 (AIFS-background) confirmation:
# extract 2022 inputs -> gate -> link 2021 tails -> generate bg_aifs_2022 on a
# free whole card. Mirrors train_year_prep.sh Stage B/C conventions; the per-year
# head overlap with 2021 is intentional (the consumer assembles/dedupes).
# Runs as systemd unit aifs-2022.service (Restart=on-failure).
set -u
ROOT=/home/jlzhang/Documents/offgrid-da
cd "$ROOT/experiments/aifs-background"
PY=$ROOT/.venv/bin/python
LOG=train_year_prep.log
GPU_MIN_FREE_MIB=21000
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

if [ ! -f data/bg_aifs_2022/.complete ]; then
  say "2022-chain: extracting 2022 inputs (workers=12)"
  $PY prepare_aifs_inputs.py --year 2022 --workers 12 >> prepare_inputs_2022.log 2>&1
  RC=$?
  N=$(ls data/aifs_inputs/2022/*.npy.zst 2>/dev/null | wc -l)
  say "2022-chain: extraction exit=$RC files=$N"
  [ $RC -ne 0 ] && { say "2022-chain ABORT: extraction failed"; exit 1; }
  [ "$N" -lt 1460 ] && { say "2022-chain ABORT: inputs incomplete ($N < 1460)"; exit 1; }
  for STEM in 2021123112 2021123118; do
    [ -f "data/aifs_inputs/2021/$STEM.npy.zst" ] || { say "2022-chain ABORT: missing 2021 tail $STEM"; exit 1; }
    [ -e "data/aifs_inputs/2022/$STEM.npy.zst" ] || ln "data/aifs_inputs/2021/$STEM.npy.zst" "data/aifs_inputs/2022/$STEM.npy.zst"
  done
  say "2022-chain: inputs complete ($(ls data/aifs_inputs/2022/*.npy.zst | wc -l) incl 2021 tails)"

  say "2022-chain: waiting for a free whole card (>= ${GPU_MIN_FREE_MIB}MiB)"
  while :; do
    G=$($PY "$ROOT/scripts/pick_gpu.py" --min-free-mib "$GPU_MIN_FREE_MIB" 2>>"$LOG") && break
    sleep 60
  done
  say "2022-chain: generating backgrounds on cuda:$G"
  RC=1
  for attempt in $(seq 1 10); do
    CUDA_VISIBLE_DEVICES=$G ANEMOI_INFERENCE_NUM_CHUNKS=32 v1.1/.venv/bin/python \
      run_aifs_background.py --inputs data/aifs_inputs/2022 --out data/bg_aifs_2022 \
      >> bg_gen_2022.log 2>&1 && { RC=0; break; }
    say "2022-chain: gen attempt $attempt failed rc=$?; retry in 120s (resumes)"
    sleep 120
  done
  [ $RC -ne 0 ] && { say "2022-chain ABORT: generation failed after retries"; exit 1; }
  NB=$(ls data/bg_aifs_2022/*.npz 2>/dev/null | wc -l)
  say "2022-chain: backgrounds $NB (min 1460)"
  [ "$NB" -lt 1460 ] && { say "2022-chain ABORT: backgrounds incomplete ($NB < 1460)"; exit 1; }
  touch data/bg_aifs_2022/.complete
fi
say "2022-chain COMPLETE: inputs + $(ls data/bg_aifs_2022/*.npz | wc -l) backgrounds"
