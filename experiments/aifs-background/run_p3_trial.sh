#!/bin/bash
# P3 trial driver: 4 arms x 256 steps on the AIFS-background experiment.
# GPU budget: at most 3 cards. Arms da_s42 / nc_s42 / da_s43 start in parallel
# on GPU 0/1/2; nc_s43 follows on GPU0. Logs under the experiment dir.
set -u
cd /home/jlzhang/Documents/offgrid-da
PY=./.venv/bin/python
EXP=outputs/single_da_12_aifs_background
say() { echo "$(date -Is) $*" >> "$EXP/trial_driver.status"; }

run() {  # run <kind> <seed> <gpu>
  $PY scripts/run_single_da_12.py --kind "$1" --seed "$2" --gpu "$3" \
    > "$EXP/trial_${1}_s$2.log" 2>&1
  say "arm $1 s$2 rc=$?"
}

say "trial driver start (3 GPUs; 4th arm queues on GPU0)"
run da 42 0 & P1=$!
run neural_correction 42 1 & P2=$!
run da 43 2 & P3=$!
wait $P1 $P2 $P3
run neural_correction 43 0
say "ALL TRIAL ARMS DONE"
