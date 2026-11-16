#!/usr/bin/env bash
# Training-year (2018-2020) data preparation, per 2026-11-15 conditional approval.
# Gate: P2 full-year recompute completes with integrity checks (this script does NOT
# judge the science gate; P3 training start remains separately gated).
# Flow: P2 gate -> small-shard acceptance (2017/2020 tails + 2 days of 2018 +
# inference smoke) -> continuous year extraction -> per-year background generation.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=/home/jlzhang/Documents/offgrid-da
PY=$ROOT/.venv/bin/python
VENV=$PWD/v1.1/.venv/bin/python
LOG=train_year_prep.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&2; }  # stderr: never captured by $(...) callers

# ---------- GPU selection (shared machine: other users' jobs appear/grow anytime) ----------
GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-21000}  # MEASURED: one inference step holds 18.34 GiB of torch
                                             # allocations and then asks for a further 2.07 GiB
                                             # block => ~20.4 GiB process peak (20890 MiB), so a
                                             # card needs >=21 GiB free.
                                             # ANEMOI_INFERENCE_NUM_CHUNKS plateaus at this level:
                                             # 2/4/8/16/32/64/128/256 all peak at 18.3-19.3 GiB
                                             # (32 is not special, but unchunked C=1 is not viable).
                                             # Consequence on this host: cards carrying a ~3.8 GiB
                                             # neighbour job show only ~20.2 GiB free -> AIFS
                                             # inference cannot fit -> pick_gpu waits for a card
                                             # that is genuinely free.
GPU_WAIT_TRIES=${GPU_WAIT_TRIES:-180}        # poll up to 3 h for a free card before giving up
pick_gpu() {  # pick_gpu [tries] -> echoes a free device index; polls while cards are busy
  local tries=${1:-30} idx n=0
  while [ "$n" -lt "$tries" ]; do
    n=$((n + 1))
    if idx=$($ROOT/.venv/bin/python "$ROOT/scripts/pick_gpu.py" --min-free-mib "$GPU_MIN_FREE_MIB" 2>>"$LOG"); then
      echo "$idx"; return 0
    fi
    [ $((n % 10)) -eq 1 ] && say "no GPU with >=${GPU_MIN_FREE_MIB}MiB free yet (poll ${n}/${tries}); retrying in 60s"
    sleep 60
  done
  return 1
}

run_inference() {  # run_inference <inputs> <out> <tag> [logfile] [wait_tries]; skip-if-exists resume
  local inputs=$1 out=$2 tag=$3 logf=${4:-$LOG} wait=${5:-$GPU_WAIT_TRIES} gpu rc attempt
  for attempt in 1 2 3; do
    gpu=$(pick_gpu "$wait") || { say "$tag: no GPU with >=${GPU_MIN_FREE_MIB}MiB free (waited ${wait}min)"; return 2; }
    say "$tag: attempt $attempt on cuda:$gpu"
    CUDA_VISIBLE_DEVICES=$gpu ANEMOI_INFERENCE_NUM_CHUNKS=32 $VENV run_aifs_background.py \
      --inputs "$inputs" --out "$out" >> "$logf" 2>&1
    rc=$?
    [ $rc -eq 0 ] && return 0
    say "$tag: attempt $attempt failed rc=$rc; re-picking GPU after 120s (resumes)"
    sleep 120
  done
  return 1
}

verify_smoke_rmse() {
$PY - <<'EOF' >> "$LOG" 2>&1 || { say "ABORT: smoke T2M out of boxavg class"; return 1; }
import numpy as np, xarray as xr, glob
files = sorted(glob.glob("data/bg_aifs_smoke_2018/*.npz"))
assert len(files) == 9, f"expected 9 smoke backgrounds, got {len(files)}"
assert any("state_20180101T00" in f for f in files), "2018 head background missing"
sh = xr.open_dataset(f"{__import__('pathlib').Path('.').resolve()}/../archesweather-reproduction/data/era5_240/full/era5_240_2018_0h.nc")
w = np.cos(np.deg2rad(np.linspace(90, -90, 121)))[:, None]
rs = []
for f in files:
    stem = f.split("/")[-1].split(".")[0].replace("state_", "")
    t = np.datetime64(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}T{stem[9:11]}:00:00.000000000")
    d = np.load(f)
    hh = int(stem[9:11]); shard = xr.open_dataset(f"../archesweather-reproduction/data/era5_240/full/era5_240_2018_{hh}h.nc") if hh else sh
    ref = np.flipud(shard.sel(time=t)["2m_temperature"].transpose("latitude", "longitude").values)
    e = d["surface"][2, 0] - ref
    rs.append(float(np.sqrt((e**2 * w).sum() / (w.sum() * 240))))
print("smoke T2M RMSE:", [round(x, 4) for x in rs])
assert np.mean(rs) < 0.65, f"smoke T2M {np.mean(rs):.4f} out of boxavg class (<0.65)"
EOF
}

# ---------- storage budget ----------
FREE_GB=$(df --output=avail -B 1G /home | tail -1 | tr -d ' ')
if [ "$FREE_GB" -lt 1500 ]; then say "ABORT: only ${FREE_GB}G free (<1500G)"; exit 1; fi
say "storage OK: ${FREE_GB}G free (need ~900G)"

# ---------- gate: P2 full-year recompute done + integrity ----------
while ! grep -q "P2 FINAL DONE" p2_final.log 2>/dev/null; do sleep 600; done
say "P2 final chain signalled; checking integrity"
$PY - <<'EOF' >> "$LOG" 2>&1 || { echo "P2 INTEGRITY FAILED"; exit 1; }
import json
r = json.load(open("p2_paired82/results.json"))
n = r["n_pairs"]; probe = {int(k): v for k, v in r["time_probe_t2m_sse"].items()}
t2m_a, t2m_w = r["rmse_aifs"][2], r["rmse_arches"][2]
assert n >= 1450, f"paired count {n} < 1450"
assert probe[0] < 0.2 * min(probe[-6], probe[6]), "time probe not aligned"
assert 0.5 < t2m_a / t2m_w < 1.1, f"t2m ratio out of sanity range {t2m_a/t2m_w:.3f}"
import math
assert all(math.isfinite(v) for v in r["rmse_aifs"] + r["rmse_arches"]), "nonfinite channel rmse"
print(f"P2 integrity OK: n={n}, t2m {t2m_a:.4f} vs {t2m_w:.4f} (ratio {t2m_a/t2m_w:.3f}), probe 0h vs +-6h {probe[0]:.3g}/{probe[-6]:.3g}/{probe[6]:.3g}")
EOF
say "P2 integrity passed"

# ---------- small-shard acceptance ----------
say "STAGE A: small shards (2017 tail x2, 2 days of 2018, 2020 tail x2)"
$PY prepare_aifs_inputs.py --year 2017 --start 2017-12-31T12 --end 2018-01-01T00 >> "$LOG" 2>&1 || { say "ABORT: 2017 tail extraction failed"; exit 1; }
$PY prepare_aifs_inputs.py --year 2018 --start 2018-01-01T00 --end 2018-01-03T00 >> "$LOG" 2>&1 || { say "ABORT: 2018 probe extraction failed"; exit 1; }
$PY prepare_aifs_inputs.py --year 2020 --start 2020-12-31T12 --end 2021-01-01T00 >> "$LOG" 2>&1 || { say "ABORT: 2020 tail extraction failed"; exit 1; }
mv data/aifs_inputs/2017/2017123112.npy.zst data/aifs_inputs/2017/2017123118.npy.zst data/aifs_inputs/2018/ || { say "ABORT: move 2017 tails"; exit 1; }
mv data/aifs_inputs/2020/2020123112.npy.zst data/aifs_inputs/2020/2020123118.npy.zst data/aifs_inputs/2021/ || { say "ABORT: move 2020 tails"; exit 1; }
rmdir data/aifs_inputs/2017 2>/dev/null
$PY verify_shard.py data/aifs_inputs/2018/201{7123112,7123118,8010100,8010106,8010112,8010118,8010200,8010206,8010212,8010218}.npy.zst data/aifs_inputs/2021/20201231{12,18}.npy.zst >> "$LOG" 2>&1 || { say "ABORT: shard verification failed"; exit 1; }
say "shard verification passed; running inference smoke (2018 probe) + 2021 head regeneration"
# Bounded smoke inputs: exactly the 10 probe stamps. The live 2018 input dir grows
# during Stage B, and run_aifs_background iterates EVERY shard in --inputs, so an
# unbounded smoke would generate the whole partial year into the smoke dir and
# then fail verify_smoke_rmse's 9-file assertion.
SMOKE_IN=data/aifs_inputs_smoke
mkdir -p "$SMOKE_IN"
for S in 2017123112 2017123118 2018010100 2018010106 2018010112 2018010118 2018010200 2018010206 2018010212 2018010218; do
  [ -f "$SMOKE_IN/$S.npy.zst" ] || ln "data/aifs_inputs/2018/$S.npy.zst" "$SMOKE_IN/$S.npy.zst" \
    || { say "ABORT: probe shard $S missing from 2018 inputs"; exit 1; }
done
# Data acceptance is independent of GPU availability (shards already verified above).
# On a busy shared host, defer the GPU smoke instead of stalling the network-bound
# extraction: it must still pass before the first background generation (see below).
SMOKE_DEFERRED=0
if run_inference "$SMOKE_IN" data/bg_aifs_smoke_2018 smoke "$LOG" 3; then
  run_inference data/aifs_inputs/2021 data/bg_aifs_2021 "2021-head" \
    || { say "ABORT: 2021 head regeneration failed"; exit 1; }
else
  say "NOTE: GPU busy - inference smoke deferred; extraction proceeds, smoke gates backgrounds"
  SMOKE_DEFERRED=1
fi
if [ "$SMOKE_DEFERRED" = "0" ]; then
  for NEED in state_20210101T00.npz state_20211231T18.npz; do
    [ -f "data/bg_aifs_2021/$NEED" ] || { say "ABORT: bg_aifs_2021 missing $NEED"; exit 1; }
  done
  say "2021 val backgrounds complete (head+tail verified)"
fi
N=$(ls data/bg_aifs_smoke_2018/*.npz 2>/dev/null | wc -l)
say "smoke produced $N backgrounds"
if [ "$SMOKE_DEFERRED" = "1" ]; then
  say "STAGE A data acceptance passed (GPU smoke deferred)"
else
  say "verifying smoke T2M against era5"
  verify_smoke_rmse || exit 1
  say "STAGE A PASSED (small-shard acceptance, incl GPU smoke)"
fi

# ---------- continuous extraction (network phase, runs to completion first) ----------
# Two phases on purpose: extraction is network-bound and must not be blocked by a
# busy GPU; background generation is GPU-bound and runs afterwards.
# Head backgrounds for year Y need Y-1 Dec-31 12Z/18Z inputs, which live in the
# previous year's directory -> hardlink them in.
link_prev_tails() {
  local prev=$1 cur=$2 STEM
  for STEM in "${prev}123112" "${prev}123118"; do
    [ -f "data/aifs_inputs/$prev/$STEM.npy.zst" ] || { say "ABORT: missing $prev tail $STEM"; return 1; }
    [ -e "data/aifs_inputs/$cur/$STEM.npy.zst" ] || ln "data/aifs_inputs/$prev/$STEM.npy.zst" "data/aifs_inputs/$cur/$STEM.npy.zst"
  done
}

for YEAR in 2018 2019 2020; do
  say "STAGE B: extracting $YEAR"
  $PY prepare_aifs_inputs.py --year $YEAR --workers 12 >> "prepare_inputs_$YEAR.log" 2>&1
  RC=$?
  N=$(ls data/aifs_inputs/$YEAR/*.npy.zst 2>/dev/null | wc -l)
  say "year $YEAR extraction exit=$RC files=$N"
  if [ $RC -ne 0 ] || [ "$N" -lt 1460 ]; then say "ABORT: $YEAR incomplete ($N)"; exit 1; fi
  if [ "$YEAR" != "2018" ]; then
    link_prev_tails $((YEAR - 1)) "$YEAR" || { say "ABORT: $YEAR head inputs"; exit 1; }
  fi
done
say "EXTRACTION PHASE COMPLETE (2018-2020)"
# GPU budget on this host is capped (other work may hold cards). EXTRACT_ONLY=1
# stops here; combine with GPU_MIN_FREE_MIB=999999 to also ensure the Stage A
# smoke defers instead of touching a card that belongs to someone else.
if [ "${EXTRACT_ONLY:-0}" = "1" ]; then
  say "EXTRACT_ONLY set: stopping before the GPU phase (backgrounds not generated)"
  exit 0
fi

# ---------- background generation (GPU phase) ----------
if [ "$SMOKE_DEFERRED" = "1" ]; then
  say "GPU smoke was deferred; it must pass before any background is generated"
  run_inference "$SMOKE_IN" data/bg_aifs_smoke_2018 smoke \
    || { say "ABORT: deferred inference smoke failed"; exit 1; }
  run_inference data/aifs_inputs/2021 data/bg_aifs_2021 "2021-head" \
    || { say "ABORT: 2021 head regeneration failed"; exit 1; }
  verify_smoke_rmse || exit 1
  for NEED in state_20210101T00.npz state_20211231T18.npz; do
    [ -f "data/bg_aifs_2021/$NEED" ] || { say "ABORT: bg_aifs_2021 missing $NEED"; exit 1; }
  done
  say "deferred GPU smoke + 2021 head/tail verified"
  SMOKE_DEFERRED=0
fi

for YEAR in 2018 2019 2020; do
  say "STAGE C: generating $YEAR backgrounds (device picked at runtime)"
  run_inference data/aifs_inputs/$YEAR data/bg_aifs_train_$YEAR "bg-$YEAR" "bg_gen_$YEAR.log" \
    || { say "ABORT: $YEAR background generation failed"; exit 1; }
  NB=$(ls data/bg_aifs_train_$YEAR/*.npz 2>/dev/null | wc -l)
  MIN=1460; [ "$YEAR" = "2020" ] && MIN=1464   # 2020 is a leap year (366 x 4)
  say "year $YEAR backgrounds: $NB (min $MIN)"
  if [ "$NB" -lt "$MIN" ]; then say "ABORT: $YEAR backgrounds incomplete ($NB < $MIN)"; exit 1; fi
  [ -f "data/bg_aifs_train_$YEAR/state_${YEAR}0101T00.npz" ] || { say "ABORT: $YEAR head background missing"; exit 1; }
done
say "final gate: assembling the single_da_12 background set (union must be complete)"
$ROOT/.venv/bin/python $ROOT/scripts/assemble_aifs_backgrounds.py >> "$LOG" 2>&1 \
  || { say "ABORT: assembler failed"; exit 1; }
grep -q '"complete": true' $ROOT/outputs/single_da_12_aifs_background/backgrounds/manifest.json \
  || { say "ABORT: assembled background set incomplete"; exit 1; }
say "TRAINING-YEAR PREP COMPLETE: inputs + backgrounds for 2018-2020 (+tails), assembly complete"
