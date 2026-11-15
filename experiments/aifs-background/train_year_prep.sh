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
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

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
CUDA_VISIBLE_DEVICES=0 ANEMOI_INFERENCE_NUM_CHUNKS=32 $VENV run_aifs_background.py --inputs data/aifs_inputs/2018 --out data/bg_aifs_smoke_2018 >> "$LOG" 2>&1 \
  || { say "ABORT: inference smoke failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 ANEMOI_INFERENCE_NUM_CHUNKS=32 $VENV run_aifs_background.py --inputs data/aifs_inputs/2021 --out data/bg_aifs_2021 >> "$LOG" 2>&1 \
  || { say "ABORT: 2021 head regeneration failed"; exit 1; }
for NEED in state_20210101T00.npz state_20211231T18.npz; do
  [ -f "data/bg_aifs_2021/$NEED" ] || { say "ABORT: bg_aifs_2021 missing $NEED"; exit 1; }
done
say "2021 val backgrounds complete (head+tail verified)"
N=$(ls data/bg_aifs_smoke_2018/*.npz 2>/dev/null | wc -l)
say "smoke produced $N backgrounds; verifying T2M against era5"
$PY - <<'EOF' >> "$LOG" 2>&1 || { echo "SMOKE RMSE FAILED"; exit 1; }
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
say "STAGE A PASSED (small-shard acceptance)"

# ---------- continuous extraction ----------
for YEAR in 2018 2019 2020; do
  say "STAGE B: extracting $YEAR"
  $PY prepare_aifs_inputs.py --year $YEAR >> "prepare_inputs_$YEAR.log" 2>&1
  RC=$?
  N=$(ls data/aifs_inputs/$YEAR/*.npy.zst 2>/dev/null | wc -l)
  say "year $YEAR extraction exit=$RC files=$N"
  if [ $RC -ne 0 ] || [ "$N" -lt 1460 ]; then say "ABORT: $YEAR incomplete ($N)"; exit 1; fi
  say "STAGE C: generating $YEAR backgrounds (GPU0)"
  CUDA_VISIBLE_DEVICES=0 ANEMOI_INFERENCE_NUM_CHUNKS=32 $VENV run_aifs_background.py --inputs data/aifs_inputs/$YEAR --out data/bg_aifs_train_$YEAR >> "bg_gen_$YEAR.log" 2>&1
  NB=$(ls data/bg_aifs_train_$YEAR/*.npz 2>/dev/null | wc -l)
  MIN=1464; [ "$YEAR" = "2020" ] || MIN=1460
  say "year $YEAR backgrounds: $NB (min $MIN)"
  if [ "$NB" -lt "$MIN" ]; then say "ABORT: $YEAR backgrounds incomplete ($NB < $MIN)"; exit 1; fi
done
say "final gate: assembling the single_da_12 background set (union must be complete)"
$ROOT/.venv/bin/python $ROOT/scripts/assemble_aifs_backgrounds.py >> "$LOG" 2>&1 \
  || { say "ABORT: assembler failed"; exit 1; }
grep -q '"complete": true' $ROOT/outputs/single_da_12_aifs_background/backgrounds/manifest.json \
  || { say "ABORT: assembled background set incomplete"; exit 1; }
say "TRAINING-YEAR PREP COMPLETE: inputs + backgrounds for 2018-2020 (+tails), assembly complete"
