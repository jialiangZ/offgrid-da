# Full-state multivariate Cycling workflow

Entry: `.venv/bin/python -u scripts/run_full_update_cycling_pipeline.py`

The dispatcher is running and waits for `outputs/single_da_07_full82_small_stopped/status.json` to
report complete. It verifies all four selected model results/checkpoints and
training/validation signatures before freezing code and normalization hashes.
No random models are used as experimental forecast-analysis models.

GPU2/3 are reserved by convention for seed42/43, respectively. The dispatcher
waits for memory <1 GiB and utilization <10% before each stage. It does not
interrupt existing GPU users. This polling is not a cluster resource reservation.

Order: unit contracts -> upstream completion -> freeze -> real fixed training
batch fit (16 steps on in-memory clones; no frozen weights changed) -> two-step
smoke -> four-season five-day pilot -> coverage/finite/free-run reproduction
checks -> 24 five-day windows -> report and curve generation. Failures stop the
pipeline and record a failed status. No model or scale is reselected using
Cycling performance. The pilot gate checks engineering validity, not a claim of
positive scientific performance or physical balance.

All models consume 82 channels. DA additionally consumes temperature, dewpoint,
U/V, and valid sea-level pressure tokens. DA and NC update every channel. The
masked-DA trajectory must reproduce free forecast exactly. Channels are
converted through physical units to/from the separate forecast normalization,
including longitude and pressure-level layout conversion.

Reported primary score is dimensionless ten-group balanced normalized RMSE.
Individual channels are reported in native units. Heldout station scores use
physical T/U/V/MSLP only; dewpoint remains input-only until an appropriate
observation operator exists. The previous T2M-only station scoring is not reused
for all multivariate tokens. QC/error assumptions remain those of Single-cycle.

Output: `/home/jlzhang/Documents/offgrid-da/outputs/cycling_full82_01_small_superseded/`.
Plan: `workflow_plan.json`; state: `status.json`; log: `pipeline.log`;
engineering: `real_batch_gate.json`; frozen model/code hashes: `frozen.json`;
results: `pilot_summary.json`, `formal_summary.json`; reports:
`pilot_report.md`, `formal_report.md` with per-seed curves. Regions, per-lead,
per-window and per-channel results are retained in summaries and JSONL records.

Only 2021 development windows are used; 2023 confirmation/data preparation is
outside this workflow. No original T2M experimental source or result is modified.
