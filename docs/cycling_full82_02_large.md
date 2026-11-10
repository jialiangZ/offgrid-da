# Large full-state multivariate Cycling workflow

Model: dim128, depth4, heads8. Cycling loads the exact selected large Single-cycle weights; it does not train another DA model. Previous small-model dispatcher was superseded.


Entry: `.venv/bin/python -u scripts/run_full_update_large_cycling_pipeline.py`

The dispatcher is running and waits for `outputs/single_da_08_full82_large/status.json` to
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

Output: `/home/jlzhang/Documents/offgrid-da/outputs/cycling_full82_02_large/`.
Plan: `workflow_plan.json`; state: `status.json`; log: `pipeline.log`;
engineering: `real_batch_gate.json`; frozen model/code hashes: `frozen.json`;
results: `pilot_summary.json`, `formal_summary.json`; reports:
`pilot_report.md`, `formal_report.md` with per-seed curves. Regions, per-lead,
per-window and per-channel results are retained in summaries and JSONL records.

Only 2021 development windows are used; 2023 confirmation/data preparation is
outside this workflow. No original T2M experimental source or result is modified.

## CPU input prefetch optimization

Cycling runs one independent seed on GPU2 and GPU3, respectively. Two spawned
CPU workers per seed prefetch references and multivariate observation tokens
with depth 2 and pinned memory. Recursive forecast/analysis states remain
strictly sequential. No changes to batch size, model, weights, windows or
normalization. Each seed reuses its forecast/DA/NC instances; methods execute
sequentially within the seed to avoid duplicating large attention activations.

The loader reads only observation tokens and native forecast-normalized ERA5
reference tensors. It skips unused independent one-step background and duplicate
full-state supervised target reads. Input order and time are checked at every
cycle. Reference tensors only enter scoring, never the forecast/analysis inputs.

Regional weighted sums are computed in a batched GPU reduction and copied to
CPU once per field instead of synchronizing once per region. The primary score
and individual physical units are unchanged.

`verify_cycling_prefetch.py` checks 24 real samples across two windows, exact
serial/two-worker tensor and order equivalence, equality to old observation
inputs, exact GPU reference reconstruction, and regional reduction equivalence.
The CPU-only benchmark has no simulated GPU compute, so its wait times are not
an end-to-end Cycling speedup estimate. The original-loop real two-step smoke
and free-forecast replay checks remain required after trained weights exist.
The dispatcher understands upstream `parallel_seed_status.json` failures.
