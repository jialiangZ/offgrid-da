# Cross-attention integration validation

Run on research32-hkuwifi in ~/Documents/offgrid-da with the project .venv.
Innovation is reserved in ObservationEncoder and disabled in CrossAttentionAnalysisModel.
All observations retain their irregular positions; no observation gridding is performed.

## Reproduce

Run from the repository root:

    OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_cross_attention.py tests/test_observation_encoder.py tests/test_observation_padding.py
    OMP_NUM_THREADS=2 .venv/bin/python scripts/validate_cross_attention.py

The script writes results.json and two diagnostic state dictionaries under
outputs/single_da_00_model_check_crossattn/. It uses one GPU and six real cycles only.

## Acceptance checks

- 19 targeted model/encoder/padding tests passed.
- Full offline regression: 55 passed, 39 requires_gcs tests deselected.
- Real-data integration, normalization and padding: 16 passed.
- git diff --check passed.
- Both entry and layered injection have finite forward and backward results
  for ordinary, mixed-empty, all-empty, zero-length and nonfinite-record batches.
- Reordering observations leaves the result unchanged to numerical tolerance.
- Modifying padded records does not change the result.
- Perturbing valid observation values changes the analysis.
- Actual input allowlist excludes reference fields and held-out station observations.
  Context and held-out station sets are checked for disjointness.
- Raw background and reference tensors are unchanged after model execution.

The empty-set dummy attention key is unmasked to prevent all-masked softmax.
Its update is explicitly zeroed for empty rows. Invalid coordinates are sanitized
before differentiable operations so they cannot cause NaN parameter gradients.

## Actual real batch

Sample IDs: 20200101T00, 20200101T06.
Background and analysis shape: [2, 1, 121, 240].
Observation tokens: [2, 3740, 16]; valid counts: 3740 and 3733.
Forward plus backward: 1.220 seconds on CUDA device 0; peak allocated memory:
122.7 MiB (PyTorch allocated memory, not total process GPU memory).
Valid-observation perturbation maximum output change: 0.007841 normalized units.
Observation permutation maximum difference: 2.38e-7.

## Tiny training diagnostic

Seed 42, hidden dimension 16, two heads, two grid blocks, Adam lr 0.001,
40 updates, gradient norm clipping at 1.0.
Train cycles: 20200101T00, T06, T12, T18.
Diagnostic cycles: 20200102T00, T06.
These disjoint nearby cycles all lie in the official training period;
the diagnostic cycles are not the official validation split.

The model receives normalized background and context observation values.
The target is physical ERA5 transformed using BACKGROUND training mean/std,
so a residual analysis output shares the background coordinate system.
The separately normalized reference field is not used as a residual target.
Loss is cosine-latitude-weighted MSE; RMSE below is in physical Kelvin.

| Method | Initial train RMSE | Final train RMSE | Diagnostic RMSE |
| --- | ---: | ---: | ---: |
| Baseline 0: background | 0.4984 | 0.4984 | 0.4871 |
| Entry attention | 7.0780 | 0.6150 | 0.6046 |
| Layered attention | 6.6713 | 0.6169 | 0.6047 |

Training loss decreased for both variants. Neither beat Baseline 0 in this
bounded diagnostic. This confirms a working optimization and data path,
not improved assimilation performance or generalization. Both variants
allocate 13713 parameters; entry injection leaves later attention modules
unused, so this run is not a matched-active-parameter architecture study.
Full ablations, official validation evaluation and cycling are outside this run.

Machine-readable results: outputs/single_da_00_model_check_crossattn/results.json.
