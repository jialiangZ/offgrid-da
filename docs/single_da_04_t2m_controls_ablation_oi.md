# Single-cycle supplemental experiments v2

The running DA input ablation is untouched. All new results use 2021 validation;
2022 is not accessed. This is model development, not an independent blind test.

- Train background-only Neural Correction for both masked T2M and full82 inputs,
  seeds 42/43. Same frozen recipe, data, target and initialized convolutional
  branch as the respective DA arm. The full template retains inactive attention
  parameters, so equal total counts do not mean equal active computation.
- Each mode first passes a 256-step trial. Formal runs select best validation
  checkpoint and support exact trainer resume. Save frequency is 1000 steps,
  with best and final states; SwanLab is offline.
- Evaluate each completed DA checkpoint on all 1460 validation cycles: true
  observations, masked observations, and three deterministic within-cycle
  permutations of valid temperature values. Locations and metadata stay fixed.
- Score local OI once on identical T2M backgrounds and context observations.
  Great-circle exponential background covariance, fixed sigma_b=1 K,
  sigma_o=2.5 K, length=500 km, radius=1000 km, nearest 16 observations.
  The observation scale includes approximate representativeness error and is
  not estimated from validation. This is an untuned conventional baseline;
  poor performance alone cannot establish superiority over optimized OI.
- All methods use the same training T2M normalization and pooled cosine-latitude
  squared errors, followed by one square root. Per-cycle SSE is retained.
- GPUs 1/2 only. Five consecutive idle checks at 30-second intervals are required
  (memory <1024 MiB, utilization <10%). One job per GPU, no interruption of
  existing tasks. External users can still race the check; this is not a cluster
  resource reservation system.
- Frozen source/config snapshots, process IDs, logs and status are stored in
  /home/jlzhang/Documents/offgrid-da/outputs/single_da_04_t2m_controls_ablation_oi/.
  Any failed job halts further scheduling, leaving logs and checkpoints for repair.

Entry point: .venv/bin/python -u scripts/run_single_cycle_controls.py
