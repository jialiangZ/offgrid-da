# Training-error-balanced full-state DA experiment

The 82-channel state, forecast checkpoint, observations, architecture (dim128,
depth4, heads8), two training seeds 42/43, optimizer and ten-epoch budget are
retained. Old loss results and weights are not overwritten or resumed as new
training. This experiment retrains DA and NC from the same convolutional
initialization; it changes only the loss/selection objective and reporting.

1. Accumulate area-weighted squared background-reference error across all 4384
   training times in 2018–2020. No 2021/2022 data fit the statistics.
2. Per channel, use error RMS with a lower bound equal to max(10% of its
   variable group's RMS, 1e-6 in existing normalized units). This avoids excessive
   weight on nearly zero-error channels. Four surface variables and six upper-air
   variable groups define ten groups; 13 pressure levels are averaged within
   each upper-air group.
3. Adjust each group's final coefficient to contribute exactly 0.1 to the
   TRAINING background MSE after the floor. The training baseline score is 1.
   Validation/background scores need not be 1, nor need their contributions
   remain 10%. This is an empirical objective choice, not an optimal likelihood.
4. Preserve the existing state and observation normalization; modify loss
   weights only. Selection uses dimensionless balanced RMSE with explicit
   score keys. Physical-unit channel RMSE, baseline and group contributions
   are saved in every validation record. Scores cannot be directly compared
   to the old objective's scalar values.
5. GPU acceptance uses a fixed real training sample: 64 optimizer updates,
   finite gradients, improvement over initialization, exact zero initialization,
   no-observation identity (DA), observation independence (NC), checkpoint reload.
6. Diagnose both OLD validation-selected DA checkpoints on all 1460 times:
   true observations; full masking; three within-time, within-variable
   permutations; attention bypass with trained convolution branch active.
   Saved per-channel SSE permits paired and group-level follow-up analysis.
   Frozen ablations are developmental validation evidence, not blind tests.
7. Fresh matched DA/NC 256-step trials, followed by formal 43840-step runs
   (early stopping remains enabled). Trials and final checkpoint reloads gate
   progression. Verify identical initialized convolution-branch fingerprints.
8. Query all GPUs every 30 seconds; require memory below 1536 MiB and
   utilization below 10% for three consecutive checks. Never stop existing jobs.
   Two seeds each receive a free GPU, with matching light NC sharing that GPU.
   This polling is not an external cluster reservation system.

Main entry: scripts/run_error_balanced_pipeline.py
Outputs: /home/jlzhang/Documents/offgrid-da/outputs/single_da_09_full82_loss_balanced/
status.json contains phase, child PIDs and failures. Frozen source fingerprints
and snapshots are saved before work. Any failing stage stops later stages;
already-running sibling jobs are allowed to finish. Re-runs reuse completed
results and atomically saved last.pt states. Statistics rerun if interrupted.

2021 remains the validation/selection year, 2022 is not opened by this pipeline.
Innovation is still disabled; this experiment first tests the objective issue.
