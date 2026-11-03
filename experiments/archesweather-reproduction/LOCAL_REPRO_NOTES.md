# Local reproduction adjustments

The upstream GeoArches checkout remains pinned at v0.2.2 commit fb2285fc960fca9f11b2b8d0126deba810f87895. The original 20-year patch is saved in patches/archesweather-20year.patch; the combined 20-year and 6-hour patch is saved in patches/archesweather-20year-6h.patch.

- Training years: 1998-2017 inclusive (20 years).
- Recent-data fine-tuning: 2007-2017 inclusive.
- Validation/test domains: paper-style 2019/2020.
- Download shards: 1998-2020 inclusive, 00/06/12/18 UTC.
- Input statistics are calculated from 1998-2017; the 6-hour loss delta standard deviations are computed from adjacent 6-hour pairs with endpoints within the training split, using per-variable/per-level unweighted time-latitude-longitude population moments.

## Runtime adaptation

- Server driver: NVIDIA 555.42.06, CUDA 12.5. The lockfile PyTorch 2.13 CUDA 13.0 wheel cannot initialize on this driver.
- The isolated project environment uses PyTorch 2.5.1+cu124 and torchvision 0.20.1+cu124. scripts/prepare_train_env.sh documents how to recreate this override. SwanLab 0.10.3 records experiment logs offline under geoarches/swanlogs/.
- The official ArchesWeather model is unchanged. The local run enables its built-in gradient checkpointing and sets AdamW foreach=False to fit on 24 GB RTX 4090 cards.

## Preflight results

- All 92 ERA5 files validated; total size 421,725,637,516 bytes.
- Six-hour train loader: 29,218 samples. Validation loader: 1,460 samples. Test loader: 1,462 starts from 2020-01-01 00Z through 2020-12-31 06Z.
- First train sample verified as prev_state=t-6h, state=t, next_state=t+6h; two-step validation fields have shape (2, 6, 13, 121, 240) and finite values.
- A four-GPU short diagnostic completed one training batch and a two-step validation batch with finite losses. Full training has not started.
- Without a 2021 boundary shard, the final two 6-hour starts of the 2020 two-step test cannot be evaluated.

## Six-hour lead adaptation

- Each sample uses prev_state=t-6h and state=t to predict t+6h; ERA5 anchors are sampled every six hours.
- Training loss uses one autoregressive step. Validation rolls out two steps (6h and 12h) and reports metrics for both.
- scripts/compute_train_norms_6h.py writes stats/era5_norm_stats_1998_2017_lead6h.pt; existing 24-hour statistics are preserved.
- The dataloader per-step lead_time_hours remains 6 during multistep rollout, so timestamps and conditioning advance by 6 hours per step.
- This is an ArchesWeather-M temporal adaptation, not an exact reproduction of the paper's 24-hour forecast protocol.

## Mixed-precision smoke finding

- A 20-step four-GPU FP16 mixed-precision probe reached 20 Lightning global steps but AdamW parameter states advanced only 16 times, indicating skipped optimizer updates. A matched BF16 mixed-precision probe advanced all 20 optimizer steps. The research32 local profile and 20-year launcher therefore use bf16-mixed.
