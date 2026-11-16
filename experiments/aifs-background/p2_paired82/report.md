# P2 paired 82-channel comparison (2021 full paired set)

paired valid times: **1458**

| group | AIFS RMSE (mean) | Arches RMSE (mean) | AIFS wins |
|---|---:|---:|---:|
| surface | 6.815 | 8.146 | 4/4 |
| geopotential | 23.97 | 28.14 | 13/13 |
| u | 0.7782 | 0.9246 | 13/13 |
| v | 0.7793 | 0.952 | 13/13 |
| temperature | 0.2756 | 0.3378 | 13/13 |
| specific | 0.0001188 | 0.0001289 | 13/13 |
| vertical | 0.05099 | 0.05165 | 11/13 |

## Key channels

| channel | AIFS | Arches | ratio |
|---|---:|---:|---:|
| 2m_temperature | 0.42977 | 0.44139 | 0.974 |
| 10m_u | 0.3842 | 0.45421 | 0.846 |
| 10m_v | 0.39219 | 0.47061 | 0.833 |
| mslp | 26.053 | 31.219 | 0.835 |
| t@850 | 0.24033 | 0.27617 | 0.870 |
| t@500 | 0.21636 | 0.25551 | 0.847 |
| z@500 | 18.804 | 21.867 | 0.860 |
| q@850 | 0.0002628 | 0.00028919 | 0.909 |
| w@500 | 0.075315 | 0.077763 | 0.969 |

time-semantics probe (T2M SSE, subset): {'-6': 2395368.622012795, '0': 45969.353063633076, '6': 2248887.7609966407}

alignment correct iff offset 0h minimizes SSE.

## Conventions and contamination check

- Area weighting: sum(w)*lon denominator (true cos-lat weighting) on BOTH
  sides - identical convention by construction (paired, same loop).
- Regrid convention check: a wrong downscale (linear point sampling instead
  of the 7x7 box average) costs ~+0.2 K T2M RMSE against era5_240
  (decompose_regrid.py measured); the pooled T2M here sits in the
  box-average class, i.e. the stored backgrounds carry the boxavg
  convention end-to-end.
- No old-convention results are reused: all numbers in this file are
  recomputed from the raw npz + era5_240 shards in this run; the
  retracted 0.3513 (N*M denominator) remains retracted in
  P0_NOTES/README/EXPERIMENT_REGISTRY.

## Monthly key channels (T2M)

| month | AIFS t2m | Arches t2m | ratio |
|---|---:|---:|---:|
| 01 | 0.4552 | 0.4567 | 0.997 |
| 02 | 0.4585 | 0.4636 | 0.989 |
| 03 | 0.4417 | 0.4530 | 0.975 |
| 04 | 0.4319 | 0.4462 | 0.968 |
| 05 | 0.4165 | 0.4391 | 0.949 |
| 06 | 0.4219 | 0.4378 | 0.964 |
| 07 | 0.4168 | 0.4332 | 0.962 |
| 08 | 0.4047 | 0.4236 | 0.955 |
| 09 | 0.3985 | 0.4155 | 0.959 |
| 10 | 0.4094 | 0.4265 | 0.960 |
| 11 | 0.4450 | 0.4474 | 0.995 |
| 12 | 0.4541 | 0.4534 | 1.002 |
