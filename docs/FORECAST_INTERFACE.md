# Forecast to OffGrid-DA interface v1

This contract describes the inspected six-hour ArchesWeather-M profile on
research32-hkuwifi, not a generic forecast model. The authoritative machine
record is configs/forecast/archesweather_6h_contract.json.
The current training process and its configuration are unchanged.

## Forecast input and output

The model takes two normalized full states: prev_state at t-6h and state at t,
plus timestamp (Unix UTC seconds at t). Direct forward returns a decoded,
normalized full NEXT STATE at t+6h. Do not add state a second time or treat
the output as a delta. Loss-delta standard deviations are training loss
weights, not physical-output de-normalization scales.

| Group | Tensor shape | Variable order | Physical units |
| --- | --- | --- | --- |
| surface | [B,4,1,121,240] | U10, V10, T2M, MSLP | m/s, m/s, K, Pa |
| level | [B,6,13,121,240] | geopotential, U, V, T, specific humidity, vertical velocity | m2/s2, m/s, m/s, K, kg/kg, Pa/s |

Pressure order in hPa: 50,100,150,200,250,300,400,500,600,700,850,925,1000.
The upstream short label SP actually maps to mean_sea_level_pressure, not
surface pressure. Geopotential must not silently be interpreted as height;
vertical_velocity here is pressure vertical velocity (omega).

## Grid boundary

Both systems use 121 latitude by 240 longitude points at 1.5 degrees.
Latitude is north-to-south: 90 to -90. Canonical longitude is 0 to 358.5.
GeoArches internally rolls longitude by +120 cells. Export must undo it with
roll(-120,-1); corrected analyses use roll(+120,-1) on reentry.
No interpolation or observation gridding is needed.

Current DA consumes physical T2M at native surface index 2:
[B,1,121,240], float32 Kelvin. Full forecast states still carry all 82
scalar channels (4 + 6*13); the one-channel DA field cannot replace that state.

## Normalization boundary

Forecast input/output uses the 1998-2017 six-hour train statistics:
experiments/archesweather-reproduction/stats/era5_norm_stats_1998_2017_lead6h.pt.
DA currently uses outputs/normalization/norm_stats_train_2018_2020.json.
Both fingerprints are recorded in the JSON contract.

Always convert through physical units:

    T_kelvin = forecast_normalized_T * forecast_T_std + forecast_T_mean
    DA_input = (T_kelvin - DA_background_mean) / DA_background_std
    T_analysis_kelvin = DA_analysis * DA_background_std + DA_background_mean
    next_forecast_T = (T_analysis_kelvin - forecast_T_mean) / forecast_T_std

Use float32 for physical conversion, including means/stds, even if network
inference uses BF16. Observation values and DA residual targets use the same
DA background coordinate system. Do not invert a DA residual result using
the separately fitted reference mean/std. Existing DA stats remain fixed
for interface validation; they were fitted to Pangu backgrounds, so the
forecast source change requires DA retraining and a normalization audit.
Never refit statistics on validation/test data.

## Time and availability

One step is SIX HOURS. From an original init time, steps 1 and 2 have leads
6 and 12 hours; valid_time = init_time + lead_hours. Metadata must explicitly
identify UTC init and valid times, including the older history timestamp.
At each valid_time, METAR timeObs is restricted to (valid_time-60min, valid_time],
and timeReceived <= valid_time+30min, matching the current DA configuration.
METAR offsets are relative to the forecast valid time being analyzed.

Inspected Era5Dataset.convert_trajectory_to_xarray still constructs timedelta
coordinates in DAYS. Do not rely on that helper's labels for this six-hour
profile. The new boundary adapter calculates valid times explicitly in HOURS.

## Interface helpers and cycling

src/offgrid_da/data/forecast_contract.py supplies:

- archesweather_t2m_background(prediction, stats, init_times, lead_hours=6):
  validates full-state shape/finiteness, exports physical T2M on the DA grid,
  and computes UTC valid times. It accepts a decoded single state, not a
  trajectory tensor or backbone output.
- replace_archesweather_t2m(prediction, analysis_kelvin, stats):
  clones the full state and replaces T2M only, preserving all other channels.

After DA at t+6h, move state_at_t into prev_state, use the corrected full
state_at_t+6h as state, and advance timestamp by 21600 seconds. Keep a state
cache or explicit history to do this. These helpers do not load a checkpoint,
call forecast.forward, or implement the cycling loop.

## Acceptance and remaining decisions

Eight contract tests passed: units/grid/time conversion, unchanged original
states and other channels, invalid lead/time/nonfinite rejection, BF16-safe
statistics handling. A real ERA5 T2M field was normalized, rolled, exported
and round-tripped using the actual forecast statistics. This checks boundary
conversion; it is not trained-checkpoint inference or a forecast skill test.

Reproduce:

    .venv/bin/python -m pytest -q tests/test_forecast_contract.py
    experiments/archesweather-reproduction/geoarches/.venv/bin/python scripts/probe_forecast_contract.py

Probe evidence is saved at outputs/forecast_contract/probe.json.
The second command uses the forecast environment for its NetCDF dependency.

Once training is ready, select and freeze the checkpoint (path and SHA256
are intentionally unset), pair it with the recorded config and statistics,
verify actual forward output against this contract, and then run single-cycle
DA evaluation before cycling. The local GeoArches checkout is modified;
the upstream commit alone is insufficient to reconstruct the trained model.
