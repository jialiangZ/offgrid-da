# Variable Mapping (Pangu / ERA5 / METAR)

Generated: 2026-10-26 16:08:10.934961 — names verified against live schemas.

| internal name | variable_id | units | Pangu zarr var | ERA5 zarr var | METAR column | METAR DD col | obs error (default) |
|---|---|---|---|---|---|---|---|
| 2m_temperature | 0 | K | 2m_temperature | 2m_temperature (units K) | temperature (K) | temperatureDD | 1.0 K |
| 10m_u_component_of_wind | 1 | m s-1 | 10m_u_component_of_wind | 10m_u_component_of_wind | derived: u = -V·sin(θ) from windSpeed/windDir (later stage) | windSpeedDD+windDirDD | 1.5 m s-1 |
| 10m_v_component_of_wind | 2 | m s-1 | 10m_v_component_of_wind | 10m_v_component_of_wind | derived: v = -V·cos(θ) from windSpeed/windDir (later stage) | windSpeedDD+windDirDD | 1.5 m s-1 |

## Grid conventions

- Both zarrs: 240×121 equiangular with poles, 1.5°; native dim order (…, longitude, latitude); latitude stored ascending → canonicalized to descending 90→-90.
- Longitude [0, 360) in both zarrs; METAR station longitude [-180, 180] → normalized with mod 360.
- ERA5 T2M units K (zarr attr); METAR temperature K (MADIS T spec, verified values). No unit conversion needed for T2M.

## METAR DD data-descriptor flags (empirically characterized)

| flag | n (Jan-2020 sample) | value present | interpretation | default QC | quality code |
|---|---|---|---|---|---|
| S | 337,696 | 100% | single value | accept | 3 |
| V | 653,604 | 100% | dominant flag, ~95% AUTO stations | accept | 2 |
| C | 934,061 (visibility) | 100% | unit-converted (visibility miles→m) | accept | 3 |
| T | rare | 100% | time-averaged | accept | 2 |
| B | 773 | 100% | rare, values plausible | accept | 2 |
| G | 390 | 100% | rare, values plausible | accept | 2 |
| Q | 12,202 | 100% | rare, values plausible | accept | 1 |
| X | 94 | 100% | implausible constant (333.15 K) | reject | 0 |
| Z | 6,744 | 0% | missing | reject | 0 |

Note: the WB2 parquet export does **not** include the numeric MADIS quality codes (QC levels 0–3); only these DD descriptors are available.