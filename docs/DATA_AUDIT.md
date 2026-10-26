# OffGrid-DA Data Audit (Phase 0)

Generated: 2026-10-26 16:08:10.934586 | all values from live reads of gs://weatherbench2

## 1. Pangu forecast (background)

- URL: `gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr`
- Dims: `{'time': 3652, 'prediction_timedelta': 40, 'longitude': 240, 'latitude': 121, 'level': 13}`
- Lead coordinate name: **`prediction_timedelta`** (NOT `lead_time` as assumed in the plan)
- Lead times (h): [6, 12, 18, 24, 30, 36, 42, 48] ... [222, 228, 234, 240] (n=40); **6h lead exists: True**
- Init times: 2018-01-01T00:00 .. 3652 samples=3652 at hours [0, 12] UTC
- Variables (n=11): 10m_u_component_of_wind, 10m_v_component_of_wind, 10m_wind_speed, 2m_temperature, geopotential, mean_sea_level_pressure, specific_humidity, temperature, u_component_of_wind, v_component_of_wind, wind_speed
- Grid: 121x240, lat 90.0->-90.0 (descending after canonicalization; source stores ascending), lon 0.0..358.5, d=1.5000000000000426 deg, poles included: True
- Native dim order: (time, prediction_timedelta, longitude, latitude)
- Units attrs present in zarr: False (units follow MADIS/ECMWF conventions, see §4)
- Sample field (T2M, init 2020-01-01T00Z, +6h): shape [121, 240], NaN 0, range 227.01..317.44 K, mean 277.33 K

## 2. ERA5 analysis (reference)

- URL: `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr`
- Dims: `{'time': 93544, 'longitude': 240, 'latitude': 121, 'level': 13}`; time 1959-01-01T00:00 .. 2023-01-10T18:00 (n=93544) at hours [0, 6, 12, 18] UTC
- Variables: n=62; intersection with Pangu (n=11): 10m_u_component_of_wind, 10m_v_component_of_wind, 10m_wind_speed, 2m_temperature, geopotential, mean_sea_level_pressure, specific_humidity, temperature, u_component_of_wind, v_component_of_wind, wind_speed
- Key units: {'2m_temperature': 'K', '10m_u_component_of_wind': 'm s**-1', 'mean_sea_level_pressure': 'Pa'}
- Grid identical to Pangu: True (121x240)
- Sample field (T2M, 2020-01-01T06Z): NaN 0, range 225.18..317.94 K

## 3. Time matching (Pangu <-> ERA5)

- Rule: valid_time = init_time(00/12Z) + 6h => analysis hours 06/18 UTC (all present in ERA5)
- Minimal window 2020-01-01 .. 2020-01-31: **60 aligned samples**, valid hours [6, 18]
- Split train (2018-01-01..2020-12-31): 2190 samples
- Split val (2021-01-01..2021-12-31): 728 samples
- Split test (2022-01-01..2022-12-31): 728 samples

## 4. MADIS METAR (observations)

- Layout: `metar-timeObs-by-hour_2001-07-01T00_2024-01-01T00` / partitions `year=YYYY/month=M/day=D/hour=H/YYYY-MM-DDTHH.parquet`
- Columns: n=43, schema matches code constant: True (see metar_schema.csv)
- Example partition rows: 8247 (2020-01-01T06)
- Longitude convention: [-180, 180] (verified from data: negative longitudes present)
- Units (verified against MADIS docs + values): {
  "temperature": "K (MADIS: T air temperature K; values ~282K in sample)",
  "dewpoint": "K",
  "windDir": "deg, meteorological FROM direction (MADIS DD deg)",
  "windSpeed": "m s-1 (MADIS FF)",
  "seaLevelPress": "Pa (MADIS SLP Pa; values ~101300 in sample)",
  "visibility": "m",
  "elevation": "m"
}
- QC: one <var>DD descriptor per variable (S/T/V/C/Z); numeric MADIS QC codes NOT exported
- Time checks per partition (MADIS promises [H-15min, H+44min]):
  - 2020-01-01T05: rows=8179, timeObs 2020-01-01T05:00..2020-01-01T05:59, covers[H-15,H+44]=False, recv lag p50=7.0 min, p99=29.0 min, max=78.0 min, recv<obs frac=0.030
  - 2020-01-01T06: rows=8247, timeObs 2020-01-01T06:00..2020-01-01T06:59, covers[H-15,H+44]=False, recv lag p50=5.0 min, p99=25.0 min, max=75.0 min, recv<obs frac=0.029
  - 2020-01-15T18: rows=8415, timeObs 2020-01-15T18:00..2020-01-15T18:59, covers[H-15,H+44]=False, recv lag p50=6.0 min, p99=24.0 min, max=94.0 min, recv<obs frac=0.024
  - 2020-01-31T23: rows=8106, timeObs 2020-01-31T23:00..2020-01-31T23:59, covers[H-15,H+44]=False, recv lag p50=5.0 min, p99=25.0 min, max=84.0 min, recv<obs frac=0.032
- Sampled 121 partitions (1011113 rows): 5460 unique stations
- Valid counts per variable (sampled): {'temperature': 1004369, 'dewpoint': 996698, 'windDir': 962135, 'windSpeed': 998699, 'seaLevelPress': 249104, 'altimeter': 945258, 'visibility': 934831}
- DD descriptor distribution (variable|flag -> count, sampled):
  - altimeter|B: 632
  - altimeter|G: 393
  - altimeter|Q: 27811
  - altimeter|S: 336791
  - altimeter|V: 579631
  - altimeter|Z: 65855
  - dewpoint|B: 773
  - dewpoint|G: 390
  - dewpoint|Q: 9010
  - dewpoint|S: 336656
  - dewpoint|V: 649769
  - dewpoint|X: 100
  - dewpoint|Z: 14415
  - seaLevelPress|B: 232
  - seaLevelPress|G: 236
  - seaLevelPress|Q: 13045
  - seaLevelPress|S: 4424
  - seaLevelPress|V: 231167
  - seaLevelPress|Z: 762009
  - temperature|B: 773
  - temperature|Q: 12202
  - temperature|S: 337696
  - temperature|V: 653604
  - temperature|X: 94
  - temperature|Z: 6744
  - visibility|B: 770
  - visibility|C: 934061
  - visibility|Z: 76282
  - windDir|B: 345
  - windDir|G: 385
  - windDir|Q: 17380
  - windDir|S: 311711
  - windDir|V: 632314
  - windDir|Z: 48978
  - windSpeed|B: 360
  - windSpeed|G: 388
  - windSpeed|Q: 17458
  - windSpeed|S: 337351
  - windSpeed|V: 643142
  - windSpeed|Z: 12414
- Regional report density (rows, sampled):
  - global (all): 1011113
  - northern extra-tropics [30,60]: 795013
  - tropics [-30,30]: 148851
  - southern extra-tropics [-60,-30]: 13724
  - US [24,50]x[-125,-66]: 570106
  - Europe [35,60]x[-10,30]: 137262
  - East Asia [20,45]x[100,135]: 13955

## 5. Conclusions for Phase 1

1. Pangu provides a real 6h lead forecast for every 00/12Z init; ERA5 contains all resulting 06/18Z valid times -> single-cycle supervision is fully covered for 2018-2022.
2. Pangu and ERA5 share the same 11 variables and identical 240x121 grid, so no spatial regridding is needed for grid supervision.
3. METAR `timeObs`/`timeNominal`/`timeReceived` allow an explicit observation window and a received-time cutoff. Empirically each timeObs-by-hour partition holds timeObs in [H:00, H:59] (the WB2 export rebuckets raw MADIS hours), so a 60-min observation window ending at valid time t reads exactly the two partitions floor(t-1h)..floor(t); MetarReader.read_window covers this conservatively by starting 45 min earlier.
4. The only QC carried in the parquet is the per-variable DD descriptor. Empirical characterization over Jan-2020: only 'Z' rows lack values; flags S/V/C/T/B/G/Q all carry plausible values (V dominates, mostly AUTO stations; C marks unit-converted visibility); 'X' rows are implausible constants. Default QC therefore accepts S/V/C/T/B/G/Q, rejects Z and X, plus finite/range/location checks, per-variable counts logged. Numeric MADIS quality codes are absent from the export.
5. METAR temperatures are in Kelvin — consistent with ERA5 T2M (K); no unit conversion needed for T2M, longitude must be shifted from [-180,180] to [0,360).