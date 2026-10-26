# Known Issues

1. **Dataset cold init ≈ 3 min.** `SingleCycleDADataset.__init__` scans all hourly
   partitions to enumerate stations (~124 files Jan window; disk-cached afterwards).
   Mitigation planned: persist per-month consolidated observation arrays + station lists.

2. **gcsfs 2026.10.1 path bug for non-zarr objects.** `fs.open("gs://…")` mangles object
   paths; zarr reads unaffected. Workaround: METAR parquet fetched via plain HTTPS
   (`utils/io.py`). Revisit if gcsfs is fixed.

3. **METAR DD flags B/G/Q/X are not officially documented** in the WB2 export. Current
   treatment (accept, low quality weight) is based on empirical characterization over
   Jan-2020 (~5 partitions × 5 variables). Risk: semantics may differ for other
   variables/months. Action: re-check when extending to U10/V10 and longer periods.

4. **`timeReceived` slightly before `timeObs`** for ~3% of reports (negative lag,
   typically ≤ 2 min, stations reporting early). Current cutoff only enforces
   `timeReceived ≤ t+30min`, so these pass. No evidence of harm; documented in
   DATA_AUDIT §4.

5. **Representativeness floor:** obs−ERA5 innovation RMSE ≈ 2.48 K vs grid forecast
   error 0.46 K. Any station-space "assimilation gain" claims must exceed this floor
   carefully; plan §5.6 requires stratified reporting.

6. **Pangu zarr lacks units attributes**; units follow WB2 conventions (K, m s-1, Pa).
   Verified for T2M only in this round; re-verify for wind variables in Phase 5.

7. **ERA5 zarr time coverage ends 2023-01-10** — fine for the 2018–2022 protocol, but
   blocks extending test years beyond 2022 without switching datasets (would be a
   protocol change requiring plan update).

8. **Station split over Jan-2020 window only.** The full 2018–2020 station pool (stations
   appearing/disappearing over years) must be rebuilt for full training; current split is
   minimal-window-scoped by design.

9. **`_read_hour` LRU cache is per-process**; multi-worker DataLoader (num_workers>0)
   re-downloads from local disk cache but re-parses parquet per worker. Acceptable for
   M2 single-GPU training; profile before scaling.
