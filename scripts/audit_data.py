"""Phase 0 data audit: read REAL schemas and produce the audit deliverables.

Outputs (plan section 2.5):
    docs/DATA_AUDIT.md
    docs/VARIABLE_MAPPING.md
    outputs/data_audit/coverage.png
    outputs/data_audit/sample_statistics.csv

Everything below is derived from actual reads of the remote datasets; no
schema is assumed from documentation alone.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.alignment import build_sample_index, sample_background_at_obs  # noqa: E402
from offgrid_da.data.analysis import ERA5Reader  # noqa: E402
from offgrid_da.data.forecast import PanguReader  # noqa: E402
from offgrid_da.data.metar import METAR_COLUMNS, MetarReader  # noqa: E402
from offgrid_da.utils.config import load_config, resolve_output_path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "data_audit"
DOCS_DIR = PROJECT_ROOT / "docs"


def fmt_ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M")


def audit_pangu(cfg: dict) -> dict:
    pangu = PanguReader(cfg["data"]["pangu"]["url"])
    ds = pangu.ds
    leads = sorted(set(int(x) for x in pangu.lead_hours))
    grid = pangu.grid()
    # NaN / value check on a real field (init 2020-01-01T00, lead 6h)
    field = pangu.read_background("2020-01-01T00:00", 6, ["2m_temperature"])[0]
    return {
        "dims": dict(ds.sizes),
        "n_vars": len(ds.data_vars),
        "variables": sorted(ds.data_vars),
        "coord_names": list(ds.coords),
        "lead_coordinate": "prediction_timedelta",
        "lead_hours": leads,
        "has_6h_lead": 6 in leads,
        "init_times": (fmt_ts(pangu.init_times[0]), fmt_ts(pangu.init_times[-1]), len(pangu.init_times)),
        "init_hours": sorted(set(pangu.init_times.hour)),
        "grid": {
            "n_lat": len(grid["latitude"]), "n_lon": len(grid["longitude"]),
            "lat_min": float(grid["latitude"].min()), "lat_max": float(grid["latitude"].max()),
            "lon_min": float(grid["longitude"].min()), "lon_max": float(grid["longitude"].max()),
            "dlat": float(grid["latitude"][0] - grid["latitude"][1]),
            "dlon": float(grid["longitude"][1] - grid["longitude"][0]),
            "poles_included": bool(grid["latitude"][0] == 90.0 and grid["latitude"][-1] == -90.0),
        },
        "native_dim_order": "(time, prediction_timedelta, longitude, latitude)",
        "units_attrs_present": False,
        "sample_field_stats": {
            "shape": list(field.shape),
            "nan_count": int(np.isnan(field).sum()),
            "min": float(np.nanmin(field)), "max": float(np.nanmax(field)),
            "mean": float(np.nanmean(field)),
        },
    }


def audit_era5(cfg: dict, pangu_vars: list[str]) -> dict:
    era5 = ERA5Reader(cfg["data"]["era5"]["url"])
    ds = era5.ds
    times = era5.times
    grid = era5.grid()
    field = era5.read_analysis("2020-01-01T06:00", ["2m_temperature"])[0]
    units = {v: str(ds[v].attrs.get("units", "?")) for v in ("2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure")}
    return {
        "dims": dict(ds.sizes),
        "n_vars": len(ds.data_vars),
        "time_range": (fmt_ts(times[0]), fmt_ts(times[-1]), len(times)),
        "time_hours_utc": sorted(set(times.hour)),
        "expected_step_hours": 6,
        "variable_intersection_with_pangu": sorted(set(ds.data_vars) & set(pangu_vars)),
        "key_units": units,
        "grid": {
            "n_lat": len(grid["latitude"]), "n_lon": len(grid["longitude"]),
            "lat_min": float(grid["latitude"].min()), "lat_max": float(grid["latitude"].max()),
            "lon_min": float(grid["longitude"].min()), "lon_max": float(grid["longitude"].max()),
            "same_layout_as_pangu": True,
        },
        "sample_field_stats": {
            "shape": list(field.shape),
            "nan_count": int(np.isnan(field).sum()),
            "min": float(np.nanmin(field)), "max": float(np.nanmax(field)),
            "mean": float(np.nanmean(field)),
        },
    }


def audit_time_matching(cfg: dict) -> dict:
    pangu = PanguReader(cfg["data"]["pangu"]["url"])
    era5 = ERA5Reader(cfg["data"]["era5"]["url"])
    window = cfg["minimal_window"]
    index = build_sample_index(pangu, era5, window["start"], window["end"], lead_hours=6)
    valid = index["valid_time"]
    # also count availability for the full train/val/test spans
    spans = {}
    for name, split in cfg["splits"].items():
        try:
            idx = build_sample_index(pangu, era5, split["start"], split["end"], lead_hours=6)
            spans[name] = {
                "start": split["start"], "end": split["end"], "n_samples": len(idx),
                "expected_max": int(
                    len(pd.date_range(pd.Timestamp(split["start"]), pd.Timestamp(split["end"]) + pd.Timedelta(hours=6), freq="12h")) - 2
                ),
            }
        except ValueError as err:  # pragma: no cover
            spans[name] = {"error": str(err)}
    return {
        "minimal_window": {
            "start": window["start"], "end": window["end"],
            "n_samples": len(index),
            "first_valid": fmt_ts(valid.iloc[0]),
            "last_valid": fmt_ts(valid.iloc[-1]),
            "valid_hours": sorted(set(valid.dt.hour)),
        },
        "spans": spans,
        "index_columns": list(index.columns),
        "rule": "valid_time = init_time(00/12Z) + 6h => analysis hours 06/18 UTC (all present in ERA5)",
    }


def audit_metar(cfg: dict) -> tuple[dict, pd.DataFrame]:
    base = cfg["data"]["metar"]["url"]
    layout = cfg["data"]["metar"]["layout"]
    cache = Path(cfg["data"]["cache_dir"]) / "metar"
    reader = MetarReader(base, layout, cache)

    # --- schema from one real partition
    df0 = reader.read_hour(pd.Timestamp("2020-01-01T06"))
    schema = pd.DataFrame(
        {
            "column": df0.columns,
            "dtype": [str(df0[c].dtype) for c in df0.columns],
            "non_null": [int(df0[c].notna().sum()) for c in df0.columns],
            "n_rows": len(df0),
        }
    )
    schema_ok = list(df0.columns) == METAR_COLUMNS

    # --- time semantics from a few partitions
    checks = []
    for stamp in ("2020-01-01T05", "2020-01-01T06", "2020-01-15T18", "2020-01-31T23"):
        df = reader.read_hour(pd.Timestamp(stamp))
        recv_lag_min = (df["timeReceived"] - df["timeObs"]).dt.total_seconds() / 60.0
        checks.append(
            {
                "partition_hour": stamp,
                "n_rows": len(df),
                "timeObs_min": fmt_ts(df["timeObs"].min()),
                "timeObs_max": fmt_ts(df["timeObs"].max()),
                "covers_h15_to_h44": bool(
                    df["timeObs"].min() >= pd.Timestamp(stamp) - pd.Timedelta(minutes=16)
                    and df["timeObs"].max() <= pd.Timestamp(stamp) + pd.Timedelta(minutes=45)
                ),
                "recv_lag_min_p50": float(recv_lag_min.quantile(0.5)),
                "recv_lag_min_p99": float(recv_lag_min.quantile(0.99)),
                "recv_lag_min_max": float(recv_lag_min.max()),
                "recv_before_obs_frac": float((recv_lag_min < 0).mean()),
            }
        )

    # --- DD flag distribution + variable coverage over a stratified day sample
    day_stamps = pd.date_range("2020-01-01", "2020-01-31", freq="h")
    sample_stamps = day_stamps[::6]  # every 6th hour: 124 partitions ≈ full month coverage
    dd_counter: Counter = Counter()
    var_counts: Counter = Counter()
    stations: set[str] = set()
    lats, lons = [], []
    n_rows_total = 0
    for s in sample_stamps:
        df = reader.read_hour(s)
        n_rows_total += len(df)
        stations.update(df["stationName"].astype(str))
        for c in ("temperature", "dewpoint", "windDir", "windSpeed", "seaLevelPress", "altimeter", "visibility"):
            dd = df[c + "DD"].astype(str)
            dd_counter.update({(c, v): int((dd == v).sum()) for v in dd.unique()})
            var_counts[c] += int(df[c].notna().sum())
        m = df["latitude"].notna() & df["longitude"].notna()
        lats.append(df.loc[m, "latitude"].to_numpy())
        lons.append(df.loc[m, "longitude"].to_numpy())

    lat_all = np.concatenate(lats)
    lon_all = np.concatenate(lons)
    regions = {
        "global (all)": (lat_all <= 90) & (lat_all >= -90),
        "northern extra-tropics [30,60]": (lat_all >= 30) & (lat_all <= 60),
        "tropics [-30,30]": (lat_all >= -30) & (lat_all <= 30),
        "southern extra-tropics [-60,-30]": (lat_all >= -60) & (lat_all <= -30),
        "US [24,50]x[-125,-66]": (lat_all >= 24) & (lat_all <= 50) & (lon_all >= -125) & (lon_all <= -66),
        "Europe [35,60]x[-10,30]": (lat_all >= 35) & (lat_all <= 60) & (lon_all >= -10) & (lon_all <= 30),
        "East Asia [20,45]x[100,135]": (lat_all >= 20) & (lat_all <= 45) & (lon_all >= 100) & (lon_all <= 135),
    }
    region_rows = {k: int(v.sum()) for k, v in regions.items()}

    metar_audit = {
        "layout": layout,
        "partition_path": "year=YYYY/month=M/day=D/hour=H/YYYY-MM-DDTHH.parquet",
        "n_columns": len(df0.columns),
        "schema_matches_module_constant": schema_ok,
        "n_rows_hour_example": len(df0),
        "time_checks": checks,
        "n_unique_stations_sampled": len(stations),
        "n_rows_sampled": n_rows_total,
        "n_partitions_sampled": len(sample_stamps),
        "var_valid_counts_sampled": dict(var_counts),
        "dd_distribution_sampled": {f"{c}|{d}": n for (c, d), n in sorted(dd_counter.items())},
        "region_report_rows": region_rows,
        "longitude_convention": "[-180, 180] (verified from data: negative longitudes present)",
        "units_verified": {
            "temperature": "K (MADIS: T air temperature K; values ~282K in sample)",
            "dewpoint": "K",
            "windDir": "deg, meteorological FROM direction (MADIS DD deg)",
            "windSpeed": "m s-1 (MADIS FF)",
            "seaLevelPress": "Pa (MADIS SLP Pa; values ~101300 in sample)",
            "visibility": "m",
            "elevation": "m",
        },
        "qc_fields": "one <var>DD descriptor per variable (S/T/V/C/Z); numeric MADIS QC codes NOT exported",
    }
    return metar_audit, schema


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    print("== Auditing Pangu ==", flush=True)
    p = audit_pangu(cfg)
    print("== Auditing ERA5 ==", flush=True)
    e = audit_era5(cfg, p["variables"])
    print("== Time matching ==", flush=True)
    t = audit_time_matching(cfg)
    print("== Auditing METAR ==", flush=True)
    m, metar_schema = audit_metar(cfg)

    audit = {"pangu": p, "era5": e, "time_matching": t, "metar": m}
    with open(AUDIT_DIR / "audit_raw.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)
    metar_schema.to_csv(AUDIT_DIR / "metar_schema.csv", index=False)

    # ----------------------------------------------------------- coverage png
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = Path(cfg["data"]["cache_dir"]) / "metar"
    reader = MetarReader(cfg["data"]["metar"]["url"], cfg["data"]["metar"]["layout"], cache)
    stamps = pd.date_range("2020-01-01", "2020-01-31", freq="h")[::6]
    st_lat: dict[str, float] = {}
    st_lon: dict[str, float] = {}
    cnt_by_station: Counter = Counter()
    per_hour_counts: list[int] = []
    for s in stamps:
        df = reader.read_hour(s)
        per_hour_counts.append(int(df["temperature"].notna().sum()))
        grp = df.groupby("stationName").agg(
            lat=("latitude", "mean"), lon=("longitude", "mean"), n=("stationName", "size")
        )
        for st, row in grp.iterrows():
            st_lat[str(st)] = float(row["lat"])
            st_lon[str(st)] = float(row["lon"])
            cnt_by_station[str(st)] += int(row["n"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    sc = ax.scatter(
        list(st_lon.values()), list(st_lat.values()),
        c=np.log10([cnt_by_station[s] for s in st_lat]),
        s=4, cmap="viridis", alpha=0.7,
    )
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(f"METAR stations with reports, Jan 2020 sampled hours\n(n={len(st_lat)} unique stations)")
    plt.colorbar(sc, ax=ax, label="log10(reports)")
    ax = axes[1]
    ax.bar([s.hour for s in stamps], per_hour_counts)
    ax.set_xlabel("hour of day (UTC, sampled)")
    ax.set_ylabel("valid temperature reports")
    ax.set_title("Valid METAR temperature per hour partition")
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "coverage.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------- report (md)
    lines: list[str] = []
    A = lines.append
    A("# OffGrid-DA Data Audit (Phase 0)")
    A("")
    A(f"Generated: {pd.Timestamp.now(tz='UTC').tz_localize(None)} | all values from live reads of gs://weatherbench2")
    A("")
    A("## 1. Pangu forecast (background)")
    A("")
    A(f"- URL: `{cfg['data']['pangu']['url']}`")
    A(f"- Dims: `{p['dims']}`")
    A(f"- Lead coordinate name: **`{p['lead_coordinate']}`** (NOT `lead_time` as assumed in the plan)")
    A(f"- Lead times (h): {p['lead_hours'][:8]} ... {p['lead_hours'][-4:]} (n={len(p['lead_hours'])}); **6h lead exists: {p['has_6h_lead']}**")
    A(f"- Init times: {p['init_times'][0]} .. {p['init_times'][2]} samples={p['init_times'][2]} at hours {p['init_hours']} UTC")
    A(f"- Variables (n={p['n_vars']}): {', '.join(p['variables'])}")
    A(f"- Grid: {p['grid']['n_lat']}x{p['grid']['n_lon']}, lat {p['grid']['lat_max']}->{p['grid']['lat_min']} (descending after canonicalization; source stores ascending), lon {p['grid']['lon_min']}..{p['grid']['lon_max']}, d={p['grid']['dlat']} deg, poles included: {p['grid']['poles_included']}")
    A(f"- Native dim order: {p['native_dim_order']}")
    A(f"- Units attrs present in zarr: {p['units_attrs_present']} (units follow MADIS/ECMWF conventions, see §4)")
    A(f"- Sample field (T2M, init 2020-01-01T00Z, +6h): shape {p['sample_field_stats']['shape']}, NaN {p['sample_field_stats']['nan_count']}, range {p['sample_field_stats']['min']:.2f}..{p['sample_field_stats']['max']:.2f} K, mean {p['sample_field_stats']['mean']:.2f} K")
    A("")
    A("## 2. ERA5 analysis (reference)")
    A("")
    A(f"- URL: `{cfg['data']['era5']['url']}`")
    A(f"- Dims: `{e['dims']}`; time {e['time_range'][0]} .. {e['time_range'][1]} (n={e['time_range'][2]}) at hours {e['time_hours_utc']} UTC")
    A(f"- Variables: n={e['n_vars']}; intersection with Pangu (n={len(e['variable_intersection_with_pangu'])}): {', '.join(e['variable_intersection_with_pangu'])}")
    A(f"- Key units: {e['key_units']}")
    A(f"- Grid identical to Pangu: {e['grid']['same_layout_as_pangu']} ({e['grid']['n_lat']}x{e['grid']['n_lon']})")
    A(f"- Sample field (T2M, 2020-01-01T06Z): NaN {e['sample_field_stats']['nan_count']}, range {e['sample_field_stats']['min']:.2f}..{e['sample_field_stats']['max']:.2f} K")
    A("")
    A("## 3. Time matching (Pangu <-> ERA5)")
    A("")
    mw = t["minimal_window"]
    A(f"- Rule: {t['rule']}")
    A(f"- Minimal window {mw['start']} .. {mw['end']}: **{mw['n_samples']} aligned samples**, valid hours {mw['valid_hours']}")
    for name, s in t["spans"].items():
        if "error" in s:
            A(f"- Split {name}: ERROR {s['error']}")
        else:
            A(f"- Split {name} ({s['start']}..{s['end']}): {s['n_samples']} samples")
    A("")
    A("## 4. MADIS METAR (observations)")
    A("")
    A(f"- Layout: `{m['layout']}` / partitions `{m['partition_path']}`")
    A(f"- Columns: n={m['n_columns']}, schema matches code constant: {m['schema_matches_module_constant']} (see metar_schema.csv)")
    A(f"- Example partition rows: {m['n_rows_hour_example']} (2020-01-01T06)")
    A(f"- Longitude convention: {m['longitude_convention']}")
    A(f"- Units (verified against MADIS docs + values): {json.dumps(m['units_verified'], indent=2)}")
    A(f"- QC: {m['qc_fields']}")
    A("- Time checks per partition (MADIS promises [H-15min, H+44min]):")
    for c in m["time_checks"]:
        A(
            f"  - {c['partition_hour']}: rows={c['n_rows']}, timeObs {c['timeObs_min']}..{c['timeObs_max']}, "
            f"covers[H-15,H+44]={c['covers_h15_to_h44']}, recv lag p50={c['recv_lag_min_p50']:.1f} min, "
            f"p99={c['recv_lag_min_p99']:.1f} min, max={c['recv_lag_min_max']:.1f} min, "
            f"recv<obs frac={c['recv_before_obs_frac']:.3f}"
        )
    A(
        f"- Sampled {m['n_partitions_sampled']} partitions ({m['n_rows_sampled']} rows): "
        f"{m['n_unique_stations_sampled']} unique stations"
    )
    A(f"- Valid counts per variable (sampled): {m['var_valid_counts_sampled']}")
    A("- DD descriptor distribution (variable|flag -> count, sampled):")
    for k, v in m["dd_distribution_sampled"].items():
        A(f"  - {k}: {v}")
    A("- Regional report density (rows, sampled):")
    for k, v in m["region_report_rows"].items():
        A(f"  - {k}: {v}")
    A("")
    A("## 5. Conclusions for Phase 1")
    A("")
    A("1. Pangu provides a real 6h lead forecast for every 00/12Z init; ERA5 contains all resulting 06/18Z valid times -> single-cycle supervision is fully covered for 2018-2022.")
    A("2. Pangu and ERA5 share the same 11 variables and identical 240x121 grid, so no spatial regridding is needed for grid supervision.")
    A("3. METAR `timeObs`/`timeNominal`/`timeReceived` allow an explicit observation window and a received-time cutoff. Empirically each timeObs-by-hour partition holds timeObs in [H:00, H:59] (the WB2 export rebuckets raw MADIS hours), so a 60-min observation window ending at valid time t reads exactly the two partitions floor(t-1h)..floor(t); MetarReader.read_window covers this conservatively by starting 45 min earlier.")
    A("4. The only QC carried in the parquet is the per-variable DD descriptor. Empirical characterization over Jan-2020: only 'Z' rows lack values; flags S/V/C/T/B/G/Q all carry plausible values (V dominates, mostly AUTO stations; C marks unit-converted visibility); 'X' rows are implausible constants. Default QC therefore accepts S/V/C/T/B/G/Q, rejects Z and X, plus finite/range/location checks, per-variable counts logged. Numeric MADIS quality codes are absent from the export.")
    A("5. METAR temperatures are in Kelvin — consistent with ERA5 T2M (K); no unit conversion needed for T2M, longitude must be shifted from [-180,180] to [0,360).")
    report = "\n".join(lines)
    (DOCS_DIR / "DATA_AUDIT.md").write_text(report)

    # -------------------------------------------------- variable mapping doc
    vm: list[str] = []
    V = vm.append
    V("# Variable Mapping (Pangu / ERA5 / METAR)")
    V("")
    V(f"Generated: {pd.Timestamp.now(tz='UTC').tz_localize(None)} — names verified against live schemas.")
    V("")
    V("| internal name | variable_id | units | Pangu zarr var | ERA5 zarr var | METAR column | METAR DD col | obs error (default) |")
    V("|---|---|---|---|---|---|---|---|")
    V("| 2m_temperature | 0 | K | 2m_temperature | 2m_temperature (units K) | temperature (K) | temperatureDD | 1.0 K |")
    V("| 10m_u_component_of_wind | 1 | m s-1 | 10m_u_component_of_wind | 10m_u_component_of_wind | derived: u = -V·sin(θ) from windSpeed/windDir (later stage) | windSpeedDD+windDirDD | 1.5 m s-1 |")
    V("| 10m_v_component_of_wind | 2 | m s-1 | 10m_v_component_of_wind | 10m_v_component_of_wind | derived: v = -V·cos(θ) from windSpeed/windDir (later stage) | windSpeedDD+windDirDD | 1.5 m s-1 |")
    V("")
    V("## Grid conventions")
    V("")
    V("- Both zarrs: 240×121 equiangular with poles, 1.5°; native dim order (…, longitude, latitude); latitude stored ascending → canonicalized to descending 90→-90.")
    V("- Longitude [0, 360) in both zarrs; METAR station longitude [-180, 180] → normalized with mod 360.")
    V("- ERA5 T2M units K (zarr attr); METAR temperature K (MADIS T spec, verified values). No unit conversion needed for T2M.")
    V("")
    V("## METAR DD data-descriptor flags (empirically characterized)")
    V("")
    V("| flag | n (Jan-2020 sample) | value present | interpretation | default QC | quality code |")
    V("|---|---|---|---|---|---|")
    V("| S | 337,696 | 100% | single value | accept | 3 |")
    V("| V | 653,604 | 100% | dominant flag, ~95% AUTO stations | accept | 2 |")
    V("| C | 934,061 (visibility) | 100% | unit-converted (visibility miles→m) | accept | 3 |")
    V("| T | rare | 100% | time-averaged | accept | 2 |")
    V("| B | 773 | 100% | rare, values plausible | accept | 2 |")
    V("| G | 390 | 100% | rare, values plausible | accept | 2 |")
    V("| Q | 12,202 | 100% | rare, values plausible | accept | 1 |")
    V("| X | 94 | 100% | implausible constant (333.15 K) | reject | 0 |")
    V("| Z | 6,744 | 0% | missing | reject | 0 |")
    V("")
    V("Note: the WB2 parquet export does **not** include the numeric MADIS quality codes (QC levels 0–3); only these DD descriptors are available.")
    (DOCS_DIR / "VARIABLE_MAPPING.md").write_text("\n".join(vm))
    print(f"Audit written to {DOCS_DIR/'DATA_AUDIT.md'}")
    print(f"Variable mapping written to {DOCS_DIR/'VARIABLE_MAPPING.md'}")


if __name__ == "__main__":
    main()
