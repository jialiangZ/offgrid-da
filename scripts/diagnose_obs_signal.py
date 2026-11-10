#!/usr/bin/env python3
"""Step-1 diagnostic of the synthetic-observation experiment plan (§3).

Directly measures, at station locations, for the 2021 validation year:
  d     = y_METAR - H(x_ref)          (obs - reference discrepancy)
  r     = H(x_ref) - H(xb)            (background error = correction target)
  innov = y_METAR - H(xb) = r + d     (real innovation the DA network sees)

H is the same periodic-longitude bilinear grid->station interpolation used by
evaluate_baseline0.py. Statistics are reported pooled and per group (month,
latitude band, station elevation band, |time offset| band), for context
(DA-visible) and heldout stations separately. Per-cycle correlations
corr(innov, r) are the primary utility measure; day-block bootstrap gives
uncertainty. No parameters are fitted here; this is pure diagnosis.

Outputs: outputs/obs_diagnostics_2021/summary.json and report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.alignment import sample_background_at_obs  # noqa: E402

LAT_BANDS = [(60, 90, "60-90N"), (30, 60, "30-60N"), (0, 30, "0-30N"),
             (-30, 0, "30S-0"), (-60, -30, "60S-30S"), (-90, -60, "90S-60S")]
ELEV_BANDS = [(0, 100, "0-100m"), (100, 500, "100-500m"),
              (500, 1500, "500-1500m"), (1500, 1e9, ">1500m")]
OFFSET_BANDS = [(-59.01, -30, "30-59min"), (-30, 0.01, "0-30min")]


def band_of(value, bands):
    for lo, hi, name in bands:
        if lo <= value < hi:
            return name
    return bands[-1][2]


def cycle_stats(y, ref, bg):
    """Per-cycle statistics for one obs group."""
    d = y - ref
    r = ref - bg
    innov = y - bg
    n = len(y)
    out = {
        "n": n,
        "mean_d": float(d.mean()), "std_d": float(d.std()),
        "mean_r": float(r.mean()), "std_r": float(r.std()),
        "mean_innov": float(innov.mean()), "std_innov": float(innov.std()),
        "rms_d": float(np.sqrt((d ** 2).mean())),
        "rms_r": float(np.sqrt((r ** 2).mean())),
        "rms_innov": float(np.sqrt((innov ** 2).mean())),
    }
    if n >= 100 and out["std_innov"] > 0 and out["std_r"] > 0:
        corr_ir = float(np.corrcoef(innov, r)[0, 1])
        corr_dr = float(np.corrcoef(d, r)[0, 1]) if out["std_d"] > 0 else 0.0
        out["corr_innov_r"] = corr_ir
        out["corr_d_r"] = corr_dr
        # optimal linear estimate of r from innov: r_hat = a*innov + b
        a = float(np.polyfit(innov, r, 1)[0])
        resid = r - (a * innov + r.mean() - a * innov.mean())
        out["lin_beta"] = a
        out["resid_std_r"] = float(resid.std())
        out["frac_r_explained"] = max(0.0, 1.0 - float(resid.var()) / float(r.var()))
    return out


def pooled(stats_list, key_corr="corr_innov_r"):
    n = sum(s["n"] for s in stats_list)
    rms_d = math.sqrt(sum(s["n"] * s["rms_d"] ** 2 for s in stats_list) / n)
    rms_r = math.sqrt(sum(s["n"] * s["rms_r"] ** 2 for s in stats_list) / n)
    rms_i = math.sqrt(sum(s["n"] * s["rms_innov"] ** 2 for s in stats_list) / n)
    corrs = [s[key_corr] for s in stats_list if key_corr in s]
    # variance decomposition: Var(innov) = Var(r) + Var(d) + 2Cov
    var_r = sum(s["n"] * s["std_r"] ** 2 for s in stats_list) / n
    var_d = sum(s["n"] * s["std_d"] ** 2 for s in stats_list) / n
    var_i = sum(s["n"] * s["std_innov"] ** 2 for s in stats_list) / n
    frac = [s["frac_r_explained"] for s in stats_list if "frac_r_explained" in s]
    return {
        "n_obs": n, "n_cycles": len(stats_list),
        "rms_d": rms_d, "rms_r": rms_r, "rms_innov": rms_i,
        "var_r": var_r, "var_d": var_d, "var_innov": var_i,
        "snr_var_r_over_var_innov": var_r / var_i,
        "corr_d_r_weighted_implied": (var_i - var_r - var_d) / (2 * math.sqrt(var_r * var_d)) if var_r > 0 and var_d > 0 else float("nan"),
        "per_cycle_corr_median": float(np.median(corrs)) if corrs else None,
        "per_cycle_corr_q25_q75": [float(np.percentile(corrs, 25)), float(np.percentile(corrs, 75))] if corrs else None,
        "per_cycle_frac_r_explained_median": float(np.median(frac)) if frac else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-glob",
                    default="outputs/da_sample_cache/4868f5c1134c1f1bcf4a8ea5af0ed7384804b72f7a7d6d5fdb35c0944c6117b9/arches_2021*.pt")
    ap.add_argument("--out", default="outputs/obs_diagnostics_2021")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.cache_glob))
    if not files:
        raise SystemExit("no sample files matched")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cycles = []           # per-cycle records for context and heldout
    per_group = defaultdict(lambda: defaultdict(list))  # group_type -> group_name -> list[cycle_stats]
    obs_rows = {"context": [], "heldout": []}           # per-cycle arrays for bootstrap

    for path in files:
        s = torch.load(path, map_location="cpu", weights_only=False)
        valid_time = s["valid_time"]
        if isinstance(valid_time, torch.Tensor):
            valid_time = valid_time.item()
        vt = valid_time if isinstance(valid_time, datetime) else datetime.fromtimestamp(valid_time.timestamp())
        bg = s["background"][0].numpy().astype(np.float64)
        ref = s["reference"][0].numpy().astype(np.float64)
        glat = s["grid_latitude"].numpy().astype(np.float64)
        glon = s["grid_longitude"].numpy().astype(np.float64)
        rec = {"valid_time": str(vt), "month": vt.month}
        for grp, prefix in (("context", ""), ("heldout", "heldout_")):
            if s[f"{prefix}obs_values"].numel() == 0:
                continue
            y = s[f"{prefix}obs_values"].numpy().astype(np.float64)
            loc = s[f"{prefix}obs_locations"].numpy().astype(np.float64)
            toff = s[f"{prefix}obs_time_offsets"].numpy().astype(np.float64)
            hbg = sample_background_at_obs(bg, glat, glon, loc[:, 0], loc[:, 1])
            href = sample_background_at_obs(ref, glat, glon, loc[:, 0], loc[:, 1])
            ok = np.isfinite(hbg) & np.isfinite(href)
            y, href, hbg = y[ok], href[ok], hbg[ok]
            loc, toff = loc[ok], toff[ok]
            st = cycle_stats(y, href, hbg)
            st["month"] = vt.month
            rec[grp] = st
            obs_rows[grp].append({
                "d": y - href, "r": href - hbg, "innov": y - hbg,
                "lat": loc[:, 0], "elev": loc[:, 2], "toff": toff,
            })
            # vectorized grouping accumulation
            latb = np.array([band_of(v, LAT_BANDS) for v in loc[:, 0]])
            elevb = np.array([band_of(v, ELEV_BANDS) for v in loc[:, 2]])
            offb = np.array([band_of(v, OFFSET_BANDS) for v in toff])
            d_v, r_v = y - href, href - hbg
            for name, arr in (("lat", latb), ("elev", elevb), ("offset", offb)):
                for b in np.unique(arr):
                    m = arr == b
                    sub = cycle_stats(d_v[m] + href[m], href[m], href[m] - r_v[m])
                    sub["month"] = vt.month
                    per_group[name][str(b)].append(sub)
            per_group["month"][str(vt.month)].append(st)
        cycles.append(rec)

    # pooled + bootstrap
    summary = {"year": 2021, "variable": "T2M (K)", "n_files": len(files)}
    rng = np.random.default_rng(0)
    for grp in ("context", "heldout"):
        rows = obs_rows[grp]
        stats_all = [c[grp] for c in cycles if grp in c]
        summary[grp] = pooled(stats_all)
        # day-block bootstrap over cycles
        day_stats = defaultdict(list)
        for c in cycles:
            if grp in c:
                day_stats[c["valid_time"][:10]].append(c[grp])
        days = sorted(day_stats)
        med_corrs, snrs = [], []
        for _ in range(args.bootstrap):
            sub = [s for di in rng.integers(0, len(days), len(days))
                   for s in day_stats[days[di]]]
            p = pooled(sub)
            if p["per_cycle_corr_median"] is not None:
                med_corrs.append(p["per_cycle_corr_median"])
                snrs.append(p["snr_var_r_over_var_innov"])
        if med_corrs:
            summary[grp]["corr_median_CI95"] = [float(np.percentile(med_corrs, 2.5)),
                                                float(np.percentile(med_corrs, 97.5))]
            summary[grp]["snr_CI95"] = [float(np.percentile(snrs, 2.5)),
                                        float(np.percentile(snrs, 97.5))]
        for gtype in ("month", "lat", "elev", "offset"):
            if gtype in per_group:
                summary[grp][f"by_{gtype}"] = {
                    k: pooled(v) for k, v in sorted(per_group[gtype].items(),
                                                    key=lambda kv: int(kv[0]) if gtype == "month" else kv[0])
                }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")

    # compact report
    lines = ["# Step-1 obs-signal diagnostics (T2M, 2021)", ""]
    for grp in ("context", "heldout"):
        p = summary[grp]
        lines.append(f"## {grp} stations")
        lines.append(f"- n_obs={p['n_obs']}, cycles={p['n_cycles']}")
        lines.append(f"- RMS d={p['rms_d']:.3f} K, RMS r={p['rms_r']:.3f} K, RMS innov={p['rms_innov']:.3f} K")
        lines.append(f"- SNR Var(r)/Var(innov)={p['snr_var_r_over_var_innov']*100:.1f}%  CI95={np.round(np.asarray(p.get('snr_CI95',[float('nan')]))*100,1).tolist()}")
        if p["per_cycle_corr_median"] is not None:
            lines.append(f"- per-cycle corr(innov,r): median={p['per_cycle_corr_median']:.3f} "
                         f"IQR={np.round(p['per_cycle_corr_q25_q75'],3).tolist()} "
                         f"bootstrap CI95={np.round(p.get('corr_median_CI95',[float('nan')]),3).tolist()}")
            lines.append(f"- per-cycle optimal-linear fraction of Var(r) explained: median={p['per_cycle_frac_r_explained_median']*100:.1f}%")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written:", out_dir)


if __name__ == "__main__":
    main()
