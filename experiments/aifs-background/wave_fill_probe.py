#!/usr/bin/env python3
"""Wave-band fill pre-experiment (P1 gate for the 6 missing h* channels).

ARCO v3 lacks the 6 ECMWF half-period wave-band channels (h1012..h2530) that
AIFS-single-2.0 consumes. Before committing to filling them (climatology/zero)
in historical initial states, quantify their impact on the 82 atmospheric
channels we actually consume:

  forecast A: Open Data initial state, full (real h* values)
  forecast B: same state, h* channels zeroed (worst-case fill)
  compare    : per-channel RMS/Max |A - B| on the 82ch set, against the
               scale of the model's own 6h error (e.g. T2m ~0.45 K)

Run: .venv/bin/python wave_fill_probe.py [--date ...]
"""
import argparse
import copy
import numpy as np

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners.default import DefaultRunner

H_BANDS = ["h1012", "h1214", "h1417", "h1721", "h2125", "h2530"]
CH82 = (["2t", "10u", "10v", "msl"]
        + [f"{v}_{lv}" for lv in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
           for v in ("u", "v", "w", "t", "q", "z")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-time", default="6h")
    args = ap.parse_args()
    overrides = [f"lead_time={args.lead_time}"]
    config = RunConfiguration.load("inference.yaml", overrides)
    runner = DefaultRunner(config)

    prognostic = runner.create_prognostics_input().create_input_state(date=getattr(config, "date", None))
    constants = runner.create_constant_coupled_forcings_input().create_input_state(date=getattr(config, "date", None))
    forcings = runner.create_dynamic_forcings_input().create_input_state(date=getattr(config, "date", None))
    full_state = runner._combine_states(prognostic, constants, forcings)

    missing = [h for h in H_BANDS if h not in full_state["fields"]]
    print("date:", full_state.get("date"), "| h* present:", [h for h in H_BANDS if h in full_state["fields"]])
    assert not missing, f"h* channels not in initial state: {missing}"

    # A: full initial state
    out_a = list(runner.run(input_state=full_state, lead_time=args.lead_time))[-1]["fields"]

    # B: h* zeroed (worst-case fill) — shallow structural copy; earthkit GRIB
    # objects in the state are not deep-copyable, and run() copies defensively.
    zero_state = dict(full_state)
    zero_state["fields"] = dict(full_state["fields"])
    for h in H_BANDS:
        zero_state["fields"][h] = np.zeros_like(np.asarray(zero_state["fields"][h]))
    out_b = list(runner.run(input_state=zero_state, lead_time=args.lead_time))[-1]["fields"]

    print(f"{'channel':>8} {'rms(A-B)':>10} {'max|A-B|':>10}   (A value rms)")
    rows = {}
    for name in CH82:
        a, b = np.asarray(out_a[name], dtype=np.float64), np.asarray(out_b[name], dtype=np.float64)
        d = a - b
        w = np.cos(np.deg2rad(np.linspace(90, -90, d.shape[0])))[:, None] if d.ndim == 2 else None
        rms = float(np.sqrt((d ** 2).mean()))
        rows[name] = {"rms": rms, "max": float(np.abs(d).max()), "scale": float(np.sqrt((a ** 2).mean()))}
        print(f"{name:>8} {rows[name]['rms']:10.6g} {rows[name]['max']:10.6g}   {rows[name]['scale']:10.4g}")

    import json
    json.dump({"date": str(full_state.get("date")), "lead_time": args.lead_time,
               "h_bands_zeroed": H_BANDS, "channels": rows},
              open("wave_fill_probe.json", "w"), indent=1)
    worst = max(rows, key=lambda k: rows[k]["rms"] / max(rows[k]["scale"], 1e-12))
    print("\nworst relative channel:", worst, "rel rms = %.3g" % (rows[worst]["rms"] / rows[worst]["scale"]))


if __name__ == "__main__":
    main()
