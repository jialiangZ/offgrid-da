"""Read-only inspection of the local ArchesWeather forecast interface."""
import hashlib
import json
import sys
from pathlib import Path

import torch
import xarray as xr
import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from offgrid_da.data.forecast_contract import archesweather_t2m_background, replace_archesweather_t2m
experiment = root / "experiments/archesweather-reproduction"
repo = experiment / "geoarches"
config_path = repo / "modelstore/archesweather-m-20y-6h-seed0-continue150k/config.yaml"
config = yaml.safe_load(config_path.read_text())
stats_path = Path(config["dataloader"]["dataset"]["norm_stats_path"])
stats = torch.load(stats_path, map_location="cpu", weights_only=True)
nc_path = experiment / "data/era5_240/full/era5_240_2018_6h.nc"
with xr.open_dataset(nc_path) as ds:
    fields = {name: {"units": ds[name].attrs.get("units"), "dims": list(ds[name].dims)}
              for name in ds.data_vars}
    coords = {name: ds[name].values.tolist() for name in ("latitude", "longitude", "level")}
    physical_t2m = torch.tensor(ds["2m_temperature"].isel(time=0).transpose("latitude","longitude").values.copy())
    if ds.latitude[0] < ds.latitude[-1]:
        physical_t2m = physical_t2m.flip(0)
    native_surface = torch.zeros(1,4,1,121,240)
    native_surface[:,2] = ((physical_t2m.roll(120,-1)-stats["surface_mean"][2,0,0,0])
                           / stats["surface_std"][2,0,0,0])
    dummy_next_state = {"surface": native_surface, "level": torch.zeros(1,6,13,121,240)}
    converted = archesweather_t2m_background(dummy_next_state, stats, ["2018-01-01T00:00Z"])
    error = float((converted["background"][0,0]-physical_t2m).abs().max())
    assert error < 1e-3
    restored = replace_archesweather_t2m(dummy_next_state, converted["background"], stats)
    torch.testing.assert_close(restored["surface"], native_surface, atol=1e-5, rtol=1e-5)
report = {
    "config_path": str(config_path), "stats_path": str(stats_path),
    "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    "lead_time_hours": config["dataloader"]["dataset"]["lead_time_hours"],
    "load_prev": config["dataloader"]["dataset"]["load_prev"],
    "stats_tensors": {k: {"shape": list(v.shape), "finite": bool(v.isfinite().all())}
                      for k,v in stats.items() if isinstance(v, torch.Tensor)},
    "sample_nc": str(nc_path), "fields": fields, "coords": coords,
    "conversion_check": {"max_kelvin_error": error,
                         "native_roundtrip": "passed",
                         "scope": "real T2M conversion only; synthetic normalized state, not trained-model inference"},
}
output = root / "outputs/forecast_contract"
output.mkdir(parents=True, exist_ok=True)
(output / "probe.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
