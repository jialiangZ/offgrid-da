"""Latitude/longitude mapping: canonical orientation, normalization, obs grid sanity."""
from __future__ import annotations

import numpy as np
import pytest

from offgrid_da.data.alignment import normalize_longitude_360
from offgrid_da.data.metar import MetarReader

pytestmark = pytest.mark.requires_gcs


def test_reader_grid_canonical(dataset):
    lat = dataset.pangu.grid()["latitude"]
    lon = dataset.pangu.grid()["longitude"]
    assert lat[0] == 90 and lat[-1] == -90
    assert (np.diff(lat) < 0).all()
    assert lon[0] == 0 and lon[-1] == 358.5
    assert (np.diff(lon) > 0).all()


def test_obs_locations_in_canonical_range(dataset):
    s = dataset[0]
    loc = s["obs_locations"].numpy()
    assert (loc[:, 0] >= -90).all() and (loc[:, 0] <= 90).all()
    assert (loc[:, 1] >= 0).all() and (loc[:, 1] < 360).all()
    assert np.isfinite(loc).all()


def test_obs_longitude_matches_station_geography(dataset, cfg):
    """Context vs held-out: obs must come from context stations only (checked in
    test_station_split); here verify longitudes normalized from [-180,180]."""
    reader = MetarReader(cfg["data"]["metar"]["url"], cfg["data"]["metar"]["layout"], cfg["data"]["cache_dir"])
    df = reader.read_hour(dataset.sample_table()["valid_time"].iloc[0])
    lon_west = df["longitude"].to_numpy()
    normalized = normalize_longitude_360(lon_west)
    assert (normalized >= 0).all() and (normalized < 360).all()
    # a station at -70 (Santiago area) must map to 290
    m = lon_west < -60
    if m.any():
        assert np.allclose(normalized[m], lon_west[m] + 360.0)


def test_field_rows_correspond_to_latitudes(dataset):
    """Sanity: background row at lat 90 must equal a direct zarr read of the pole row."""
    from offgrid_da.data.forecast import PanguReader

    row = dataset.sample_table().iloc[0]
    p = PanguReader(dataset.cfg["data"]["pangu"]["url"])
    da = p.ds["2m_temperature"].sel(
        time=np.datetime64(row["init_time"], "ns"),
        prediction_timedelta=6,
    ).values  # (longitude, latitude); last latitude is -90 in source
    canonical = dataset[0]["background"][0].numpy()
    # canonical row 120 (=-90) == source column at latitude=-90
    assert np.allclose(canonical[120], da[:, 0], atol=0)  # source lat ascending -> col 0 is -90
    assert np.allclose(canonical[0], da[:, -1], atol=0)   # col -1 is +90
