"""Schema tests against real remote data (marked requires_gcs)."""
from __future__ import annotations

import numpy as np
import pytest

from offgrid_da.data.metar import METAR_COLUMNS

pytestmark = pytest.mark.requires_gcs


def test_metar_schema_real(metar_hour):
    assert list(metar_hour.columns) == METAR_COLUMNS
    for col in ("stationName", "latitude", "longitude", "elevation", "timeObs",
                "timeNominal", "timeReceived", "temperature", "temperatureDD"):
        assert col in metar_hour.columns
    assert len(metar_hour) > 1000


def test_metar_longitude_convention(metar_hour):
    lon = metar_hour["longitude"].to_numpy()
    assert lon.min() >= -180.0 and lon.max() <= 180.0
    assert (lon < 0).any()  # negative longitudes exist -> convention [-180, 180]


def test_pangu_schema(cfg):
    from offgrid_da.data.forecast import PanguReader

    p = PanguReader(cfg["data"]["pangu"]["url"])
    assert "prediction_timedelta" in p.ds.coords
    assert "lead_time" not in p.ds.coords  # verified absent in the real zarr
    assert p.ds.sizes["longitude"] == 240 and p.ds.sizes["latitude"] == 121
    assert 6 in set(int(x) for x in p.lead_hours)


def test_era5_schema(cfg):
    from offgrid_da.data.analysis import ERA5Reader

    e = ERA5Reader(cfg["data"]["era5"]["url"])
    assert e.ds.sizes["longitude"] == 240 and e.ds.sizes["latitude"] == 121
    assert "2m_temperature" in e.ds
    assert e.ds["2m_temperature"].attrs.get("units") == "K"
    assert sorted(set(e.times.hour)) == [0, 6, 12, 18]


def test_canonical_grid_identical(cfg):
    from offgrid_da.data.analysis import ERA5Reader
    from offgrid_da.data.forecast import PanguReader

    pg = PanguReader(cfg["data"]["pangu"]["url"]).grid()
    eg = ERA5Reader(cfg["data"]["era5"]["url"]).grid()
    assert np.allclose(pg["latitude"], eg["latitude"])
    assert np.allclose(pg["longitude"], eg["longitude"])
    # canonical orientation: descending latitude
    assert pg["latitude"][0] == 90.0 and pg["latitude"][-1] == -90.0
