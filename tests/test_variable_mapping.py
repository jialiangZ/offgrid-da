"""Variable mapping consistency (registry <-> real dataset names)."""
from __future__ import annotations

import pytest

from offgrid_da.data.variables import (
    DD_QUALITY_CODE,
    GRID_VARIABLE_MAP,
    DEFAULT_REGISTRY,
)

pytestmark = pytest.mark.requires_gcs


def test_registry_ids_unique():
    ids = [s.variable_id for s in DEFAULT_REGISTRY.variables.values()]
    assert sorted(ids) == list(range(len(ids)))


def test_t2m_mapping_real_names(cfg):
    from offgrid_da.data.analysis import ERA5Reader
    from offgrid_da.data.forecast import PanguReader

    pangu_vars = set(PanguReader(cfg["data"]["pangu"]["url"]).ds.data_vars)
    era5_vars = set(ERA5Reader(cfg["data"]["era5"]["url"]).ds.data_vars)
    for internal, names in GRID_VARIABLE_MAP.items():
        assert names["pangu"] in pangu_vars, f"{internal}: {names['pangu']} missing in Pangu"
        assert names["era5"] in era5_vars, f"{internal}: {names['era5']} missing in ERA5"


def test_metar_column_exists_in_real_schema(cfg):
    from offgrid_da.data.metar import MetarReader

    reader = MetarReader(cfg["data"]["metar"]["url"], cfg["data"]["metar"]["layout"], cfg["data"]["cache_dir"])
    import pandas as pd

    df = reader.read_hour(pd.Timestamp("2020-01-01T06"))
    for spec in DEFAULT_REGISTRY.variables.values():
        if spec.metar_column is not None:
            assert spec.metar_column in df.columns
            assert spec.metar_dd_column in df.columns


def test_quality_codes_consistent():
    for flag, code in DD_QUALITY_CODE.items():
        assert isinstance(code, int) and 0 <= code <= 3
    assert DD_QUALITY_CODE["Z"] == 0  # missing
    assert DD_QUALITY_CODE["X"] == 0  # implausible extremes
