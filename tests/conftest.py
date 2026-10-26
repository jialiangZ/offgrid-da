"""Shared fixtures.

Remote-data fixtures use a narrow 3-day window (2020-01-01..03, 6 samples)
so the suite stays fast; parquet partitions and zarr chunks are cached on
disk after the first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offgrid_da.data.dataset import SingleCycleDADataset  # noqa: E402
from offgrid_da.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset(cfg):
    return SingleCycleDADataset(cfg, split="minimal", start="2020-01-01", end="2020-01-03")


@pytest.fixture(scope="session")
def sample(dataset) -> dict:
    return dataset[0]


@pytest.fixture(scope="session")
def metar_hour(cfg):
    from offgrid_da.data.metar import MetarReader

    reader = MetarReader(cfg["data"]["metar"]["url"], cfg["data"]["metar"]["layout"], cfg["data"]["cache_dir"])
    return reader.read_hour(pd.Timestamp("2020-01-01T06"))
