"""Forecast lead-time availability (Pangu prediction_timedelta)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_gcs


def test_pangu_leads(cfg):
    from offgrid_da.data.forecast import PanguReader

    p = PanguReader(cfg["data"]["pangu"]["url"])
    leads = set(int(x) for x in p.lead_hours)
    assert 6 in leads
    assert min(leads) == 6 and max(leads) == 240
    assert leads == set(range(6, 241, 6))  # verified: 40 values, 6h steps


def test_init_hours(cfg):
    from offgrid_da.data.forecast import PanguReader

    p = PanguReader(cfg["data"]["pangu"]["url"])
    assert set(p.init_times.hour) == {0, 12}


def test_single_lead_read_roundtrip(cfg):
    from offgrid_da.data.forecast import PanguReader

    p = PanguReader(cfg["data"]["pangu"]["url"])
    a = p.read_background("2020-01-05T00:00", 6, ["2m_temperature"])
    b = p.read_background("2020-01-05T00:00", 6, ["2m_temperature"])
    assert a.shape == (1, 121, 240)
    assert np.array_equal(a, b)
    with pytest.raises(KeyError):
        p.read_background("2020-01-05T00:00", 7, ["2m_temperature"])
