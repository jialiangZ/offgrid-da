"""Normalization statistics (train-only computation, save/load roundtrip)."""
from __future__ import annotations

import numpy as np

from offgrid_da.utils.normalization import NormStats


def test_update_and_stats():
    st = NormStats()
    st.update("background", "2m_temperature", np.array([270.0, 280.0, 290.0]))
    assert abs(st.mean("background", "2m_temperature") - 280.0) < 1e-12
    assert abs(st.std("background", "2m_temperature") - 10.0) < 1e-12


def test_save_load_roundtrip(tmp_path):
    st = NormStats()
    rng = np.random.default_rng(0)
    data = rng.normal(277.0, 20.0, 1000)
    st.update("reference", "2m_temperature", data)
    p = tmp_path / "stats.json"
    st.save(p)
    st2 = NormStats.load(p)
    assert abs(st2.mean("reference", "2m_temperature") - st.mean("reference", "2m_temperature")) < 1e-9
    assert abs(st2.std("reference", "2m_temperature") - st.std("reference", "2m_temperature")) < 1e-9


def test_nan_ignored():
    st = NormStats()
    st.update("g", "v", np.array([1.0, np.nan, 3.0]))
    assert abs(st.mean("g", "v") - 2.0) < 1e-12


def test_empty_raises():
    import pytest

    st = NormStats()
    with pytest.raises(ValueError):
        st.update("g", "v", np.array([]))
