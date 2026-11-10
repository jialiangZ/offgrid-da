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


def test_batch_updates_match_numpy_statistics():
    st = NormStats()
    first = np.array([270.0, 275.0, np.nan, 280.0])
    second = np.array([260.0, 290.0, 282.0])
    st.update("g", "v", first)
    st.update("g", "v", second)
    combined = np.concatenate([first, second])
    assert abs(st.mean("g", "v") - np.nanmean(combined)) < 1e-12
    assert abs(st.std("g", "v") - np.nanstd(combined, ddof=1)) < 1e-12


def test_merge_sufficient_statistics_matches_direct_update():
    left = NormStats()
    right = NormStats()
    merged = NormStats()
    a = np.array([250.0, 270.0, 280.0])
    b = np.array([290.0, 300.0, np.nan])
    left.update("g", "v", a)
    right.update("g", "v", b)
    merged.merge(left)
    merged.merge(right)
    combined = np.concatenate([a, b])
    assert abs(merged.mean("g", "v") - np.nanmean(combined)) < 1e-12
    assert abs(merged.std("g", "v") - np.nanstd(combined, ddof=1)) < 1e-12


def test_normalize_uses_training_mean_and_sample_std():
    stats = NormStats()
    stats.update("background", "2m_temperature", np.array([270.0, 280.0, 290.0]))
    normalized = stats.normalize(
        "background", "2m_temperature", np.array([280.0, 290.0], dtype=np.float32)
    )
    np.testing.assert_allclose(normalized, [0.0, 1.0], rtol=1e-6, atol=1e-6)


def test_normalize_rejects_zero_variance():
    import pytest

    stats = NormStats()
    stats.update("background", "constant", np.array([4.0, 4.0, 4.0]))
    with pytest.raises(ValueError, match="invalid normalization std"):
        stats.normalize("background", "constant", np.array([4.0]))
