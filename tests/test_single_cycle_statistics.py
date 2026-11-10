"""Independent known-answer checks of paired pooled-RMSE block resampling."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_single_cycle import circular_block_bootstrap


def test_identical_methods_have_exact_zero_paired_intervals():
    weights = np.arange(1., 61.)
    values = np.column_stack([weights * 2] * 3)
    report = circular_block_bootstrap(values, weights, repetitions=100)
    for block in report.values():
        for interval in block.values():
            assert interval == {'lower_95_K': 0., 'upper_95_K': 0.}


def test_constant_known_rmse_differences_ignore_unequal_daily_weights():
    weights = np.arange(1., 61.)
    # Background RMSE=3 K, NC=2 K, DA=1 K regardless of resampled blocks.
    values = weights[:, None] * np.array([9., 4., 1.])[None, :]
    report = circular_block_bootstrap(values, weights, repetitions=100)
    for block in report.values():
        for key, expected in [('DA_minus_background', -2.), ('DA_minus_neural_correction', -1.)]:
            assert block[key]['lower_95_K'] == pytest.approx(expected, abs=1e-12)
            assert block[key]['upper_95_K'] == pytest.approx(expected, abs=1e-12)

