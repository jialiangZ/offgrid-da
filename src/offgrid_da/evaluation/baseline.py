"""Simple deterministic baselines for single-cycle DA."""
from __future__ import annotations

import numpy as np


def baseline0(background: np.ndarray) -> np.ndarray:
    """Baseline 0: return the background unchanged as the analysis estimate."""
    return np.asarray(background)
