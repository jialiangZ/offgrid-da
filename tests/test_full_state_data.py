from pathlib import Path

import numpy as np
import pandas as pd
import torch

from offgrid_da.data.full_state_data import FullStateInputDataset, PRESSURE, digest, read_state


def files(tmp_path):
    surface = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None, None, None], (4, 1, 121, 240)).copy()
    level = np.broadcast_to((100 + np.arange(78, dtype=np.float32)).reshape(6, 13, 1, 1), (6, 13, 121, 240)).copy()
    path = tmp_path / 'full.npz'
    np.savez_compressed(path, surface=surface, level=level, latitude=np.linspace(90, -90, 121),
                        longitude=np.arange(240)*1.5, pressure_hpa=np.array(PRESSURE))
    cache = tmp_path / 'old.pt'
    torch.save({'valid_time': pd.Timestamp('2018-01-01'), 'cache_signature': 'old-cache',
                'background': torch.full((1, 121, 240), -999.),
                'reference_normalized': torch.ones(1, 121, 240),
                'obs_values_normalized': torch.tensor([.1, .2])}, cache)
    return path, cache


def test_flat_channel_order_is_surface_then_variable_major_pressure_minor(tmp_path):
    path, _ = files(tmp_path)
    value = read_state(path, digest(path))
    assert value.shape == (82, 121, 240)
    assert value[2, 0, 0] == 2
    assert value[4, 0, 0] == 100
    assert value[4+13, 0, 0] == 113
    assert value[-1, 0, 0] == 177


def test_input_mask_preserves_targets_observations_and_new_t2m_background(tmp_path):
    path, cache = files(tmp_path)
    def dataset(mode):
        d = FullStateInputDataset.__new__(FullStateInputDataset)
        d.input_mode = mode
        d.cache_paths = [cache]
        d.legacy_signature = 'old-cache'
        d.records = [{'valid_utc': '2018-01-01T00:00:00Z', 'file': path.name, 'sha256': digest(path)}]
        d.background_dir = tmp_path
        d.mean = torch.zeros(82, 1, 1)
        d.std = torch.ones(82, 1, 1)
        return d
    a, b = dataset('full82')[0], dataset('t2m_masked')[0]
    assert torch.equal(a['background_normalized'], b['background_normalized'])
    assert torch.equal(a['background'], torch.full((1, 121, 240), 2.))
    assert torch.equal(a['reference_normalized'], b['reference_normalized'])
    assert torch.equal(a['obs_values_normalized'], b['obs_values_normalized'])
    assert torch.equal(a['grid_state'][2], b['grid_state'][2])
    other = [i for i in range(82) if i != 2]
    assert torch.count_nonzero(b['grid_state'][other]) == 0
    assert torch.count_nonzero(a['grid_state'][other]) > 0

