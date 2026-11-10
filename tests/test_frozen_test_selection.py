import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_single_cycle_test import assert_selected_checkpoint


def checkpoint():
    return {'step': 39456, 'signature': {'config_sha256': 'fixed-config',
                                       'data': {'train': 'train-hash', 'val': 'val-hash'}, 'trial': False}}


def test_accepts_only_validation_selected_weights():
    assert_selected_checkpoint(checkpoint(), {'best_step': 39456}, 'fixed-config',
                               {'train': 'train-hash', 'val': 'val-hash'})


@pytest.mark.parametrize('case', ['step', 'config', 'data', 'trial'])
def test_rejects_alternative_or_mismatched_weights(case):
    c = copy.deepcopy(checkpoint())
    if case == 'step':
        c['step'] = 43840
    elif case == 'config':
        c['signature']['config_sha256'] = 'modified-config'
    elif case == 'data':
        c['signature']['data']['val'] = 'another-year'
    else:
        c['signature']['trial'] = True
    with pytest.raises(ValueError):
        assert_selected_checkpoint(c, {'best_step': 39456}, 'fixed-config',
                                   {'train': 'train-hash', 'val': 'val-hash'})

