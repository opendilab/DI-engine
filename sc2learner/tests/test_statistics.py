import pytest
import torch
import torch.nn.functional as F
import numpy as np
import warnings

from sc2learner.data.fake_dataset import FakeActorDataset, fake_stat_processed


@pytest.fixture
def stat_processed():
    return fake_stat_processed()


def compare(a, b, ignore_list=[''], trace='ROOT'):
    print('Testing {}'.format(trace))
    if not type(a) == type(b):
        print(trace)
        print(a, b)
        return 1
    if isinstance(a, dict):
        for k, v in a.items():
            if k in ignore_list:
                continue
            if compare(a[k], b[k], trace=trace + ':' + str(k)):
                return 1
    elif isinstance(a, list) or isinstance(a, tuple):
        for n in range(len(a)):
            if compare(a[n], b[n], trace=trace + ':' + str(n)) and n not in ignore_list:
                return 1
    elif isinstance(a, torch.Tensor):
        if a.size() != b.size():
            print(trace)
            print('Tensor size mismatch {} {}'.format(a.size(), b.size()))
            return 1
        if torch.abs(F.l1_loss(a.float(), b.float())) > 1e-7:
            print(trace)
            print('Tensor mismatch {} {}'.format(a, b))
            return 1
    elif isinstance(a, np.ndarray):
        if not (a == b).all():
            print(trace)
            print(a, b)
            print(np.argwhere(a != b))
            return 1
    else:
        if not a == b:
            print(trace)
            print(a, b)
            return 1
    return 0


IGNORE_LIST = []


def recu_check_keys(ref, under_test, trace='ROOT'):
    # only testing shape and type
    for item in IGNORE_LIST:
        if item in trace:
            print('Skipped {}'.format(trace))
            return
    print('Checking {}'.format(trace))
    if under_test is None and ref is not None\
       or ref is None and under_test is not None:
        warnings.warn('Only one is None. REF={} DUT={} {}'.format(ref, under_test, trace))
    elif isinstance(under_test, torch.Tensor) or isinstance(ref, torch.Tensor):
        assert(isinstance(under_test, torch.Tensor) and isinstance(ref, torch.Tensor)),\
            'one is tensor and the other is not tensor or None {}'.format(trace)
        if under_test.size() != ref.size():
            warnings.warn('Mismatch size: REF={} DUT={} {}'.format(ref.size(), under_test.size(), trace))
    elif isinstance(under_test, list) or isinstance(under_test, tuple):
        if len(under_test) != len(ref):
            warnings.warn('Mismatch length: REF={} DUT={} {}'.format(len(ref), len(under_test), trace))
        for n in range(min(len(ref), len(under_test))):
            recu_check_keys(ref[n], under_test[n], trace=trace + ':' + str(n))
    elif isinstance(under_test, dict):
        assert isinstance(ref, dict)
        for k, v in ref.items():
            if k in under_test:
                recu_check_keys(v, under_test[k], trace=trace + ':' + str(k))
            else:
                warnings.warn('Missing key: {}'.format(trace + ':' + str(k)))


'''
def test_transformed_load_export(stat_processed):
    stat = Statistics(player_num=2)
    stat.load_from_transformed_stat(stat_processed, 0)
    assert stat.cached_transformed_stat[0] is not None
    assert stat.cached_transformed_stat[1] is None
    # the mmr should be set to 6200 for everything except SL
    stat_processed['mmr'] = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    assert compare(stat_processed, stat.get_transformed_stat(player=0)) == 0
    stat.cached_transformed_stat[0] = None
    dumped = stat.get_transformed_stat(player=0)
    assert compare(stat_processed, dumped) == 0
    print(stat.build_order_statistics)
    print(stat.get_z(0))
    fad = FakeActorDataset(trajectory_len=1)
    data = fad.get_1v1_agent_data()[0]
    ref = data['home']['behaviour_z']
    recu_check_keys(ref, stat.get_z(0))
'''
