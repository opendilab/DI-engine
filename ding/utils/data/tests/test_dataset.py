import pytest
import torch
from easydict import EasyDict
import os
from ding.utils.data import offline_data_save_type, create_dataset, NaiveRLDataset, D4RLDataset, HDF5Dataset

cfg1 = dict(policy=dict(collect=dict(
    data_type='naive',
    data_path='./expert.pkl',
), ))

cfg2 = dict(policy=dict(collect=dict(
    data_type='hdf5',
    data_path='./expert_demos.hdf5',
    normalize_states=True,
), ))

cfg3 = dict(env=dict(env_id='hopper-expert-v0'), policy=dict(collect=dict(data_type='d4rl', ), ))

cfgs = [cfg1, cfg2]  # cfg3
unittest_args = ['naive', 'hdf5']

# fake transition & data
transition = {}
transition['obs'] = torch.zeros((3, 1))
transition['next_obs'] = torch.zeros((3, 1))
transition['action'] = torch.zeros((1, 1))
transition['reward'] = torch.tensor((1, ))
transition['done'] = False
transition['collect_iter'] = 0

fake_data = [transition for i in range(32)]
expert_data_path = './expert.pkl'


@pytest.mark.parametrize('data_type', unittest_args)
@pytest.mark.unittest
def test_offline_data_save_type(data_type):
    offline_data_save_type(exp_data=fake_data, expert_data_path=expert_data_path, data_type=data_type)


@pytest.mark.parametrize('cfg', cfgs)
@pytest.mark.unittest
def test_dataset(cfg):
    cfg = EasyDict(cfg)
    create_dataset(cfg)


@pytest.mark.parametrize('cfg', [cfg1])
@pytest.mark.unittest
def test_NaiveRLDataset(cfg):
    cfg = EasyDict(cfg)
    NaiveRLDataset(cfg)
    dataset = NaiveRLDataset(expert_data_path)
    assert type(len(dataset)) == int
    assert dataset[0] is not None


# @pytest.mark.parametrize('cfg', [cfg3])
# @pytest.mark.unittest
# def test_D4RLDataset(cfg):
#     cfg = EasyDict(cfg)
#     dataset = D4RLDataset(cfg)


@pytest.mark.parametrize('cfg', [cfg2])
@pytest.mark.unittest
def test_HDF5Dataset(cfg):
    cfg = EasyDict(cfg)
    dataset = HDF5Dataset(cfg)
    assert dataset.mean is not None and dataset.std[0] is not None
    assert dataset._data['obs'].mean(0)[0] == 0
    assert type(len(dataset)) == int
    assert dataset[0] is not None


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):

    def remove_test_dir():
        if os.path.exists('./expert.pkl'):
            os.remove('./expert.pkl')
        if os.path.exists('./expert_demos.hdf5'):
            os.remove('./expert_demos.hdf5')

    request.addfinalizer(remove_test_dir)
