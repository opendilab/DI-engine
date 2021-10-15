import pytest
import threading
import time
import torch
import torch.nn as nn
from easydict import EasyDict

from ding.utils.data import offline_data_save_type, create_dataset

cfg1 = dict(policy=dict(collect=dict(
    data_type='naive',
    data_path='./expert.pkl',
), ))

cfg2 = dict(policy=dict(collect=dict(
    data_type='hdf5',
    data_path='./expert_demos.hdf5',
), ))

cfg3 = dict(env=dict(env_id='hopper-expert-v0'), policy=dict(collect=dict(data_type='d4rl', ), ))

cfgs = [cfg1, cfg2, cfg3]
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
def test_offline_data_save_type(data_type):
    offline_data_save_type(exp_data=fake_data, expert_data_path=expert_data_path, data_type=data_type)


@pytest.mark.parametrize('cfg', cfgs)
def test_dataset(cfg):
    cfg = EasyDict(cfg)
    create_dataset(cfg)
