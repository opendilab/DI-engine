import pytest
import torch
from easydict import EasyDict
from ding.reward_model.gail_irl_model import GailRewardModel
from ding.utils.data import offline_data_save_type
from tensorboardX import SummaryWriter
import os

obs_space_1d, obs_space_3d = 4, [4, 84, 84]
expert_data_path_1d, expert_data_path_3d = './expert_data_1d', './expert_data_3d'
if not os.path.exists('./expert_data_1d'):
    try:
        os.mkdir('./expert_data_1d')
    except FileExistsError:
        pass
if not os.path.exists('./expert_data_3d'):
    try:
        os.mkdir('./expert_data_3d')
    except FileExistsError:
        pass
device = 'cpu'
action_space = 3

cfg1 = dict(
    input_size=obs_space_1d + 1,
    hidden_size=64,
    batch_size=5,
    learning_rate=1e-3,
    update_per_collect=2,
    data_path=expert_data_path_1d,
),

cfg2 = dict(
    input_size=obs_space_3d,
    hidden_size=64,
    batch_size=5,
    learning_rate=1e-3,
    update_per_collect=2,
    data_path=expert_data_path_3d,
    action_size=action_space,
),

# create fake expert dataset
data_1d = []
for i in range(20):
    d = {}
    d['obs'] = torch.zeros(obs_space_1d)
    d['action'] = torch.Tensor([1.])
    data_1d.append(d)

data_3d = []
for i in range(20):
    d = {}
    d['obs'] = torch.zeros(obs_space_3d)
    d['action'] = torch.Tensor([1.])
    data_3d.append(d)


@pytest.mark.parametrize('cfg', cfg1)
@pytest.mark.unittest
def test_dataset_1d(cfg):
    offline_data_save_type(
        exp_data=data_1d, expert_data_path=expert_data_path_1d + '/expert_data.pkl', data_type='naive'
    )
    data = data_1d
    cfg = EasyDict(cfg)
    policy = GailRewardModel(cfg, device, tb_logger=SummaryWriter())
    policy.load_expert_data()
    assert len(policy.expert_data) == 20
    state = policy.state_dict()
    policy.load_state_dict(state)
    policy.collect_data(data)
    assert len(policy.train_data) == 20
    for _ in range(5):
        policy.train()
    train_data_augmented = policy.estimate(data)
    assert 'reward' in train_data_augmented[0].keys()
    policy.clear_data()
    assert len(policy.train_data) == 0
    os.popen('rm -rf {}'.format(expert_data_path_1d))


@pytest.mark.parametrize('cfg', cfg2)
@pytest.mark.unittest
def test_dataset_3d(cfg):
    offline_data_save_type(
        exp_data=data_3d, expert_data_path=expert_data_path_3d + '/expert_data.pkl', data_type='naive'
    )
    data = data_3d
    cfg = EasyDict(cfg)
    policy = GailRewardModel(cfg, device, tb_logger=SummaryWriter())
    policy.load_expert_data()
    assert len(policy.expert_data) == 20
    state = policy.state_dict()
    policy.load_state_dict(state)
    policy.collect_data(data)
    assert len(policy.train_data) == 20
    for _ in range(5):
        policy.train()
    train_data_augmented = policy.estimate(data)
    assert 'reward' in train_data_augmented[0].keys()
    policy.clear_data()
    assert len(policy.train_data) == 0
    os.popen('rm -rf {}'.format(expert_data_path_3d))
