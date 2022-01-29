import copy

import pytest
import torch
from easydict import EasyDict
from ding.policy.cql import CQLPolicy, CQLDiscretePolicy
from ding.utils.data import offline_data_save_type
from tensorboardX import SummaryWriter
from ding.model.wrapper.model_wrappers import ArgmaxSampleWrapper, EpsGreedySampleWrapper, TargetNetworkWrapper
import os
from typing import List
from collections import namedtuple
from ding.utils import deep_merge_dicts

obs_space = 5
action_space = 3

cfg1 = EasyDict(CQLPolicy.config)
cfg1.model.obs_shape = obs_space
cfg1.model.action_shape = action_space

cfg2 = copy.deepcopy(cfg1)
cfg2.learn.auto_alpha = False
cfg2.learn.log_space = False

cfg3 = EasyDict(CQLDiscretePolicy.config)
cfg3.model = {}
cfg3.model.obs_shape = obs_space
cfg3.model.action_shape = action_space

cfg4 = copy.deepcopy(cfg3)
cfg4.learn.auto_alpha = False


def get_batch(size=8):
    data = {}
    for i in range(size):
        obs = torch.zeros(obs_space)
        data[i] = obs
    return data


def get_transition(size=20):
    data = []
    for i in range(size):
        sample = {}
        sample['obs'] = torch.zeros(obs_space)
        sample['action'] = torch.zeros(action_space)
        sample['done'] = False
        sample['next_obs'] = torch.zeros(obs_space)
        sample['reward'] = torch.Tensor([1.])
        data.append(sample)
    return data


def get_transition_batch(bs=1):
    sample = {}
    sample['obs'] = torch.zeros(bs, obs_space)
    sample['action'] = torch.zeros(bs, action_space)
    return sample


@pytest.mark.parametrize('cfg', [cfg1, cfg2])
@pytest.mark.unittest
def test_cql_continuous(cfg):
    policy = CQLPolicy(cfg, enable_field=['collect', 'eval', 'learn'])
    assert type(policy._target_model) == TargetNetworkWrapper
    q_value = policy._get_q_value(get_transition_batch(cfg.learn.num_actions))
    assert q_value[0].shape[-1] == 1 and q_value[0].shape[-2] == cfg.learn.num_actions
    act, log_prob = policy._get_policy_actions(get_transition_batch(cfg.learn.num_actions))
    assert list(act.shape) == [cfg.learn.num_actions * 10, action_space]
    sample = get_transition(size=20)
    out = policy._forward_learn(sample)


def get_transition_discrete(size=20):
    data = []
    for i in range(size):
        sample = {}
        sample['obs'] = torch.zeros(obs_space)
        sample['action'] = torch.tensor(i % action_space)
        sample['done'] = False
        sample['next_obs'] = torch.zeros(obs_space)
        sample['reward'] = torch.Tensor([1.])
        data.append(sample)
    return data


@pytest.mark.parametrize('cfg', [cfg3, cfg4])
@pytest.mark.unittest
def test_cql_discrete(cfg):
    policy = CQLDiscretePolicy(cfg, enable_field=['collect', 'eval', 'learn'])
    assert type(policy._learn_model) == ArgmaxSampleWrapper
    assert type(policy._target_model) == TargetNetworkWrapper
    assert type(policy._collect_model) == EpsGreedySampleWrapper
    sample = get_transition_batch(bs=20)
    samples = policy._get_train_sample(sample)
    assert len(samples['obs']) == 20
    state = policy._state_dict_learn()
    policy._load_state_dict_learn(state)
    sample = get_transition_discrete(size=1)
    out = policy._forward_learn(sample)
    out = policy._forward_collect(get_batch(size=8), eps=0.1)
