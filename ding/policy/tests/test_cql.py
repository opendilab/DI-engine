import pytest
import torch
from easydict import EasyDict
from ding.policy.cql import CQLPolicy, CQLDiscretePolicy
from ding.utils.data import offline_data_save_type
from torch.utils.tensorboard import SummaryWriter
from ding.model.wrapper.model_wrappers import ArgmaxSampleWrapper, HiddenStateWrapper, EpsGreedySampleWrapper
import os
from typing import List
from collections import namedtuple

obs_space = 5
action_space = 3

cfg = dict(
    on_policy = False,
    cuda=True,
    model=dict(
        obs_shape=obs_space,
        action_shape=action_space,
        twin_critic=True,
        action_space='reparameterization',
        actor_head_hidden_size=256,
        critic_head_hidden_size=256,
    ),
    learn=dict(
        data_path=None,
        train_epoch=30000,
        batch_size=256,
        learning_rate_q=3e-4,
        learning_rate_policy=1e-4,
        learning_rate_alpha=1e-4,
        ignore_done=False,
        target_theta=0.005,
        discount_factor=0.99,
        alpha=0.2,
        reparameterization=True,
        auto_alpha=False,
        lagrange_thresh=-1.0,
        min_q_weight=5.0,
    ),
    collect=dict(
        unroll_len=1,
        data_type='d4rl',
    ),
    command=dict(),
    eval=dict(evaluator=dict(eval_freq=500, )),
    other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
)
cfg = EasyDict(cfg)


def get_batch(size=8):
    data = {}
    for i in range(size):
        obs = torch.zeros(obs_space)
        data[i] = obs
    return data


def get_transition(size=20):
    data = []
    import numpy as np
    for i in range(size):
        sample = {}
        sample['obs'] = torch.zeros(obs_space)
        sample['action'] = torch.tensor(np.array([int(i % action_space)]))
        sample['done'] = False
        sample['prev_state'] = [torch.randn(1, 1, 512) for __ in range(2)]
        sample['reward'] = torch.Tensor([1.])
        sample['IS'] = 1.
        sample['is_expert'] = bool(i % 2)
        data.append(sample)
    return data


@pytest.mark.parametrize('cfg', [cfg])
@pytest.mark.unittest
def test_cql_continuous(cfg):
    policy = CQLPolicy(cfg, enable_field=['collect', 'eval'])
    '''policy._init_learn()
    assert type(policy._learn_model) == ArgmaxSampleWrapper
    assert type(policy._target_model) == HiddenStateWrapper
    policy._reset_learn()
    policy._reset_learn([0])
    state = policy._state_dict_learn()
    policy._load_state_dict_learn(state)
    policy._init_collect()
    assert type(policy._collect_model) == EpsGreedySampleWrapper
    policy._reset_collect()
    policy._reset_collect([0])
    policy._init_eval()
    assert type(policy._eval_model) == ArgmaxSampleWrapper
    policy._reset_eval()
    policy._reset_eval([0])
    assert policy.default_model()[0] == 'drqn'
    var = policy._monitor_vars_learn()
    assert type(var) == list
    assert sum([type(s) == str for s in var]) == len(var)
    batch = get_batch(8)
    out = policy._forward_collect(batch, eps=0.1)
    assert len(set(out[0].keys()).intersection({'logit', 'prev_state', 'action'})) == 3
    assert list(out[0]['logit'].shape) == [action_space]
    timestep = namedtuple('timestep', ['reward', 'done'])
    ts = timestep(1., 0.,)
    ts = policy._process_transition(batch[0], out[0], ts)
    assert len(set(ts.keys()).intersection({'prev_state', 'action', 'reward', 'done', 'obs'})) == 5
    ts = get_transition(64 * policy._unroll_len_add_burnin_step)
    sample = policy._get_train_sample(ts)
    n_traj = len(ts) // policy._unroll_len_add_burnin_step
    assert len(sample) == n_traj + 1 if len(ts) % policy._unroll_len_add_burnin_step != 0 else n_traj
    out = policy._forward_eval(batch)
    assert len(set(out[0].keys()).intersection({'logit', 'action'})) == 2
    assert list(out[0]['logit'].shape) == [action_space]
    for i in range(len(sample)):
        sample[i]['IS'] = sample[i]['IS'][cfg.burnin_step:]
    out = policy._forward_learn(sample)
    policy._value_rescale = False
    out = policy._forward_learn(sample)'''
