import pytest
import torch
from easydict import EasyDict
from ding.policy.r2d3 import R2D3Policy
from ding.utils.data import offline_data_save_type
from tensorboardX import SummaryWriter
from ding.model.wrapper.model_wrappers import ArgmaxSampleWrapper, HiddenStateWrapper, EpsGreedySampleWrapper
import os
from typing import List
from collections import namedtuple

obs_space = 5
action_space = 4

cfg = dict(
    cuda=True,
    on_policy=False,
    priority=True,
    priority_IS_weight=True,
    model=dict(
        obs_shape=obs_space,
        action_shape=action_space,
        encoder_hidden_size_list=[128, 128, 512],
    ),
    discount_factor=0.99,
    burnin_step=2,
    nstep=5,
    learn_unroll_len=20,
    burning_step=5,
    learn=dict(
        value_rescale=True,
        update_per_collect=8,
        batch_size=64,
        learning_rate=0.0005,
        target_update_theta=0.001,
        lambda1=1.0,  # n-step return
        lambda2=1.0,  # supervised loss
        lambda3=1e-5,  # L2  it's very important to set Adam optimizer optim_type='adamw'.
        lambda_one_step_td=1,  # 1-step return
        margin_function=0.8,  # margin function in JE, here we implement this as a constant
        per_train_iter_k=0,
        ignore_done=False,
    ),
    collect=dict(
        n_sample=32,
        traj_len_inf=True,
        env_num=8,
        pho=1 / 4,
    ),
    eval=dict(env_num=8, ),
    other=dict(
        eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=100000,
        ),
        replay_buffer=dict(
            replay_buffer_size=int(1e4),
            alpha=0.6,
            beta=0.4,
        ),
    ),
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
def test_r2d3(cfg):
    policy = R2D3Policy(cfg, enable_field=['collect', 'eval'])
    policy._init_learn()
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
    ts = timestep(
        1.,
        0.,
    )
    ts = policy._process_transition(batch[0], out[0], ts)
    assert len(set(ts.keys()).intersection({'prev_state', 'action', 'reward', 'done', 'obs'})) == 5
    ts = get_transition(64 * policy._sequence_len)
    sample = policy._get_train_sample(ts)
    n_traj = len(ts) // policy._sequence_len
    assert len(sample) == n_traj + 1 if len(ts) % policy._sequence_len != 0 else n_traj
    out = policy._forward_eval(batch)
    assert len(set(out[0].keys()).intersection({'logit', 'action'})) == 2
    assert list(out[0]['logit'].shape) == [action_space]
    for i in range(len(sample)):
        sample[i]['IS'] = sample[i]['IS'][cfg.burnin_step:]
    out = policy._forward_learn(sample)
    policy._value_rescale = False
    out = policy._forward_learn(sample)
