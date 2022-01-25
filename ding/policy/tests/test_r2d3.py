import pytest
import torch
from easydict import EasyDict
from ding.policy.r2d3 import R2D3Policy
from ding.utils.data import offline_data_save_type
from torch.utils.tensorboard import SummaryWriter
import os

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
    unroll_len=20,
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
    ),
    collect=dict(
        each_iter_n_sample=32,
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

# create fake dataset
data = []
for i in range(100):
    d = {}
    d['obs'] = torch.zeros(obs_space)
    d['action'] = torch.Tensor([1.])
    data.append(d)



policy = R2D3Policy(cfg, enable_field=['collect', 'eval'])
policy._init_learn()