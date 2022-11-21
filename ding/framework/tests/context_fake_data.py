from ding.framework import Context, OnlineRLContext, OfflineRLContext
import random
import numpy as np
import treetensor.torch as ttorch
import torch

batch_size = 64
n_sample = 8
action_dim = 1
obs_dim = 4
logit_dim = 2

n_episodes = 2
n_episode_length = 16
update_per_collect = 4
collector_env_num = 8


# the range here is meaningless and just for test
def fake_train_data():
    train_data = ttorch.as_tensor(
        {
            'action': torch.randint(0, 2, size=(action_dim, )),
            'collect_train_iter': torch.randint(0, 100, size=(1, )),
            'done': torch.tensor(False),
            'env_data_id': torch.tensor([2]),
            'next_obs': torch.randn(obs_dim),
            'obs': torch.randn(obs_dim),
            'reward': torch.randint(0, 2, size=(1, )),
        }
    )
    return train_data


def fake_online_rl_context():
    ctx = OnlineRLContext(
        env_step=random.randint(0, 100),
        env_episode=random.randint(0, 100),
        train_iter=random.randint(0, 100),
        train_data=[fake_train_data() for _ in range(batch_size)],
        train_output=[{
            'cur_lr': 0.001,
            'total_loss': random.uniform(0, 2)
        } for _ in range(update_per_collect)],
        obs=torch.randn(collector_env_num, obs_dim),
        action=[np.random.randint(low=0, high=1, size=(action_dim), dtype=np.int64) for _ in range(collector_env_num)],
        inference_output={
            env_id: {
                'logit': torch.randn(logit_dim),
                'action': torch.randint(0, 2, size=(action_dim, ))
            }
            for env_id in range(collector_env_num)
        },
        collect_kwargs={'eps': random.uniform(0, 1)},
        trajectories=[fake_train_data() for _ in range(n_sample)],
        episodes=[[fake_train_data() for _ in range(n_episode_length)] for _ in range(n_episodes)],
        trajectory_end_idx=[i for i in range(n_sample)],
        eval_value=random.uniform(-1.0, 1.0),
        last_eval_iter=random.randint(0, 100),
    )
    return ctx


def fake_offline_rl_context():
    ctx = OfflineRLContext(
        train_epoch=random.randint(0, 100),
        train_iter=random.randint(0, 100),
        train_data=[fake_train_data() for _ in range(batch_size)],
        train_output=[{
            'cur_lr': 0.001,
            'total_loss': random.uniform(0, 2)
        } for _ in range(update_per_collect)],
        eval_value=random.uniform(-1.0, 1.0),
        last_eval_iter=random.randint(0, 100),
    )
    return ctx
