import pytest
import numpy as np
import random
import torch
from ding.data.level_replay.level_sampler import LevelSampler


@pytest.mark.unittest
def test_level_sampler():
    num_seeds = 500
    obs_shape = [3, 64, 64]
    action_shape = 15
    collector_env_num = 16
    level_replay_dict = dict(
        strategy='min_margin',
        score_transform='rank',
        temperature=0.1,
    )
    N = 10
    collector_sample_length = 160

    train_seeds = [i for i in range(num_seeds)]
    level_sampler = LevelSampler(train_seeds, obs_shape, action_shape, collector_env_num, level_replay_dict)

    value = torch.randn(collector_sample_length)
    reward = torch.randn(collector_sample_length)
    adv = torch.randn(collector_sample_length)
    done = torch.randn(collector_sample_length)
    logit = torch.randn(collector_sample_length, N)
    seeds = [random.randint(0, num_seeds) for i in range(collector_env_num)]
    all_seeds = torch.Tensor(
        [seeds[i] for i in range(collector_env_num) for j in range(int(collector_sample_length / collector_env_num))]
    )

    train_data = {'value': value, 'reward': reward, 'adv': adv, 'done': done, 'logit': logit, 'seed': all_seeds}
    level_sampler.update_with_rollouts(train_data, collector_env_num)
    sample_seed = level_sampler.sample()
    assert isinstance(sample_seed, int)
