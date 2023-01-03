import pytest
import torch
from easydict import EasyDict
from ding.model.wrapper.model_wrappers import ArgmaxSampleWrapper, EpsGreedySampleWrapper, TargetNetworkWrapper
from ding.policy.bdq import BDQPolicy
from dizoo.classic_control.pendulum.envs import PendulumEnv

obs_space = 3
num_branches = 1
action_bins_per_branch = 5

cfg1 = EasyDict(BDQPolicy.config)
cfg1.model = {}
cfg1.model.obs_shape = obs_space
cfg1.model.num_branches = num_branches
cfg1.model.action_bins_per_branch = action_bins_per_branch


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
        sample['action'] = torch.randint(0, action_bins_per_branch, (num_branches, ))
        sample['done'] = False
        sample['next_obs'] = torch.zeros(obs_space)
        sample['reward'] = torch.Tensor([1.])
        data.append(sample)
    return data


@pytest.mark.parametrize('cfg', [cfg1])
@pytest.mark.unittest
def test_bdq(cfg):
    policy = BDQPolicy(cfg, enable_field=['collect', 'eval', 'learn'])
    assert type(policy._learn_model) == ArgmaxSampleWrapper
    assert type(policy._target_model) == TargetNetworkWrapper
    assert type(policy._collect_model) == EpsGreedySampleWrapper
    batch_obs = get_batch()
    policy._forward_eval(batch_obs)
    policy._forward_collect(batch_obs, 0.5)

    sample = get_transition(size=20)
    policy._forward_learn(sample)
    policy._get_train_sample(sample)

    env = PendulumEnv(EasyDict({'act_scale': True, 'continuous': False}))
    env.seed(314)
    obs = env.reset()
    b_obs = {0: obs}
    raw_out = policy._forward_collect(b_obs, 0.5)[0]
    timestep = env.step(raw_out['action'].numpy())
    transition = policy._process_transition(obs, raw_out, timestep)
