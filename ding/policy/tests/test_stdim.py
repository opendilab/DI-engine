from copy import deepcopy
import pytest
import torch
from easydict import EasyDict
from ding.model.wrapper.model_wrappers import BaseModelWrapper, MultinomialSampleWrapper
from ding.policy import PPOSTDIMPolicy

obs_shape = 4
action_shape = 2

cfg1 = PPOSTDIMPolicy.config
cfg1["model"] = {}
cfg1 = EasyDict(PPOSTDIMPolicy.config)
cfg1.model.obs_shape = obs_shape
cfg1.model.action_shape = action_shape

cfg2 = deepcopy(cfg1)
cfg2.action_space = "continuous"


def get_transition_discrete(size=64):
    data = []
    for i in range(size):
        sample = {}
        sample['obs'] = torch.rand(obs_shape)
        sample['next_obs'] = torch.rand(obs_shape)
        sample['action'] = torch.tensor([0], dtype=torch.long)
        sample['value'] = torch.rand(1)
        sample['logit'] = torch.rand(size=(action_shape, ))
        sample['done'] = False
        sample['reward'] = torch.rand(1)
        data.append(sample)
    return data


@pytest.mark.parametrize('cfg', [cfg1])
@pytest.mark.unittest
def test_stdim(cfg):
    policy = PPOSTDIMPolicy(cfg, enable_field=['collect', 'eval', 'learn'])
    assert type(policy._learn_model) == BaseModelWrapper
    assert type(policy._collect_model) == MultinomialSampleWrapper
    sample = get_transition_discrete(size=64)
    state = policy._state_dict_learn()
    policy._load_state_dict_learn(state)
    sample = get_transition_discrete(size=64)
    out = policy._forward_learn(sample)
