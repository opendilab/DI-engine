import pytest
import torch

from easydict import EasyDict
from ding.reward_model.reword_model_utils import concat_state_action_pairs, combine_intrinsic_exterinsic_reward


@pytest.mark.unittest
def test_concat_state_action_pairs():
    data = [{'obs': torch.rand(3), 'action': torch.randint(0, 4, size=(1, ))} for i in range(10)]
    states_actions_tensor = concat_state_action_pairs(data)
    states_actions_test = []
    for item in data:
        state = item['obs'].flatten()
        action = item['action']
        s_a = torch.cat([state, action.float()], dim=-1)
        states_actions_test.append(s_a)
    states_actions_tensor_test = torch.stack(states_actions_test)
    assert states_actions_tensor.equal(states_actions_tensor_test)


@pytest.mark.unittest
def test_concat_state_action_pairs_one_hot():
    data = [{'obs': torch.rand(3), 'action': torch.randint(0, 4, size=(1, ))} for i in range(10)]
    action_size = 5
    states_actions_tensor = concat_state_action_pairs(data, action_size, True)
    states_actions_test = []
    for item in data:
        state = item['obs'].flatten()
        action = item['action']
        action = torch.Tensor([int(i == action) for i in range(action_size)])
        s_a = torch.cat([state, action], dim=-1)
        states_actions_test.append(s_a)
    states_actions_tensor_test = torch.stack(states_actions_test)
    assert states_actions_tensor.equal(states_actions_tensor_test)


@pytest.mark.unittest
def test_combine_intrinsic_exterinsic_reward():
    intrinsic_reward = torch.rand(1)
    train_data_augument = [{'obs': torch.rand(5), 'reward': torch.rand(1), 'intrinsic_reward': torch.rand(1)}]
    config_list = [
        {
            'intrinsic_reward_type': 'new',
            'intrinsic_reward_weight': 1,
            'extrinsic_reward_norm_max': 1,
            'extrinsic_reward_norm': True,
        }, {
            'intrinsic_reward_type': 'assign',
            'intrinsic_reward_weight': 1,
            'extrinsic_reward_norm_max': 1,
            'extrinsic_reward_norm': True,
        }
    ]
    for config in config_list:
        config = EasyDict(config)
        item = combine_intrinsic_exterinsic_reward(train_data_augument, intrinsic_reward, config)
        if config.intrinsic_reward_type == 'new':
            assert item['intrinsic_reward'] == intrinsic_reward
        elif config.intrinsic_reward_type == 'assign':
            assert item['reward'] == intrinsic_reward
