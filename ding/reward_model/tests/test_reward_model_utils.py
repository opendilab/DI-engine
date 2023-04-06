import pytest
import torch

from ding.reward_model.reword_model_utils import concat_state_action_pairs


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
