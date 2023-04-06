from typing import Union, Optional, List, Any, Tuple
from collections.abc import Iterable

import torch
import torch.optim as optim
import torch.nn.functional as F

from ding.torch_utils import one_hot


def concat_state_action_pairs(
        data: list, action_size: Optional[int] = None, one_hot_: Optional[bool] = False
) -> torch.Tensor:
    """
    Overview:
        Concatenate state and action pairs from input.
    Arguments:
        - data (:obj:`List`): List with at least ``obs`` and ``action`` keys.
    Returns:
        - state_actions_tensor (:obj:`Torch.tensor`): State and action pairs.
    """
    states_data = []
    actions_data = []
    #check data(dict) has key obs and action
    assert isinstance(data, Iterable), "data should be Iterable"
    assert "obs" in data[0] and "action" in data[0], "data member must contain key 'obs' and 'action' "
    for item in data:
        states_data.append(item['obs'].flatten())
        if one_hot_ and action_size:
            new_action = torch.Tensor([int(i == item['action']) for i in range(action_size)])
            print(new_action.shape)
            action = one_hot(torch.Tensor(item['action']), action_size).squeeze(dim=0)
            print(action.shape)
            actions_data.append(action)
        else:
            actions_data.append(item['action'])

    states_tensor: torch.Tensor = torch.stack(states_data).float()
    actions_tensor: torch.Tensor = torch.stack(actions_data).float()
    states_actions_tensor: torch.Tensor = torch.cat([states_tensor, actions_tensor], dim=1)

    return states_actions_tensor
