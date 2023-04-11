from typing import Union, Optional, List, Any, Tuple
from collections.abc import Iterable
from easydict import EasyDict

import torch
import torch.optim as optim
import torch.nn.functional as F

from ding.torch_utils import one_hot
from ding.utils import RunningMeanStd
from ding.torch_utils.data_helper import to_tensor


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
            action = one_hot(torch.Tensor(item['action']), action_size).squeeze(dim=0)
            actions_data.append(action)
        else:
            actions_data.append(item['action'])

    states_tensor: torch.Tensor = torch.stack(states_data).float()
    actions_tensor: torch.Tensor = torch.stack(actions_data).float()
    states_actions_tensor: torch.Tensor = torch.cat([states_tensor, actions_tensor], dim=1)

    return states_actions_tensor


def combine_intrinsic_exterinsic_reward(
        train_data_augmented: Any, rnd_reward: List[torch.Tensor], config: EasyDict
) -> Any:
    for item, rnd_rew in zip(train_data_augmented, rnd_reward):
        if config.intrinsic_reward_type == 'add':
            if config.extrinsic_reward_norm:
                item['reward'
                     ] = item['reward'] / config.extrinsic_reward_norm_max + rnd_rew * config.intrinsic_reward_weight
            else:
                item['reward'] = item['reward'] + rnd_rew * config.intrinsic_reward_weight
        elif config.intrinsic_reward_type == 'new':
            item['intrinsic_reward'] = rnd_rew
            if config.extrinsic_reward_norm:
                item['reward'] = item['reward'] / config.extrinsic_reward_norm_max
        elif config.intrinsic_reward_type == 'assign':
            item['reward'] = rnd_rew

    return item


def collect_states(iterator) -> List:
    res = []
    for item in iterator:
        state = item['obs']
        res.append(state)
    return res


def obs_norm(
        train_data: torch.Tensor, running_mean_std_rnd_obs: RunningMeanStd, config: EasyDict, device: str
) -> torch.Tensor:
    # Note: observation normalization: transform obs to mean 0, std 1, move norm obs to specific device
    running_mean_std_rnd_obs.update(train_data.cpu().numpy())
    train_data = (train_data - to_tensor(running_mean_std_rnd_obs.mean
                                         ).to(device)) / to_tensor(running_mean_std_rnd_obs.std).to(device)
    train_data = torch.clamp(train_data, min=config.obs_norm_clamp_min, max=config.obs_norm_clamp_max)

    return train_data
