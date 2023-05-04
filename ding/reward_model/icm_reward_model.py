from typing import Union, Tuple, List, Dict
from easydict import EasyDict

import random
import torch
import torch.nn as nn
import torch.optim as optim

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel
from .network import ICMNetwork
from .reword_model_utils import combine_intrinsic_exterinsic_reward


def collect_states(iterator: list) -> Tuple[list, list, list]:
    states = []
    next_states = []
    actions = []
    for item in iterator:
        state = item['obs']
        next_state = item['next_obs']
        action = item['action']
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
    return states, next_states, actions


@REWARD_MODEL_REGISTRY.register('icm')
class ICMRewardModel(BaseRewardModel):
    """
    Overview:
        The ICM reward model class (https://arxiv.org/pdf/1705.05363.pdf)
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_data``, \
            ``__init__``, ``_train``, ``load_state_dict``, ``state_dict``
    Config:
        == ====================  ========   =============  ====================================  =======================
        ID Symbol                Type       Default Value  Description                           Other(Shape)
        == ====================  ========   =============  ====================================  =======================
        1  ``type``              str         icm           | Reward model register name,         |
                                                           | refer to registry                   |
                                                           | ``REWARD_MODEL_REGISTRY``           |
        2  | ``intrinsic_``      str         add           | the intrinsic reward type           | including add, new
           | ``reward_type``                               |                                     | , or assign
        3  | ``learning_rate``   float       0.001         | The step size of gradient descent   |
        4  | ``obs_shape``       Tuple(      6             | the observation shape               |
                                 [int,
                                 list])
        5  | ``action_shape``    int         7             | the action space shape              |
        6  | ``batch_size``      int         64            | Training batch size                 |
        7  | ``hidden``          list        [64, 64,      | the MLP layer shape                 |
           | ``_size_list``      (int)       128]          |                                     |
        8  | ``update_per_``     int         100           | Number of updates per collect       |
           | ``collect``                                   |                                     |
        9  | ``reverse_scale``   float       1             | the importance weight of the        |
                                                           | forward and reverse loss            |
        10 | ``intrinsic_``      float       0.003         | the weight of intrinsic reward      | r = w*r_i + r_e
             ``reward_weight``
        11 | ``extrinsic_``      bool        True          | Whether to normlize
             ``reward_norm``                               | extrinsic reward
        12 | ``extrinsic_``      int         1             | the upper bound of the reward
            ``reward_norm_max``                            | normalization
        13 | ``clear_buffer``    int         1             | clear buffer per fixed iters        | make sure replay
             ``_per_iters``                                                                      | buffer's data count
                                                                                                 | isn't too few.
                                                                                                 | (code work in entry)
        == ====================  ========   =============  ====================================  =======================
    """
    config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='icm',
        # (str) The intrinsic reward type, including add, new, or assign.
        intrinsic_reward_type='add',
        # (float) The step size of gradient descent.
        learning_rate=1e-3,
        # (Tuple[int, list]), The observation shape.
        obs_shape=6,
        # (int) The action shape, support discrete action only in this version.
        action_shape=7,
        # (float) Batch size.
        batch_size=64,
        # (list) The MLP layer shape.
        hidden_size_list=[64, 64, 128],
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (float) The importance weight of the forward and reverse loss.
        reverse_scale=1,
        # (float) The weight of intrinsic reward.
        # r = intrinsic_reward_weight * r_i + r_e.
        intrinsic_reward_weight=0.003,  # 1/300
        # (bool) Whether to normlize extrinsic reward.
        # Normalize the reward to [0, extrinsic_reward_norm_max].
        extrinsic_reward_norm=True,
        # (int) The upper bound of the reward normalization.
        extrinsic_reward_norm_max=1,
        # (int) Clear buffer per fixed iters.
        clear_buffer_per_iters=100,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        super(ICMRewardModel, self).__init__()
        self.cfg = config
        assert device == "cpu" or device.startswith("cuda")
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = ICMNetwork(config.obs_shape, config.hidden_size_list, config.action_shape)
        self.reward_model.to(self.device)
        self.intrinsic_reward_type = config.intrinsic_reward_type
        assert self.intrinsic_reward_type in ['add', 'new', 'assign']
        self.train_data = []
        self.train_states = []
        self.train_next_states = []
        self.train_actions = []
        self.opt = optim.Adam(self.reward_model.parameters(), config.learning_rate)
        self.reverse_scale = config.reverse_scale
        self.estimate_cnt_icm = 0
        self.train_cnt_icm = 0

    def _train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        train_data_list = [i for i in range(0, len(self.train_states))]
        train_data_index = random.sample(train_data_list, self.cfg.batch_size)
        data_states: list = [self.train_states[i] for i in train_data_index]
        data_states: torch.Tensor = torch.stack(data_states).to(self.device)
        data_next_states: list = [self.train_next_states[i] for i in train_data_index]
        data_next_states: torch.Tensor = torch.stack(data_next_states).to(self.device)
        data_actions: list = [self.train_actions[i] for i in train_data_index]
        data_actions: torch.Tensor = torch.cat(data_actions).to(self.device)

        inverse_loss, forward_loss, accuracy = self.reward_model.learn(data_states, data_next_states, data_actions)
        loss = self.reverse_scale * inverse_loss + forward_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss, inverse_loss, forward_loss, accuracy

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            loss, inverse_loss, forward_loss, accuracy = self._train()
            self.tb_logger.add_scalar('icm_reward/total_loss', loss, self.train_cnt_icm)
            self.tb_logger.add_scalar('icm_reward/forward_loss', forward_loss, self.train_cnt_icm)
            self.tb_logger.add_scalar('icm_reward/inverse_loss', inverse_loss, self.train_cnt_icm)
            self.tb_logger.add_scalar('icm_reward/action_accuracy', accuracy, self.train_cnt_icm)
            self.train_cnt_icm += 1

    def estimate(self, data: list) -> List[Dict]:
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)
        states, next_states, actions = collect_states(train_data_augmented)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        with torch.no_grad():
            raw_icm_reward = self.reward_model.forward(states, next_states, actions)
            self.estimate_cnt_icm += 1
            self.tb_logger.add_scalar('icm_reward/raw_icm_reward_max', raw_icm_reward.max(), self.estimate_cnt_icm)
            self.tb_logger.add_scalar('icm_reward/raw_icm_reward_mean', raw_icm_reward.mean(), self.estimate_cnt_icm)
            self.tb_logger.add_scalar('icm_reward/raw_icm_reward_min', raw_icm_reward.min(), self.estimate_cnt_icm)
            self.tb_logger.add_scalar('icm_reward/raw_icm_reward_std', raw_icm_reward.std(), self.estimate_cnt_icm)
            icm_reward = (raw_icm_reward - raw_icm_reward.min()) / (raw_icm_reward.max() - raw_icm_reward.min() + 1e-8)
            icm_reward = icm_reward.to(self.device)
        train_data_augmented = combine_intrinsic_exterinsic_reward(train_data_augmented, icm_reward, self.cfg)

        return train_data_augmented

    def collect_data(self, data: list) -> None:
        self.train_data.extend(collect_states(data))
        states, next_states, actions = collect_states(data)
        self.train_states.extend(states)
        self.train_next_states.extend(next_states)
        self.train_actions.extend(actions)

    def clear_data(self, iter: int) -> None:
        assert hasattr(
            self.cfg, 'clear_buffer_per_iters'
        ), "Reward Model does not have clear_buffer_per_iters, Clear failed"
        if iter % self.cfg.clear_buffer_per_iters == 0:
            self.train_data.clear()
            self.train_states.clear()
            self.train_next_states.clear()
            self.train_actions.clear()

    def state_dict(self) -> Dict:
        return self.reward_model.state_dict()

    def load_state_dict(self, _state_dict: Dict) -> None:
        self.reward_model.load_state_dict(_state_dict)
