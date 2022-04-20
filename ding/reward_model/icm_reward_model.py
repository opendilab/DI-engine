from typing import Union, Tuple, List, Dict
from easydict import EasyDict

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ding.utils import SequenceType, REWARD_MODEL_REGISTRY
from ding.model import FCEncoder, ConvEncoder
from ding.torch_utils import one_hot
from .base_reward_model import BaseRewardModel


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


class ICMNetwork(nn.Module):
    r"""
    Intrinsic Curiosity Model (ICM Module)
    Implementation of:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf
    [2] Code implementation reference:
    https://github.com/pathak22/noreward-rl
    https://github.com/jcwleo/curiosity-driven-exploration-pytorch

    1) Embedding observations into a latent space
    2) Predicting the action logit given two consecutive embedded observations
    3) Predicting the next embedded obs, given the embeded former observation and action
    """

    def __init__(self, obs_shape: Union[int, SequenceType], hidden_size_list: SequenceType, action_shape: int) -> None:
        super(ICMNetwork, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.feature = FCEncoder(obs_shape, hidden_size_list)
        elif len(obs_shape) == 3:
            self.feature = ConvEncoder(obs_shape, hidden_size_list)
        else:
            raise KeyError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own ICM model".
                format(obs_shape)
            )
        self.action_shape = action_shape
        feature_output = hidden_size_list[-1]
        self.inverse_net = nn.Sequential(nn.Linear(feature_output * 2, 512), nn.ReLU(), nn.Linear(512, action_shape))
        self.residual = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(action_shape + 512, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 512),
                ) for _ in range(8)
            ]
        )
        self.forward_net_1 = nn.Sequential(nn.Linear(action_shape + feature_output, 512), nn.LeakyReLU())
        self.forward_net_2 = nn.Linear(action_shape + 512, feature_output)

    def forward(self, state: torch.Tensor, next_state: torch.Tensor,
                action_long: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Overview:
            Use observation, next_observation and action to genearte ICM module
            Parameter updates with ICMNetwork forward setup.
        Arguments:
            - state (:obj:`torch.Tensor`):
                The current state batch
            - next_state (:obj:`torch.Tensor`):
                The next state batch
            - action_long (:obj:`torch.Tensor`):
                The action batch
        Returns:
            - real_next_state_feature (:obj:`torch.Tensor`):
                Run with the encoder. Return the real next_state's embedded feature.
            - pred_next_state_feature (:obj:`torch.Tensor`):
                Run with the encoder and residual network. Return the predicted next_state's embedded feature.
            - pred_action_logit (:obj:`torch.Tensor`):
                Run with the encoder. Return the predicted action logit.
        Shapes:
            - state (:obj:`torch.Tensor`): :math:`(B, N)`, where B is the batch size and N is ''obs_shape''
            - next_state (:obj:`torch.Tensor`): :math:`(B, N)`, where B is the batch size and N is ''obs_shape''
            - action_long (:obj:`torch.Tensor`): :math:`(B)`, where B is the batch size''
            - real_next_state_feature (:obj:`torch.Tensor`): :math:`(B, M)`, where B is the batch size
              and M is embedded feature size
            - pred_next_state_feature (:obj:`torch.Tensor`): :math:`(B, M)`, where B is the batch size
              and M is embedded feature size
            - pred_action_logit (:obj:`torch.Tensor`): :math:`(B, A)`, where B is the batch size
              and A is the ''action_shape''
        """
        action = one_hot(action_long, num=self.action_shape)
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action logit
        concat_state = torch.cat((encode_state, encode_next_state), 1)
        pred_action_logit = self.inverse_net(concat_state)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)
            ) + pred_next_state_feature_orig
        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))
        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action_logit


@REWARD_MODEL_REGISTRY.register('icm')
class ICMRewardModel(BaseRewardModel):
    """
    Overview:
        The ICM reward model class (https://arxiv.org/pdf/1705.05363.pdf)
    Interface:
        ``estimate``, ``train``, ``collect_data``, ``clear_data``, \
            ``__init__``, ``_train``,
    """
    config = dict(
        # (str) the type of the exploration method
        type='icm',
        # (str) the intrinsic reward type, including add, new, or assign
        intrinsic_reward_type='add',
        # (float) learning rate of the optimizer
        learning_rate=1e-3,
        # (Tuple[int, list]), the observation shape,
        obs_shape=6,
        # (int) the action shape, support discrete action only in this version
        action_shape=7,
        # (float) batch size
        batch_size=64,
        # (list) the MLP layer shape
        hidden_size_list=[64, 64, 128],
        # (int) update how many times after each collect
        update_per_collect=100,
        # (float) the importance weight of the forward and reverse loss
        reverse_scale=1,
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
        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.forward_mse = nn.MSELoss(reduction='none')
        self.reverse_scale = config.reverse_scale

    def _train(self) -> None:
        train_data_list = [i for i in range(0, len(self.train_states))]
        train_data_index = random.sample(train_data_list, self.cfg.batch_size)
        data_states: list = [self.train_states[i] for i in train_data_index]
        data_states: torch.Tensor = torch.stack(data_states).to(self.device)
        data_next_states: list = [self.train_next_states[i] for i in train_data_index]
        data_next_states: torch.Tensor = torch.stack(data_next_states).to(self.device)
        data_actions: list = [self.train_actions[i] for i in train_data_index]
        data_actions: torch.Tensor = torch.cat(data_actions).to(self.device)

        real_next_state_feature, pred_next_state_feature, pred_action_logit = self.reward_model(
            data_states, data_next_states, data_actions
        )
        inverse_loss = self.ce(pred_action_logit, data_actions.long())
        forward_loss = self.forward_mse(pred_next_state_feature, real_next_state_feature.detach()).mean()
        loss = self.reverse_scale * inverse_loss + forward_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self) -> None:
        for _ in range(self.cfg.update_per_collect):
            self._train()
        self.clear_data()

    def estimate(self, data: list) -> List[Dict]:
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)
        states, next_states, actions = collect_states(train_data_augmented)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        with torch.no_grad():
            real_next_state_feature, pred_next_state_feature, _ = self.reward_model(states, next_states, actions)
            reward = self.forward_mse(real_next_state_feature, pred_next_state_feature).mean(dim=1)
            reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-8)
            reward = reward.to(train_data_augmented[0]['reward'].device)
            reward = torch.chunk(reward, reward.shape[0], dim=0)
        for item, rew in zip(train_data_augmented, reward):
            if self.intrinsic_reward_type == 'add':
                item['reward'] += rew
            elif self.intrinsic_reward_type == 'new':
                item['intrinsic_reward'] = rew
            elif self.intrinsic_reward_type == 'assign':
                item['reward'] = rew

        return train_data_augmented

    def collect_data(self, data: list) -> None:
        self.train_data.extend(collect_states(data))
        states, next_states, actions = collect_states(data)
        self.train_states.extend(states)
        self.train_next_states.extend(next_states)
        self.train_actions.extend(actions)

    def clear_data(self) -> None:
        self.train_data.clear()
        self.train_states.clear()
        self.train_next_states.clear()
        self.train_actions.clear()
