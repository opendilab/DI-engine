from typing import Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from nervex.torch_utils import to_tensor, tensor_to_list, one_hot, get_lstm
from nervex.utils import squeeze


class ComaActorNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embedding_dim: int = 64,
    ):
        super(ComaActorNetwork, self).__init__()
        self._obs_dim = squeeze(obs_dim)
        self._act_dim = action_dim
        self._embedding_dim = embedding_dim
        self._fc1 = nn.Linear(self._obs_dim, embedding_dim)
        self._act = nn.ReLU()
        self._rnn = get_lstm('normal', embedding_dim, embedding_dim)
        self._fc2 = nn.Linear(embedding_dim, squeeze(action_dim))

    def forward(self, inputs: Dict) -> Dict:
        x = self._fc1(inputs['obs'])
        x = self._act(x)
        x = x.unsqueeze(0)
        x, next_state = self._rnn(x, inputs['prev_state'])
        x = x.squeeze(0)
        x = self._fc2(x)
        return {'logit': x, 'next_state': next_state}


class ComaCriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
    ):
        super(ComaCriticNetwork, self).__init__()
        self._input_dim = squeeze(input_dim)
        self._act_dim = action_dim
        self._embedding_dim = embedding_dim
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(self._input_dim, embedding_dim)
        self._fc2 = nn.Linear(embedding_dim, embedding_dim)
        self._fc3 = nn.Linear(embedding_dim, squeeze(action_dim))

    def forward(self, data):
        """
        Overview:
            forward computation graph of qmix network
        Arguments:
            - data (:obj:`dict`): input data dict with keys ['obs', 'prev_state', 'action']
                - agent_state (:obj:`torch.Tensor`): each agent local state(obs)
                - global_state (:obj:`torch.Tensor`): global state(obs)
                - action (:obj:`torch.Tensor`): the masked action
        """
        x = self._preprocess_data(data)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        q = self._fc3(x)
        return {'total_q': q}

    def _preprocess_data(self, data):
        #assert len(data['obs']['global_state'].shape) in [2, 3]
        #if len(data['obs']['global_state'].shape) == 2:
        # (B, ) -> (T=1, B, )
        #    data['obs']['global_state'].unsqueeze(0)
        #    data['obs']['agent_state'].unsqueeze(0)
        #    data['action'].unsqueeze(0)
        t_size, batch_size, agent_num = data['obs']['agent_state'].shape[:3]
        agent_state, global_state = data['obs']['agent_state'], data['obs']['global_state']
        action_onehot = one_hot(data['action'], self._act_dim)
        global_state = global_state.unsqueeze(2).repeat(1, 1, agent_num, 1)

        x = torch.cat([global_state, agent_state, action_onehot], -1)
        return x


class ComaNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, embedding_dim):
        super(ComaNetwork, self).__init__()
        act_dim = squeeze(act_dim)
        actor_input_dim = obs_dim['agent_state'][-1]
        critic_input_dim = obs_dim['agent_state'][-1] + squeeze(obs_dim['global_state']) + act_dim
        self._actor = ComaActorNetwork(actor_input_dim, act_dim, embedding_dim)
        self._critic = ComaCriticNetwork(critic_input_dim, act_dim, embedding_dim)

    def forward(self, data, mode=None):
        assert mode in ['compute_action']
        if mode == 'compute_action':
            return self._actor(data)
        elif mode == 'comput_q_value':
            return self._critic(data)
