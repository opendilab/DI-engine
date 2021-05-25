from typing import Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from nervex.model import FCRDiscreteNet
from nervex.torch_utils import one_hot
from nervex.utils import squeeze, list_split
from ..common import register_model


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
        # rnn discrete network
        self._main = FCRDiscreteNet(obs_dim, action_dim, embedding_dim)

    def forward(self, inputs: Dict) -> Dict:
        agent_state = inputs['obs']['agent_state']
        prev_state = inputs['prev_state']
        if len(agent_state.shape) == 3:  # B, A, N
            agent_state = agent_state.unsqueeze(0)
            unsqueeze_flag = True
        else:
            unsqueeze_flag = False
        T, B, A = agent_state.shape[:3]
        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        prev_state = reduce(lambda x, y: x + y, prev_state)
        output = self._main({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        logit, next_state = output['logit'], output['next_state']
        next_state, _ = list_split(next_state, step=A)
        logit = logit.reshape(T, B, A, -1)
        if unsqueeze_flag:
            logit = logit.squeeze(0)
        return {'logit': logit, 'next_state': next_state, 'action_mask': inputs['obs']['action_mask']}


class ComaCriticNetwork(nn.Module):

    def __init__(
            self,
            input_dim: int,
            action_dim: int,
            embedding_dim: int = 128,
    ):
        super(ComaCriticNetwork, self).__init__()
        self._input_dim = squeeze(input_dim)
        self._act_dim = squeeze(action_dim)
        self._embedding_dim = embedding_dim
        self._act = nn.ReLU()
        self._fc1 = nn.Linear(self._input_dim, embedding_dim)
        self._fc2 = nn.Linear(embedding_dim, embedding_dim)
        self._fc3 = nn.Linear(embedding_dim, action_dim)

    def forward(self, data: Dict) -> Dict:
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
        return {'q_value': q}

    def _preprocess_data(self, data: Dict) -> torch.Tensor:
        t_size, batch_size, agent_num = data['obs']['agent_state'].shape[:3]
        agent_state_ori, global_state = data['obs']['agent_state'], data['obs']['global_state']

        # splite obs, last_action and agent_id
        # TODO splite here beautifully or in env
        agent_state = agent_state_ori[:, :, :, :-self._act_dim - agent_num]
        last_action = agent_state_ori[:, :, :, -self._act_dim - agent_num:-agent_num].reshape(t_size, batch_size,1,-1).repeat(1, 1, agent_num, 1)
        agent_id = agent_state_ori[:, :, :, -agent_num:]

        action = one_hot(data['action'], self._act_dim)  # T, B, A，N
        action = action.reshape(t_size, batch_size, -1, agent_num * self._act_dim).repeat(1, 1, agent_num, 1)
        action_mask = (1 - torch.eye(agent_num).to(action.device))
        action_mask = action_mask.view(-1, 1).repeat(1, self._act_dim).view(agent_num, -1)  # A, A*N
        action = (action_mask.unsqueeze(0).unsqueeze(0)) * action  # T, B, A, A*N
        global_state = global_state.unsqueeze(2).repeat(1, 1, agent_num, 1)

        x = torch.cat([global_state, agent_state, last_action, action, agent_id], -1)
        return x


class ComaNetwork(nn.Module):

    def __init__(self, agent_num: int, obs_dim: int, global_obs_dim: int, action_dim: int, rnn_hidden_dim: int,
                 critic_dim: int):
        super(ComaNetwork, self).__init__()
        actor_input_dim = obs_dim
        critic_input_dim = obs_dim + global_obs_dim + agent_num * action_dim + (agent_num - 1) * action_dim
        self._actor = ComaActorNetwork(actor_input_dim, action_dim, rnn_hidden_dim)
        self._critic = ComaCriticNetwork(critic_input_dim, action_dim, critic_dim)

    def forward(self, data: Dict, mode: Union[str, None] = None) -> Dict:
        assert mode in ['compute_action', 'compute_q_value'], mode
        if mode == 'compute_action':
            return self._actor(data)
        elif mode == 'compute_q_value':
            return self._critic(data)


register_model('coma', ComaNetwork)
