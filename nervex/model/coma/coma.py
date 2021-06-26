from typing import Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from nervex.model import FCRDiscreteNet
from nervex.torch_utils import one_hot
from nervex.torch_utils.network.nn_module import MLP
from nervex.utils import squeeze, list_split, MODEL_REGISTRY


class ComaActorNetwork(nn.Module):

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size_list: list = [128, 128, 64],
    ):
        super(ComaActorNetwork, self).__init__()
        self._obs_shape = squeeze(obs_shape)
        self._act_shape = action_shape
        self._embedding_size = hidden_size_list[-1]
        # rnn discrete network
        self._main = FCRDiscreteNet(obs_shape, action_shape, hidden_size_list)

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
        input_size: int,
        action_shape: int,
        embedding_size: int = 128,
    ):
        super(ComaCriticNetwork, self).__init__()
        self._input_size = squeeze(input_size)
        self._act_shape = squeeze(action_shape)
        self._embedding_size = embedding_size
        self._act = nn.ReLU()
        self._mlp = nn.Sequential(
            MLP(self._input_size, embedding_size, embedding_size, 2, activation=self._act),
            nn.Linear(embedding_size, action_shape)
        )

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
        q = self._mlp(x)
        return {'q_value': q}

    def _preprocess_data(self, data: Dict) -> torch.Tensor:
        t_size, batch_size, agent_num = data['obs']['agent_state'].shape[:3]
        agent_state_ori, global_state = data['obs']['agent_state'], data['obs']['global_state']

        # splite obs, last_action and agent_id
        # TODO splite here beautifully or in env
        agent_state = agent_state_ori[:, :, :, :-self._act_shape - agent_num]
        last_action = agent_state_ori[:, :, :,
                                      -self._act_shape - agent_num:-agent_num].reshape(t_size, batch_size, 1,
                                                                                       -1).repeat(1, 1, agent_num, 1)
        agent_id = agent_state_ori[:, :, :, -agent_num:]

        action = one_hot(data['action'], self._act_shape)  # T, B, A，N
        action = action.reshape(t_size, batch_size, -1, agent_num * self._act_shape).repeat(1, 1, agent_num, 1)
        action_mask = (1 - torch.eye(agent_num).to(action.device))
        action_mask = action_mask.view(-1, 1).repeat(1, self._act_shape).view(agent_num, -1)  # A, A*N
        action = (action_mask.unsqueeze(0).unsqueeze(0)) * action  # T, B, A, A*N
        global_state = global_state.unsqueeze(2).repeat(1, 1, agent_num, 1)

        x = torch.cat([global_state, agent_state, last_action, action, agent_id], -1)
        return x


@MODEL_REGISTRY.register('coma')
class ComaNetwork(nn.Module):

    def __init__(self, agent_num: int, obs_shape: dict, action_shape: Tuple, hidden_size_list: list):
        super(ComaNetwork, self).__init__()
        actor_input_size = obs_shape['agent_state']
        critic_input_size = obs_shape['agent_state'] + squeeze(
            obs_shape['global_state']
        ) + agent_num * action_shape + (agent_num - 1) * action_shape
        embedding_size = hidden_size_list[-1]
        self._actor = ComaActorNetwork(actor_input_size, action_shape, hidden_size_list)
        self._critic = ComaCriticNetwork(critic_input_size, action_shape, embedding_size)

    def forward(self, data: Dict, mode: Union[str, None] = None) -> Dict:
        assert mode in ['compute_actor', 'compute_critic'], mode
        if mode == 'compute_actor':
            return self._actor(data)
        elif mode == 'compute_critic':
            return self._critic(data)
