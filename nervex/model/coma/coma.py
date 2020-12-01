from nervex.model.rnn_actor import RnnActorNetwork
import torch
import torch.nn as nn

from typing import Tuple, Dict, Union
from nervex.torch_utils import get_lstm
from nervex.utils import squeeze
import torch.nn.functional as F
from nervex.torch_utils import to_tensor, tensor_to_list, one_hot

ComaActorNetwork = RnnActorNetwork


class ComaCriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(ComaCriticNetwork, self).__init__()
        self._input_dim = input_dim
        self._act_dim = action_dim
        self._embedding_dim = embedding_dim
        self._act = F.relu
        self._fc1 = nn.Linear(squeeze(input_dim), embedding_dim)
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
                - last_action (:obj:`torch.Tensor`): the masked last action
        """
        x = self._preprocess_data(data)
        x = self._act(self._fc1(x))
        x = self._act(self._fc2(x))
        q = self._fc3(x)
        return {'agent_q': q}

    def _preprocess_data(self, data):
        assert len(data['obs']['global_state'].shape) in [2, 3]
        if len(data['obs']['global_state'].shape) == 2:
            # (B, ) -> (T=1, B, )
            data['obs']['global_state'].unsqueeze(0)
            data['obs']['agent_state'].unsqueeze(0)
            data['action'].unsqueeze(0)
            t_size = 1
        else:
            t_size = int(data['obs']['global_state'].shape[0])
        batch_size = int(data['obs']['agent_state'].shape[1])
        agent_num = int(data['obs']['agent_state'].shape[2])
        agent_state, global_state = data['obs']['agent_state'], data['obs']['global_state']
        action = data['action']
        action_onehot = one_hot(action, self._act_dim)
        last_action = data['last_action']
        last_action_onehot = one_hot(last_action, self._act_dim)
        inputs = []
        inputs.append(global_state.repeat(1, 1, agent_num, 1).reshape(t_size, batch_size, agent_num, -1))
        inputs.append(agent_state)
        inputs.append(action_onehot)
        inputs.append(last_action_onehot)
        for i in range(len(inputs)):
            inp = inputs[i]
            assert len(inp.shape) == 4
            assert inp.shape[0] == t_size
            assert inp.shape[1] == batch_size
            assert inp.shape[2] == agent_num
        x = torch.cat(inputs, -1)
        return x
