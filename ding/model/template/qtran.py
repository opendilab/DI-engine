from typing import Union, List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, squeeze, MODEL_REGISTRY
from ding.torch_utils.network.nn_module import fc_block, MLP
from ding.torch_utils.network.transformer import ScaledDotProductAttention
from ding.torch_utils import to_tensor, tensor_to_list
from .q_learning import DRQN


@MODEL_REGISTRY.register('qtran')
class QTran(nn.Module):
    """
    Overview:
        QTRAN network
    Interface:
        __init__, forward
    """

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            global_obs_shape: int,
            action_shape: int,
            hidden_size_list: list,
            embedding_size: int,
            lstm_type: str = 'gru',
            dueling: bool = False
    ) -> None:
        """
        Overview:
            initialize QTRAN network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - global_obs_shape (:obj:`int`): the dimension of global observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size
            - embedding_size (:obj:`int`): the dimension of embedding
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - dueling (:obj:`bool`): use dueling head or not, default to False.
        """
        super(QTran, self).__init__()
        self._act = nn.ReLU()
        self._q_network = DRQN(obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
        q_input_size = global_obs_shape + hidden_size_list[-1] + action_shape
        self.Q = nn.Sequential(
            nn.Linear(q_input_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size), nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

        # V(s)
        self.V = nn.Sequential(
            nn.Linear(global_obs_shape, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size),
            nn.ReLU(), nn.Linear(embedding_size, 1)
        )
        ae_input = hidden_size_list[-1] + action_shape
        self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input), nn.ReLU(), nn.Linear(ae_input, ae_input))

    def forward(self, data: dict, single_step: bool = True) -> dict:
        """
        Overview:
            forward computation graph of qtran network
        Arguments:
            - data (:obj:`dict`): input data dict with keys ['obs', 'prev_state', 'action']
                - agent_state (:obj:`torch.Tensor`): each agent local state(obs)
                - global_state (:obj:`torch.Tensor`): global state(obs)
                - prev_state (:obj:`list`): previous rnn state
                - action (:obj:`torch.Tensor` or None): if action is None, use argmax q_value index as action to\
                    calculate ``agent_q_act``
            - single_step (:obj:`bool`): whether single_step forward, if so, add timestep dim before forward and\
                remove it after forward
        Return:
            - ret (:obj:`dict`): output data dict with keys ['total_q', 'logit', 'next_state']
                - total_q (:obj:`torch.Tensor`): total q_value, which is the result of mixer network
                - agent_q (:obj:`torch.Tensor`): each agent q_value
                - next_state (:obj:`list`): next rnn state
        Shapes:
            - agent_state (:obj:`torch.Tensor`): :math:`(T, B, A, N)`, where T is timestep, B is batch_size\
                A is agent_num, N is obs_shape
            - global_state (:obj:`torch.Tensor`): :math:`(T, B, M)`, where M is global_obs_shape
            - prev_state (:obj:`list`): math:`(B, A)`, a list of length B, and each element is a list of length A
            - action (:obj:`torch.Tensor`): :math:`(T, B, A)`
            - total_q (:obj:`torch.Tensor`): :math:`(T, B)`
            - agent_q (:obj:`torch.Tensor`): :math:`(T, B, A, P)`, where P is action_shape
            - next_state (:obj:`list`): math:`(B, A)`, a list of length B, and each element is a list of length A
        """
        agent_state, global_state, prev_state = data['obs']['agent_state'], data['obs']['global_state'], data[
            'prev_state']
        action = data.get('action', None)
        if single_step:
            agent_state, global_state = agent_state.unsqueeze(0), global_state.unsqueeze(0)
        T, B, A = agent_state.shape[:3]
        assert len(prev_state) == B and all(
            [len(p) == A for p in prev_state]
        ), '{}-{}-{}-{}'.format([type(p) for p in prev_state], B, A, len(prev_state[0]))
        prev_state = reduce(lambda x, y: x + y, prev_state)
        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        output = self._q_network({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        agent_q, next_state = output['logit'], output['next_state']
        next_state, _ = list_split(next_state, step=A)
        agent_q = agent_q.reshape(T, B, A, -1)
        if action is None:
            # For target forward process
            if len(data['obs']['action_mask'].shape) == 3:
                action_mask = data['obs']['action_mask'].unsqueeze(0)
            else:
                action_mask = data['obs']['action_mask']
            agent_q[action_mask == 0.0] = -9999999
            action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        agent_q_act = agent_q_act.squeeze(-1)  # T, B, A

        hidden_states = output['hidden_state'].reshape(T * B, A, -1)
        action = action.reshape(T * B, A).unsqueeze(-1)
        action_onehot = torch.zeros(size=(T * B, A, agent_q.shape[-1]), device=action.device)
        action_onehot = action_onehot.scatter(2, action, 1)
        agent_state_action_input = torch.cat([hidden_states, action_onehot], dim=2)
        agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(T * B * A,
                                                                                            -1)).reshape(T * B, A, -1)
        agent_state_action_encoding = agent_state_action_encoding.sum(dim=1)  # Sum across agents

        inputs = torch.cat([global_state.reshape(T * B, -1), agent_state_action_encoding], dim=1)
        q_outputs = self.Q(inputs)
        q_outputs = q_outputs.reshape(T, B)
        v_outputs = self.V(global_state.reshape(T * B, -1))
        v_outputs = v_outputs.reshape(T, B)
        if single_step:
            q_outputs, agent_q, agent_q_act, v_outputs = q_outputs.squeeze(0), agent_q.squeeze(0), agent_q_act.squeeze(
                0
            ), v_outputs.squeeze(0)
        return {
            'total_q': q_outputs,
            'logit': agent_q,
            'agent_q_act': agent_q_act,
            'vs': v_outputs,
            'next_state': next_state,
            'action_mask': data['obs']['action_mask']
        }
