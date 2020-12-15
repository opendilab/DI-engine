import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from nervex.model import FCRDiscreteNet
from nervex.utils import list_split, squeeze


class Mixer(nn.Module):
    """
    Overview:
        mixer network in QMIX, which mix up the independent q_value of each agent to a total q_value
    Interface:
        __init__, forward
    """

    def __init__(self, agent_num: int, embedding_dim: int, w_layers: int = 2) -> None:
        """
        Overview:
            initialize mixer network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - embedding_dim (:obj:`int`): the dimension of state emdedding
            - w_layers (;obj:`int`): the number of fully-connected layers of mixer weight
        """
        super(Mixer, self).__init__()
        self._agent_num = agent_num
        self._embedding_dim = embedding_dim
        self._act = nn.ReLU()
        self._w1 = []
        for i in range(w_layers - 1):
            self._w1.append(nn.Linear(embedding_dim, embedding_dim))
            self._w1.append(self._act)
        self._w1.append(nn.Linear(embedding_dim, embedding_dim * agent_num))
        self._w1 = nn.Sequential(*self._w1)

        self._w2 = []
        for i in range(w_layers - 1):
            self._w2.append(nn.Linear(embedding_dim, embedding_dim))
            self._w2.append(self._act)
        self._w2.append(nn.Linear(embedding_dim, embedding_dim))
        self._w2 = nn.Sequential(*self._w2)

        self._b1 = nn.Linear(embedding_dim, embedding_dim)
        self._b2 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), self._act, nn.Linear(embedding_dim, 1))

    def forward(self, agent_q: torch.Tensor, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            forward computation graph of mixer network
        Arguments:
            - agent_q (:obj:`torch.FloatTensor`): the independent q_value of each agent
            - state_embedding (:obj:`torch.FloatTensor`): the emdedding vector of global state
        Returns:
            - total_q (:obj:`torch.FloatTensor`): the total mixed q_value
        Shapes:
            - agent_q (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is agent_num
            - state_embedding (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is embedding_dim
            - total_q (:obj:`torch.FloatTensor`): :math:`(B, )`
        """
        agent_q = agent_q.reshape(-1, 1, self._agent_num)
        # first layer
        w1 = torch.abs(self._w1(state_embedding)).reshape(-1, self._agent_num, self._embedding_dim)
        b1 = self._b1(state_embedding).reshape(-1, 1, self._embedding_dim)
        hidden = F.elu(torch.bmm(agent_q, w1) + b1)  # bs, 1, embedding_dim
        # second layer
        w2 = torch.abs(self._w2(state_embedding)).reshape(-1, self._embedding_dim, 1)
        b2 = self._b2(state_embedding).reshape(-1, 1, 1)
        hidden = torch.bmm(hidden, w2) + b2
        return hidden.squeeze(-1).squeeze(-1)


class QMix(nn.Module):

    def __init__(self, agent_num: int, obs_dim: int, global_obs_dim: int, action_dim: int, embedding_dim: int) -> None:
        super(QMix, self).__init__()
        self._act = nn.ReLU()
        self._q_network = FCRDiscreteNet(obs_dim, action_dim, embedding_dim)
        self._mixer = Mixer(agent_num, embedding_dim)
        global_obs_dim = squeeze(global_obs_dim)
        self._global_state_encoder = self._setup_global_encoder(global_obs_dim, embedding_dim)

    def forward(self, data: dict, single_step: bool = True) -> dict:
        """
        Overview:
            forward computation graph of qmix network
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
                A is agent_num, N is obs_dim
            - global_state (:obj:`torch.Tensor`): :math:`(T, B, M)`, where M is global_obs_dim
            - prev_state (:obj:`list`): math:`(B, A)`, a list of length B, and each element is a list of length A
            - action (:obj:`torch.Tensor`): :math:`(T, B, A)`
            - total_q (:obj:`torch.Tensor`): :math:`(T, B)`
            - agent_q (:obj:`torch.Tensor`): :math:`(T, B, A, P)`, where P is action_dim
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
        global_state_embedding = self._global_state_encoder(global_state)
        output = self._q_network({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        agent_q, next_state = output['logit'], output['next_state']
        next_state = list_split(next_state, step=A)
        agent_q = agent_q.reshape(T, B, A, -1)
        if action is None:
            action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        total_q = self._mixer(agent_q_act, global_state_embedding).reshape(T, B)
        if single_step:
            total_q, agent_q = total_q.squeeze(0), agent_q.squeeze(0)
        return {
            'total_q': total_q,
            'logit': agent_q,
            'next_state': next_state,
            'action_mask': data['obs']['action_mask']
        }

    def _setup_global_encoder(self, global_obs_dim: int, embedding_dim: int) -> torch.nn.Module:
        layers = []
        layer_num = 1
        layers.append(nn.Linear(global_obs_dim, embedding_dim))
        layers.append(self._act)
        for _ in range(layer_num):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(self._act)
        return nn.Sequential(*layers)
