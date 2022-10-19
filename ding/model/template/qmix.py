from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, MODEL_REGISTRY
from ding.torch_utils import fc_block, MLP
from .q_learning import DRQN


class Mixer(nn.Module):
    """
    Overview:
        mixer network in QMIX, which mix up the independent q_value of each agent to a total q_value
    Interface:
        __init__, forward
    """

    def __init__(self, agent_num, state_dim, mixing_embed_dim, hypernet_embed=64, activation=nn.ReLU()):
        """
        Overview:
            Initialize mixer network proposed in QMIX.
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - state_dim(:obj:`int`): the dimension of global observation state
            - mixing_embed_dim (:obj:`int`): the dimension of mixing state emdedding
            - hypernet_embed (:obj:`int`): the dimension of hypernet emdedding, default to 64
            - activation (:obj:`nn.Module`): Activation function in network, defaults to nn.ReLU().
        """
        super(Mixer, self).__init__()

        self.n_agents = agent_num
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        self.act = activation
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed), self.act,
            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed), self.act, nn.Linear(hypernet_embed, self.embed_dim)
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), self.act, nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Overview:
            forward computation graph of pymarl mixer network
        Arguments:
            - agent_qs (:obj:`torch.FloatTensor`): the independent q_value of each agent
            - states (:obj:`torch.FloatTensor`): the emdedding vector of global state
        Returns:
            - q_tot (:obj:`torch.FloatTensor`): the total mixed q_value
        Shapes:
            - agent_qs (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is agent_num
            - states (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is embedding_size
            - q_tot (:obj:`torch.FloatTensor`): :math:`(B, )`
        """
        bs = agent_qs.shape[:-1]
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(*bs)
        return q_tot


@MODEL_REGISTRY.register('qmix')
class QMix(nn.Module):
    """
    Overview:
        QMIX network
    Interface:
        __init__, forward, _setup_global_encoder
    """

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            global_obs_shape: int,
            action_shape: int,
            hidden_size_list: list,
            mixer: bool = True,
            lstm_type: str = 'gru',
            activation: nn.Module = nn.ReLU(),
            dueling: bool = False
    ) -> None:
        """
        Overview:
            Initialize QMIX neural network, i.e. agent Q network and mixer.
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - global_obs_shape (:obj:`int`): the dimension of global observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size
            - mixer (:obj:`bool`): use mixer net or not, default to True
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - activation (:obj:`nn.Module`): Activation function in network, defaults to nn.ReLU().
            - dueling (:obj:`bool`): use dueling head or not, default to False.
        """
        super(QMix, self).__init__()
        self._act = activation
        self._q_network = DRQN(
            obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling, activation=activation
        )
        embedding_size = hidden_size_list[-1]
        self.mixer = mixer
        if self.mixer:
            self._mixer = Mixer(agent_num, global_obs_shape, embedding_size, activation=activation)
            self._global_state_encoder = nn.Identity()

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
        Returns:
            - ret (:obj:`dict`): output data dict with keys [``total_q``, ``logit``, ``next_state``]
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
        if self.mixer:
            global_state_embedding = self._global_state_encoder(global_state)
            total_q = self._mixer(agent_q_act, global_state_embedding)
        else:
            total_q = agent_q_act.sum(-1)
        if single_step:
            total_q, agent_q = total_q.squeeze(0), agent_q.squeeze(0)
        return {
            'total_q': total_q,
            'logit': agent_q,
            'next_state': next_state,
            'action_mask': data['obs']['action_mask']
        }

    def _setup_global_encoder(self, global_obs_shape: int, embedding_size: int) -> torch.nn.Module:
        """
        Overview:
            Used to encoder global observation.
        Arguments:
            - global_obs_shape (:obj:`int`): the dimension of global observation state
            - embedding_size (:obj:`int`): the dimension of state emdedding
        Return:
            - outputs (:obj:`torch.nn.Module`): Global observation encoding network
        """
        return MLP(global_obs_shape, embedding_size, embedding_size, 2, activation=self._act)
