from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, MODEL_REGISTRY
from ding.torch_utils.network.nn_module import fc_block, MLP
from ding.torch_utils.network.transformer import ScaledDotProductAttention
from .q_learning import DRQN
from ding.model.template.qmix import Mixer


class MixerStar(nn.Module):
    """
    Overview:
        mixer network for Q_star in WQMIX , which mix up the independent q_value of
        each agent to a total q_value and is diffrent from the Qmix's mixer network,
        here the mixing network is a feedforward network with 3 hidden layers of 256 dim.
    Interface:
        __init__, forward
    """

    def __init__(self, agent_num: int, state_dim: int, mixing_embed_dim: int) -> None:
        """
        Overview:
            initialize the mixer network of Q_star in WQMIX.
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - state_dim(:obj:`int`): the dimension of global observation state
            - mixing_embed_dim (:obj:`int`): the dimension of mixing state emdedding
        """
        super(MixerStar, self).__init__()
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        self.input_dim = self.agent_num + self.state_dim  # shape N+A
        non_lin = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim), non_lin, nn.Linear(self.embed_dim, self.embed_dim), non_lin,
            nn.Linear(self.embed_dim, self.embed_dim), non_lin, nn.Linear(self.embed_dim, 1)
        )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), non_lin, nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs: torch.FloatTensor, states: torch.FloatTensor) -> torch.FloatTensor:
        """
        Overview:
            forward computation graph of the mixer network for Q_star in WQMIX.
        Arguments:
            - agent_qs (:obj:`torch.FloatTensor`): the independent q_value of each agent
            - states (:obj:`torch.FloatTensor`): the emdedding vector of global state
        Returns:
            - q_tot (:obj:`torch.FloatTensor`): the total mixed q_value
        Shapes:
            - agent_qs (:obj:`torch.FloatTensor`): :math:`(T,B, N)`, where T is timestep,
              B is batch size, A is agent_num, N is obs_shape
            - states (:obj:`torch.FloatTensor`): :math:`(T, B, M)`, where M is global_obs_shape
            - q_tot (:obj:`torch.FloatTensor`): :math:`(T, B, )`
        """
        # in below annotations about the shape of the variables, T is timestep,
        # B is batch_size A is agent_num, N is obs_shape， for example,
        # in 3s5z, we can set T=10, B=32, A=8, N=216
        bs = agent_qs.shape[:-1]  # (T*B, A)
        states = states.reshape(-1, self.state_dim)  # T*B, N),
        agent_qs = agent_qs.reshape(-1, self.agent_num)  # (T, B, A) -> (T*B, A)
        inputs = torch.cat([states, agent_qs], dim=1)  # (T*B, N) (T*B, A)-> (T*B, N+A)
        advs = self.net(inputs)  # (T*B, 1)
        vs = self.V(states)  # (T*B, 1)
        y = advs + vs
        q_tot = y.view(*bs)  # (T*B, 1) -> (T, B)

        return q_tot


@MODEL_REGISTRY.register('wqmix')
class WQMix(nn.Module):
    """
    Overview:
        WQMIX network, which is same as Qmix network
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
            lstm_type: str = 'gru',
            dueling: bool = False
    ) -> None:
        """
        Overview:
            initialize Qmix network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - global_obs_shape (:obj:`int`): the dimension of global observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - dueling (:obj:`bool`): use dueling head or not, default to False.
        """
        super(WQMix, self).__init__()
        self._act = nn.ReLU()
        self._q_network = DRQN(obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
        self._q_network_star = DRQN(obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
        embedding_size = hidden_size_list[-1]
        self._mixer = Mixer(agent_num, global_obs_shape, mixing_embed_dim=embedding_size)
        self._mixer_star = MixerStar(
            agent_num, global_obs_shape, mixing_embed_dim=256
        )  # the mixing network of Q_star is a feedforward network with 3 hidden layers of 256 dim
        self._global_state_encoder = nn.Identity()  # nn.Sequential()

    def forward(self, data: dict, single_step: bool = True, q_star: bool = False) -> dict:
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
            - Q_star (:obj:`bool`): whether Q_star network forward. If True, using the Q_star network, where the\
                agent networks have the same architecture as Q network but do not share parameters and the mixing\
                network is a feedforward network with 3 hidden layers of 256 dim; if False, using the Q network,\
                same as the Q network in Qmix paper.
        Returns:
            - ret (:obj:`dict`): output data dict with keys [``total_q``, ``logit``, ``next_state``]
            - total_q (:obj:`torch.Tensor`): total q_value, which is the result of mixer network
            - agent_q (:obj:`torch.Tensor`): each agent q_value
            - next_state (:obj:`list`): next rnn state
        Shapes:
            - agent_state (:obj:`torch.Tensor`): :math:`(T, B, A, N)`, where T is timestep, B is batch_size\
                A is agent_num, N is obs_shape
            - global_state (:obj:`torch.Tensor`): :math:`(T, B, M)`, where M is global_obs_shape
            - prev_state (:obj:`list`): math:`(T, B, A)`, a list of length B, and each element is a list of length A
            - action (:obj:`torch.Tensor`): :math:`(T, B, A)`
            - total_q (:obj:`torch.Tensor`): :math:`(T, B)`
            - agent_q (:obj:`torch.Tensor`): :math:`(T, B, A, P)`, where P is action_shape
            - next_state (:obj:`list`): math:`(T, B, A)`, a list of length B, and each element is a list of length A
        """
        if q_star:  # forward using Q_star network
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
            output = self._q_network_star(
                {
                    'obs': agent_state,
                    'prev_state': prev_state,
                    'enable_fast_timestep': True
                }
            )  # here is the forward pass of the agent networks of Q_star
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

            global_state_embedding = self._global_state_encoder(global_state)
            total_q = self._mixer_star(
                agent_q_act, global_state_embedding
            )  # here is the forward pass of the mixer networks of Q_star

            if single_step:
                total_q, agent_q = total_q.squeeze(0), agent_q.squeeze(0)
            return {
                'total_q': total_q,
                'logit': agent_q,
                'next_state': next_state,
                'action_mask': data['obs']['action_mask']
            }
        else:  # forward using Q network
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
            output = self._q_network(
                {
                    'obs': agent_state,
                    'prev_state': prev_state,
                    'enable_fast_timestep': True
                }
            )  # here is the forward pass of the agent networks of Q
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

            global_state_embedding = self._global_state_encoder(global_state)
            total_q = self._mixer(
                agent_q_act, global_state_embedding
            )  # here is the forward pass of the mixer networks of Q

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
