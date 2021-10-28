from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, MODEL_REGISTRY
from ding.torch_utils.network.nn_module import fc_block, MLP
from ding.torch_utils.network.transformer import ScaledDotProductAttention
from .q_learning import DRQN


class Mixer(nn.Module):
    """
    Overview:
        mixer network in QMIX, which mix up the independent q_value of each agent to a total q_value
    Interface:
        __init__, forward
    """

    def __init__(self, agent_num, state_dim, mixing_embed_dim, hypernet_embed=64):
        """
        Overview:
            initialize pymarl mixer network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - state_dim(:obj:`int`): the dimension of global observation state
            - mixing_embed_dim (:obj:`int`): the dimension of mixing state emdedding
            - hypernet_embed (:obj:`int`): the dimension of hypernet emdedding, default to 64
        """
        super(Mixer, self).__init__()

        self.n_agents = agent_num
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, self.embed_dim)
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

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
            - mixer (:obj:`bool`): use mixer net or not, default to True
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - dueling (:obj:`bool`): use dueling head or not, default to False.
        """
        super(QMix, self).__init__()
        self._act = nn.ReLU()
        self._q_network = DRQN(obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
        embedding_size = hidden_size_list[-1]
        self.mixer = mixer
        if self.mixer:
            self._mixer = Mixer(agent_num, global_obs_shape, embedding_size)
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


class CollaQMultiHeadAttention(nn.Module):
    """
    Overview:
        The head of collaq attention module.
    Interface:
        __init__, forward
    """

    def __init__(
        self, n_head: int, d_model_q: int, d_model_v: int, d_k: int, d_v: int, d_out: int, dropout: float = 0.
    ):
        """
        Overview:
            initialize the head of collaq attention module
        Arguments:
            - n_head (:obj:`int`): the num of head
            - d_model_q (:obj:`int`): the size of input q
            - d_model_v (:obj:`int`): the size of input v
            - d_k (:obj:`int`): the size of k, used by Scaled Dot Product Attention
            - d_v (:obj:`int`): the size of v, used by Scaled Dot Product Attention
            - d_out (:obj:`int`): the size of output q
        """
        super(CollaQMultiHeadAttention, self).__init__()

        self.act = nn.ReLU()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_q, n_head * d_k)
        self.w_ks = nn.Linear(d_model_v, n_head * d_k)
        self.w_vs = nn.Linear(d_model_v, n_head * d_v)

        self.fc1 = fc_block(n_head * d_v, n_head * d_v, activation=self.act)
        self.fc2 = fc_block(n_head * d_v, d_out)

        self.attention = ScaledDotProductAttention(d_k=d_k)
        self.layer_norm_q = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        Overview:
            forward computation graph of collaQ multi head attention net.
        Arguments:
            - q (:obj:`torch.nn.Sequential`): the transformer information q
            - k (:obj:`torch.nn.Sequential`): the transformer information k
            - v (:obj:`torch.nn.Sequential`): the transformer information v
        Output:
            - q (:obj:`torch.nn.Sequential`): the transformer output q
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: batch_size x len_q x (n_head * d_v)
        # Separate different heads: batch_size x len_q x n_head x d_v
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)
        residual = q

        # Transpose for attention dot product: batch_size x n_head x len_q x d_v
        q, k, v = self.layer_norm_q(q).transpose(1, 2), self.layer_norm_k(k).transpose(
            1, 2
        ), self.layer_norm_v(v).transpose(1, 2)
        # Unsqueeze the mask tensor for head axis broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1)
        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: batch_size x len_q x n_head x d_v
        # Combine the last two dimensions to concatenate all the heads together: batch_size x len_q x (n*dv)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.fc2(self.fc1(q))
        return q, residual


class CollaQSMACAttentionModule(nn.Module):
    """
    Overview:
        Collaq attention module. Used to get agent's attention observation. It includes agent's observation\
            and agent's part of the observation information of the agent's concerned allies
    Interface:
        __init__, _cut_obs, forward
    """

    def __init__(
        self, q_dim: int, v_dim: int, self_feature_range: List[int], ally_feature_range: List[int], attention_size: int
    ):
        """
        Overview:
            initialize collaq attention module
        Arguments:
            - q_dim (:obj:`int`): the dimension of transformer output q
            - v_dim (:obj:`int`): the dimension of transformer output v
            - self_features (:obj:`torch.Tensor`): output self agent's attention observation
            - ally_features (:obj:`torch.Tensor`): output ally agent's attention observation
            - attention_size (:obj:`int`): the size of attention net layer
        """
        super(CollaQSMACAttentionModule, self).__init__()
        self.self_feature_range = self_feature_range
        self.ally_feature_range = ally_feature_range
        self.attention_layer = CollaQMultiHeadAttention(1, q_dim, v_dim, attention_size, attention_size, attention_size)

    def _cut_obs(self, obs: torch.Tensor):
        """
        Overview:
            cut the observed information into self's observation and allay's observation
        Arguments:
            - obs (:obj:`torch.Tensor`): input each agent's observation
        Return:
            - self_features (:obj:`torch.Tensor`): output self agent's attention observation
            - ally_features (:obj:`torch.Tensor`): output ally agent's attention observation
        """
        # obs shape = (T, B, A, obs_shape)
        self_features = obs[:, :, :, self.self_feature_range[0]:self.self_feature_range[1]]
        ally_features = obs[:, :, :, self.ally_feature_range[0]:self.ally_feature_range[1]]
        return self_features, ally_features

    def forward(self, inputs: torch.Tensor):
        """
        Overview:
            forward computation to get agent's attention observation information
        Arguments:
            - obs (:obj:`torch.Tensor`): input each agent's observation
        Return:
            - obs (:obj:`torch.Tensor`): output agent's attention observation
        """
        # obs shape = (T, B ,A, obs_shape)
        obs = inputs
        self_features, ally_features = self._cut_obs(obs)
        T, B, A, _ = self_features.shape
        self_features = self_features.reshape(T * B * A, 1, -1)
        ally_features = ally_features.reshape(T * B * A, A - 1, -1)
        self_features, ally_features = self.attention_layer(self_features, ally_features, ally_features)
        self_features = self_features.reshape(T, B, A, -1)
        ally_features = ally_features.reshape(T, B, A, -1)
        # note: we assume self_feature is near the ally_feature here so we can do this concat
        obs = torch.cat(
            [
                obs[:, :, :, :self.self_feature_range[0]], self_features, ally_features,
                obs[:, :, :, self.ally_feature_range[1]:]
            ],
            dim=-1
        )
        return obs


@MODEL_REGISTRY.register('collaq')
class CollaQ(nn.Module):
    """
    Overview:
        CollaQ network
    Interface:
        __init__, forward, _setup_global_encoder
    """

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            alone_obs_shape: int,
            global_obs_shape: int,
            action_shape: int,
            hidden_size_list: list,
            attention: bool = False,
            self_feature_range: Union[List[int], None] = None,
            ally_feature_range: Union[List[int], None] = None,
            attention_size: int = 32,
            mixer: bool = True,
            lstm_type: str = 'gru',
            dueling: bool = False,
    ) -> None:
        """
        Overview:
            initialize Collaq network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - alone_obs_shape (:obj:`int`): the dimension of each agent's observation state without\
                other agents
            - global_obs_shape (:obj:`int`): the dimension of global observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size
            - attention (:obj:`bool`): use attention module or not, default to False
            - self_feature_range (:obj:`Union[List[int], None]`): the agent's feature range
            - ally_feature_range (:obj:`Union[List[int], None]`): the agent ally's feature range
            - attention_size (:obj:`int`): the size of attention net layer
            - mixer (:obj:`bool`): use mixer net or not, default to True
            - lstm_type (:obj:`str`): use lstm or gru, default to gru
            - dueling (:obj:`bool`): use dueling head or not, default to False.
        """
        super(CollaQ, self).__init__()
        self.attention = attention
        self.attention_size = attention_size
        self._act = nn.ReLU()
        self.mixer = mixer
        if not self.attention:
            self._q_network = DRQN(obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling)
        else:
            # TODO set the attention layer here beautifully
            self._self_attention = CollaQSMACAttentionModule(
                self_feature_range[1] - self_feature_range[0],
                (ally_feature_range[1] - ally_feature_range[0]) // (agent_num - 1), self_feature_range,
                ally_feature_range, attention_size
            )
            # TODO get the obs_dim_after_attention here beautifully
            obs_shape_after_attention = self._self_attention(
                # torch.randn(
                #     1, 1, (ally_feature_range[1] - ally_feature_range[0]) //
                #           ((self_feature_range[1] - self_feature_range[0])*2) + 1, obs_dim
                # )
                torch.randn(1, 1, agent_num, obs_shape)
            ).shape[-1]
            self._q_network = DRQN(
                obs_shape_after_attention, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling
            )
        self._q_alone_network = DRQN(
            alone_obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling
        )
        embedding_size = hidden_size_list[-1]
        if self.mixer:
            self._mixer = Mixer(agent_num, global_obs_shape, embedding_size)
            self._global_state_encoder = nn.Identity()

    def forward(self, data: dict, single_step: bool = True) -> dict:
        """
        Overview:
            forward computation graph of collaQ network
        Arguments:
            - data (:obj:`dict`): input data dict with keys ['obs', 'prev_state', 'action']
                - agent_state (:obj:`torch.Tensor`): each agent local state(obs)
                - agent_alone_state (:obj:`torch.Tensor`): each agent's local state alone, \
                    in smac setting is without ally feature(obs_along)
                - global_state (:obj:`torch.Tensor`): global state(obs)
                - prev_state (:obj:`list`): previous rnn state, should include 3 parts: \
                    one hidden state of q_network, and two hidden state if q_alone_network for obs and obs_alone inputs
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
        agent_state, agent_alone_state = data['obs']['agent_state'], data['obs']['agent_alone_state']
        agent_alone_padding_state = data['obs']['agent_alone_padding_state']
        global_state, prev_state = data['obs']['global_state'], data['prev_state']
        # TODO find a better way to implement agent_along_padding_state

        action = data.get('action', None)
        if single_step:
            agent_state, agent_alone_state, agent_alone_padding_state, global_state = agent_state.unsqueeze(
                0
            ), agent_alone_state.unsqueeze(0), agent_alone_padding_state.unsqueeze(0), global_state.unsqueeze(0)
        T, B, A = agent_state.shape[:3]

        if self.attention:
            agent_state = self._self_attention(agent_state)
            agent_alone_padding_state = self._self_attention(agent_alone_padding_state)

        # prev state should be of size (B, 3, A) hidden_size)
        """
        Note: to achieve such work, we should change the init_fn of hidden_state plugin in collaQ policy
        """
        assert len(prev_state) == B and all([len(p) == 3 for p in prev_state]) and all(
            [len(q) == A] for p in prev_state for q in p
        ), '{}-{}-{}-{}'.format([type(p) for p in prev_state], B, A, len(prev_state[0]))

        alone_prev_state = [[None for _ in range(A)] for _ in range(B)]
        colla_prev_state = [[None for _ in range(A)] for _ in range(B)]
        colla_alone_prev_state = [[None for _ in range(A)] for _ in range(B)]

        for i in range(B):
            for j in range(3):
                for k in range(A):
                    if j == 0:
                        alone_prev_state[i][k] = prev_state[i][j][k]
                    elif j == 1:
                        colla_prev_state[i][k] = prev_state[i][j][k]
                    elif j == 2:
                        colla_alone_prev_state[i][k] = prev_state[i][j][k]

        alone_prev_state = reduce(lambda x, y: x + y, alone_prev_state)
        colla_prev_state = reduce(lambda x, y: x + y, colla_prev_state)
        colla_alone_prev_state = reduce(lambda x, y: x + y, colla_alone_prev_state)

        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        agent_alone_state = agent_alone_state.reshape(T, -1, *agent_alone_state.shape[3:])
        agent_alone_padding_state = agent_alone_padding_state.reshape(T, -1, *agent_alone_padding_state.shape[3:])

        colla_output = self._q_network(
            {
                'obs': agent_state,
                'prev_state': colla_prev_state,
                'enable_fast_timestep': True
            }
        )
        colla_alone_output = self._q_network(
            {
                'obs': agent_alone_padding_state,
                'prev_state': colla_alone_prev_state,
                'enable_fast_timestep': True
            }
        )
        alone_output = self._q_alone_network(
            {
                'obs': agent_alone_state,
                'prev_state': alone_prev_state,
                'enable_fast_timestep': True
            }
        )

        agent_alone_q, alone_next_state = alone_output['logit'], alone_output['next_state']
        agent_colla_alone_q, colla_alone_next_state = colla_alone_output['logit'], colla_alone_output['next_state']
        agent_colla_q, colla_next_state = colla_output['logit'], colla_output['next_state']

        colla_next_state, _ = list_split(colla_next_state, step=A)
        alone_next_state, _ = list_split(alone_next_state, step=A)
        colla_alone_next_state, _ = list_split(colla_alone_next_state, step=A)

        next_state = list(
            map(lambda x: [x[0], x[1], x[2]], zip(alone_next_state, colla_next_state, colla_alone_next_state))
        )

        agent_alone_q = agent_alone_q.reshape(T, B, A, -1)
        agent_colla_alone_q = agent_colla_alone_q.reshape(T, B, A, -1)
        agent_colla_q = agent_colla_q.reshape(T, B, A, -1)

        total_q_before_mix = agent_alone_q + agent_colla_q - agent_colla_alone_q
        # total_q_before_mix = agent_colla_q
        # total_q_before_mix = agent_alone_q
        agent_q = total_q_before_mix

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
            total_q, agent_q, agent_colla_alone_q = total_q.squeeze(0), agent_q.squeeze(0), agent_colla_alone_q.squeeze(
                0
            )
        return {
            'total_q': total_q,
            'logit': agent_q,
            'agent_colla_alone_q': agent_colla_alone_q * data['obs']['action_mask'],
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
