from typing import Union, List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from nervex.model import FCRDiscreteNet
from nervex.utils import list_split, squeeze
from nervex.torch_utils.network.nn_module import fc_block
from nervex.torch_utils.network.transformer import ScaledDotProductAttention
from nervex.torch_utils import to_tensor, tensor_to_list
from ..common import register_model


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
            - w_layers (:obj:`int`): the number of fully-connected layers of mixer weight
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
    """
    Overview:
        QMIX network
    Interface:
        __init__, forward
    """

    def __init__(
            self,
            agent_num: int,
            obs_dim: int,
            global_obs_dim: int,
            action_dim: int,
            embedding_dim: int,
            use_mixer: bool = True
    ) -> None:
        super(QMix, self).__init__()
        self._act = nn.ReLU()
        self._q_network = FCRDiscreteNet(obs_dim, action_dim, embedding_dim)
        self.use_mixer = use_mixer
        if self.use_mixer:
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
            agent_q[action_mask == 0.0] = - 9999999
            action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        if self.use_mixer:
            global_state_embedding = self._global_state_encoder(global_state)
            total_q = self._mixer(agent_q_act, global_state_embedding).reshape(T, B)
        else:
            total_q = agent_q_act.reshape(T, B, A).sum(-1)
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


class CollaQMultiHeadAttention(nn.Module):

    def __init__(self, n_head: int, d_model_q: int, d_model_v: int, d_k: int, d_v: int, d_out: int,
                 dropout: float = 0.):
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

    def __init__(self, q_dim: int, v_dim: int, self_feature_range: List[int], ally_feature_range: List[int],
                 attention_dim: int):
        super(CollaQSMACAttentionModule, self).__init__()
        self.self_feature_range = self_feature_range
        self.ally_feature_range = ally_feature_range
        self.attention_layer = CollaQMultiHeadAttention(1, q_dim,
                                                        v_dim, attention_dim,
                                                        attention_dim, attention_dim)

    def _cut_obs(self, obs: torch.Tensor):
        # obs shape = (T, B, A, obs_dim)
        self_features = obs[:, :, :, self.self_feature_range[0]:self.self_feature_range[1]]
        ally_features = obs[:, :, :, self.ally_feature_range[0]:self.ally_feature_range[1]]
        return self_features, ally_features

    def forward(self, inputs: torch.Tensor):
        # obs shape = (T, B ,A, obs_dim)
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


class CollaQ(nn.Module):

    def __init__(
            self,
            agent_num: int,
            obs_dim: int,
            obs_alone_dim: int,
            global_obs_dim: int,
            action_dim: int,
            embedding_dim: int,
            enable_attention: bool = False,
            self_feature_range: Union[List[int], None] = None,
            ally_feature_range: Union[List[int], None] = None,
            attention_dim: int = 32,
            use_mixer: bool = True
    ) -> None:
        super(CollaQ, self).__init__()
        self.enable_attention = enable_attention
        self.attention_dim = attention_dim
        self._act = nn.ReLU()
        self.use_mixer = use_mixer
        if not self.enable_attention:
            self._q_network = FCRDiscreteNet(obs_dim, action_dim, embedding_dim)
        else:
            # TODO set the attention layer here beautifully
            self._self_attention = CollaQSMACAttentionModule(
                self_feature_range[1] - self_feature_range[0],
                (ally_feature_range[1] - ally_feature_range[0]) // (agent_num - 1), self_feature_range,
                ally_feature_range, attention_dim
            )
            # TODO get the obs_dim_after_attention here beautifully
            obs_dim_after_attention = self._self_attention(
                # torch.randn(
                #     1, 1, (ally_feature_range[1] - ally_feature_range[0]) //
                #           ((self_feature_range[1] - self_feature_range[0])*2) + 1, obs_dim
                # )
                torch.randn(
                    1, 1, agent_num, obs_dim
                )
            ).shape[-1]
            self._q_network = FCRDiscreteNet(obs_dim_after_attention, action_dim, embedding_dim)
        self._q_alone_network = FCRDiscreteNet(obs_alone_dim, action_dim, embedding_dim)
        if self.use_mixer:
            self._mixer = Mixer(agent_num, embedding_dim)
            global_obs_dim = squeeze(global_obs_dim)
            self._global_state_encoder = self._setup_global_encoder(global_obs_dim, embedding_dim)

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
                A is agent_num, N is obs_dim
            - global_state (:obj:`torch.Tensor`): :math:`(T, B, M)`, where M is global_obs_dim
            - prev_state (:obj:`list`): math:`(B, A)`, a list of length B, and each element is a list of length A
            - action (:obj:`torch.Tensor`): :math:`(T, B, A)`
            - total_q (:obj:`torch.Tensor`): :math:`(T, B)`
            - agent_q (:obj:`torch.Tensor`): :math:`(T, B, A, P)`, where P is action_dim
            - next_state (:obj:`list`): math:`(B, A)`, a list of length B, and each element is a list of length A
        """
        agent_state, agent_alone_state, agent_alone_padding_state, global_state, prev_state = data['obs'][
                                                                                                  'agent_state'], \
                                                                                              data['obs'][
                                                                                                  'agent_alone_state'], \
                                                                                              data['obs'][
                                                                                                  'agent_alone_padding_state'], \
                                                                                              data['obs'][
                                                                                                  'global_state'], data[
                                                                                                  'prev_state']

        # TODO find a better way to implement agent_along_padding_state

        action = data.get('action', None)
        if single_step:
            agent_state, agent_alone_state, agent_alone_padding_state, global_state = agent_state.unsqueeze(
                0
            ), agent_alone_state.unsqueeze(0), agent_alone_padding_state.unsqueeze(0), global_state.unsqueeze(0)
        T, B, A = agent_state.shape[:3]

        if self.enable_attention:
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
            agent_q[action_mask == 0.0] = - 9999999
            action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        if self.use_mixer:
            global_state_embedding = self._global_state_encoder(global_state)
            total_q = self._mixer(agent_q_act, global_state_embedding).reshape(T, B)
        else:
            total_q = agent_q_act.reshape(T, B, A).sum(-1)
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

    def _setup_global_encoder(self, global_obs_dim: int, embedding_dim: int) -> torch.nn.Module:
        layers = []
        layer_num = 1
        layers.append(nn.Linear(global_obs_dim, embedding_dim))
        layers.append(self._act)
        for _ in range(layer_num):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(self._act)
        return nn.Sequential(*layers)


register_model('qmix', QMix)
register_model('collaq', CollaQ)
