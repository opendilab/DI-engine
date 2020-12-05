from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

from nervex.torch_utils import get_lstm
from nervex.utils import squeeze


class ATOCAttentionUnit(nn.Module):
    r"""

    .. note::

        "ATOC paper: We use two-layer MLP to implement the attention unit but it is also can be realized by RNN."

    We now implement it as two-layer MLP same as the original paper
    """

    def __init__(self, thought_dim: int, embedding_dim: int):
        super(ATOCAttentionUnit, self).__init__()
        self._thought_dim = thought_dim
        self._hidden_dim = embedding_dim
        self._output_dim = 1
        self._act1 = nn.ReLU()
        self._fc1 = nn.Linear(self._thought_dim, self._hidden_dim)
        self._fc2 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._fc3 = nn.Linear(self._hidden_dim, self._output_dim)
        self._act2 = nn.Sigmoid()

    def forward(self, data: Union[Dict, torch.Tensor]):
        x = data
        if isinstance(data, Dict):
            x = data['thought']
        x = self._fc1(x)
        x = self._act1(x)
        x = self._fc2(x)
        x = self._act1(x)
        x = self._fc3(x)
        x = self._act2(x)
        # return {'initiator': x}
        return x


class ATOCCommunicationNet(nn.Module):
    r"""
    Overview:
        bi-direction LSTM
    """

    def __init__(self, thought_dim: int):
        r"""
        communication hidden size should be half of the actor_hidden_size because of the bi-direction lstm
        """
        super(ATOCCommunicationNet, self).__init__()
        assert thought_dim % 2 == 0
        self._thought_dim = thought_dim
        self._comm_hidden_size = thought_dim / 2

        self._bi_lstm = nn.LSTM(self._thought_dim, self._comm_hidden_size, bidirectional=True)

    def forward(self, data: Union[Dict, torch.Tensor]):
        r"""
        shape:
            data['thoughts']: :math:`(T, B, N)`, T is the num of thoughts to integrate, B is batch_size and N is thought dim
        """
        x = data
        if isinstance(data, Dict):
            x = data['thoughts']
        out, _ = self._bi_lstm(x)
        # return {'thoughts': out}
        return out


class ATOCActorNet(nn.Module):
    r"""
    Overview:
        the overall integrated ATOC actor network

        .. note::
            "ATOC paper: The neural networks use ReLU and batch normalization for some hidden layers."
    """

    # TODO not consider batch yet...
    def __init__(
        self,
        obs_dim: Union[Tuple, int],
        thought_dim: int,
        action_dim: int,
        n_agent: int,
        m_group: int,
        T_initiate: int,
        attention_embedding_dim: int = 64,
        actor_1_embedding_dim: Union[int, None] = None,
        actor_2_embedding_dim: Union[int, None] = None,
    ):
        super(ATOCActorNet, self).__init__()
        # now only support obs_dim of shape (O_dim, )
        self._obs_dim = squeeze(obs_dim)
        self._thought_dim = thought_dim
        self._act_dim = action_dim
        self._n_agent = n_agent
        if not actor_1_embedding_dim:
            actor_1_embedding_dim = self._thought_dim
        if not actor_2_embedding_dim:
            actor_2_embedding_dim = self._thought_dim

        #The actor network has four hidden layers, the second layer is the thought (128 units),
        # and the output layer is the tanh activation function

        #Actor Net(I)
        actor_1_layer = []

        actor_1_layer.append(nn.Linear(self._obs_dim, actor_1_embedding_dim))
        actor_1_layer.append(nn.LayerNorm(actor_1_embedding_dim))
        actor_1_layer.append(nn.ReLU())
        actor_1_layer.append(nn.Linear(actor_1_embedding_dim, self._thought_dim))
        actor_1_layer.append(nn.LayerNorm(actor_1_embedding_dim))

        self._actor_1 = nn.Sequential(*actor_1_layer)

        #Actor Net(II)
        actor_2_layer = []
        actor_2_layer.append(nn.ReLU())

        # note that there might not be integrated thought for some agent, so we should think of a way to
        # update the thoughts
        actor_2_layer.append(nn.Linear(self._thought_dim, actor_2_embedding_dim))
        # actor_2_layer.append(nn.Linear(self._thought_dim * 2, actor_2_embedding_dim))

        # not sure if we should layer norm here
        actor_2_layer.append(nn.LayerNorm(actor_2_embedding_dim))
        actor_2_layer.append(nn.Linear(actor_2_embedding_dim, self._act_dim))
        actor_2_layer.append(nn.LayerNorm(actor_2_embedding_dim))
        actor_2_layer.append(nn.Tanh())

        self._actor_2 = nn.Sequential(*actor_2_layer)
        self._critic = None
        self._attention = ATOCAttentionUnit(self._thought_dim, attention_embedding_dim)
        self._channel = ATOCCommunicationNet(self._thought_dim)
        # C is the communication group
        # TODO consider batch shape
        self._C = torch.zeros(self._n_agent, self._n_agent).bool()

        # TODO consider batch shape
        self._is_initiator = torch.zeros(self._n_agent).bool()

        self._T = T_initiate

        self._step_count = 0

    def forward(self, data: Dict):
        obs = data['obs']
        self._current_obs = obs

        self._current_thougths = self._actor_1(obs)

        if self._step_count % self._T == 0:
            self._get_initiate_group()

        self._updata_current_thougts()

        acts = self._actor_2(self._current_thougths)

        return {'action': acts, 'groups': self._C}

    # TODO
    def _get_initiate_group(self):
        init_prob = self._attention(self._current_thougths)
        # TODO consider batch shape
        self._is_initiator = (init_prob > 0.5).bool()

        # TODO
        # calculate relative position

        # TODO
        # get observable field of the initiators

        # TODO
        # "considers the agents in its observable field"

        # TODO
        # "initiator first chooses collaborators from agents who have not been selected,
        #  then from agents selected by other initiators,
        #  finally from other initiators"

        # TODO
        # "all based on proximity"

    # TODO
    def _updata_current_thougts(self):

        for i in range(self._n_agent):
            # TODO consider batch shape
            if self._is_initiator[i]:
                thoughts_to_commute = []
                for j in range(self._n_agent):
                    # TODO consider batch shape
                    if self._C[i][j]:
                        # TODO consider batch shape
                        thoughts_to_commute.append(self._current_thougths[j])
                thoughts_to_commute = torch.stack(thoughts_to_commute)
                integrated_thoughts = self._channel(thoughts_to_commute)
                j_count = 0
                for j in range(self._n_agent):
                    # TODO consider batch shape
                    if self._C[i][j]:
                        # TODO consider batch shape
                        self._current_thougths[j] = self._merge_2_thoughts(
                            self._current_thougths[j], integrated_thoughts[j_count]
                        )
                        j_count += 1

    #TODO
    def _merge_2_thoughts(self, thought1, thought2):
        return thought2

    # TODO
    # TO DISCUSS
    def _get_attention_loss(self):
        return torch.zeros(1).mean()
