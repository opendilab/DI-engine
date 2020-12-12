from typing import Union, List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

from nervex.torch_utils import get_lstm
from nervex.utils import squeeze
from copy import deepcopy
from nervex.model.common_arch import QActorCriticBase
import queue


class ATOCAttentionUnit(nn.Module):
    r"""

    .. note::

        "ATOC paper: We use two-layer MLP to implement the attention unit but it is also can be realized by RNN."

    We now implement it as two-layer MLP same as the original paper
    """

    def __init__(self, thought_dim: int, embedding_dim: int) -> None:
        super(ATOCAttentionUnit, self).__init__()
        self._thought_dim = thought_dim
        self._hidden_dim = embedding_dim
        self._output_dim = 1
        self._act1 = nn.ReLU()
        self._fc1 = nn.Linear(self._thought_dim, self._hidden_dim)
        self._fc2 = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._fc3 = nn.Linear(self._hidden_dim, self._output_dim)
        self._act2 = nn.Sigmoid()

    def forward(self, data: Union[Dict, torch.Tensor]) -> torch.Tensor:
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
        return x.squeeze(-1)


class ATOCCommunicationNet(nn.Module):
    r"""
    Overview:
        bi-direction LSTM
    """

    def __init__(self, thought_dim: int) -> None:
        r"""
        communication hidden size should be half of the actor_hidden_size because of the bi-direction lstm
        """
        super(ATOCCommunicationNet, self).__init__()
        assert thought_dim % 2 == 0
        self._thought_dim = thought_dim
        self._comm_hidden_size = thought_dim // 2
        self._bi_lstm = nn.LSTM(self._thought_dim, self._comm_hidden_size, bidirectional=True)

    def forward(self, data: Union[Dict, torch.Tensor]):
        r"""
        shape:
            data['thoughts']: :math:`(M, B, N)`, M is the num of thoughts to integrate,\
                B is batch_size and N is thought dim
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

    def __init__(
        self,
        obs_dim: Union[Tuple, int],
        thought_dim: int,
        action_dim: int,
        n_agent: int,
        m_group: int,
        T_initiate: int,
        initiator_threshold: float = 0.4,
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
        self._m_group = m_group
        self._initiator_threshold = initiator_threshold
        if not actor_1_embedding_dim:
            actor_1_embedding_dim = self._thought_dim
        if not actor_2_embedding_dim:
            actor_2_embedding_dim = self._thought_dim

        # The actor network has four hidden layers, the second layer is the thought (128 units),
        # and the output layer is the tanh activation function

        # Actor Net(I)
        actor_1_layer = []

        actor_1_layer.append(nn.Linear(self._obs_dim, actor_1_embedding_dim))
        actor_1_layer.append(nn.LayerNorm(actor_1_embedding_dim))
        actor_1_layer.append(nn.ReLU())
        actor_1_layer.append(nn.Linear(actor_1_embedding_dim, self._thought_dim))
        actor_1_layer.append(nn.LayerNorm(self._thought_dim))

        self._actor_1 = nn.Sequential(*actor_1_layer)

        # Actor Net(II)
        actor_2_layer = []
        actor_2_layer.append(nn.ReLU())

        # note that there might not be integrated thought for some agent, so we should think of a way to
        # update the thoughts
        actor_2_layer.append(nn.Linear(self._thought_dim * 2, actor_2_embedding_dim))
        # actor_2_layer.append(nn.Linear(self._thought_dim * 2, actor_2_embedding_dim))

        # not sure if we should layer norm here
        actor_2_layer.append(nn.LayerNorm(actor_2_embedding_dim))
        actor_2_layer.append(nn.Linear(actor_2_embedding_dim, self._act_dim))
        actor_2_layer.append(nn.LayerNorm(self._act_dim))
        actor_2_layer.append(nn.Tanh())
        self.actor_2 = nn.Sequential(*actor_2_layer)

        # Communication
        self._attention = ATOCAttentionUnit(self._thought_dim, attention_embedding_dim)
        self._comm_net = ATOCCommunicationNet(self._thought_dim)

        self._get_group_freq = T_initiate
        self._step_count = 0

    def forward(self, data: Dict):
        obs = data['obs']
        assert len(obs.shape) == 3
        self._cur_batch_size, n_agent, obs_dim = obs.shape
        B, A, N = obs.shape
        assert A == self._n_agent
        assert N == self._obs_dim

        current_thoughts = self._actor_1(obs)  # B, A, thoughts_dim

        if self._step_count % self._get_group_freq == 0:
            init_prob, is_initiator, group = self._get_initiate_group(current_thoughts)

        old_thoughts = current_thoughts.clone().detach()
        new_thoughts = self._update_current_thoughts(current_thoughts, group, is_initiator)

        action = self.actor_2(torch.cat([current_thoughts, new_thoughts], dim=-1))

        return {
            'action': action,
            'group': group,
            'initiator_prob': init_prob,
            'is_initiator': is_initiator,
            'new_thoughts': new_thoughts,
            'old_thoughts': old_thoughts,
        }

    def _get_initiate_group(self, current_thoughts):
        init_prob = self._attention(current_thoughts)  # B, A
        is_initiator = (init_prob > self._initiator_threshold)
        B, A = init_prob.shape[:2]

        thoughts_pair_dot = current_thoughts.bmm(current_thoughts.transpose(1, 2))
        thoughts_square = thoughts_pair_dot.diagonal(0, 1, 2)
        curr_thought_dists = thoughts_square.unsqueeze(1) - 2 * thoughts_pair_dot + thoughts_square.unsqueeze(2)

        group = torch.zeros(B, A, A).to(init_prob.device)

        # TODO
        # "considers the agents in its observable field"

        # TODO
        # "initiator first chooses collaborators from agents who have not been selected,
        #  then from agents selected by other initiators,
        #  finally from other initiators"

        # TODO
        # "all based on proximity"

        # roughly choose m closest as group
        for b in range(B):
            for i in range(A):
                if is_initiator[b][i]:
                    index_seq = curr_thought_dists[b][i].argsort()
                    index_seq = index_seq[:self._m_group]
                    group[b][i][index_seq] = 1
        return init_prob, is_initiator, group

    def _update_current_thoughts(self, current_thoughts, group, is_initiator):
        """
        Shapes:
            - current_thoughts (:obj:`torch.Tensor`): :math:`(B, A, M)`, where M is thoughts_dim
            - group: (:obj:`torch.Tensor`): :math:`(B, A, A)`
            - is_initiator (:obj:`torch.Tensor`): :math:`(B, A)`
        """
        B, A = current_thoughts.shape[:2]
        new_thoughts = current_thoughts.clone()

        for b in range(B):
            for i in range(A):
                if is_initiator[b][i]:
                    thoughts_to_commute = []
                    for j in range(A):
                        if group[b][i][j]:
                            thoughts_to_commute.append(new_thoughts[b][j])
                    thoughts_to_commute = torch.stack(thoughts_to_commute)
                    integrated_thoughts = self._comm_net(thoughts_to_commute.unsqueeze(1)).squeeze(1)
                    j_count = 0
                    for j in range(A):
                        if group[b][i][j]:
                            new_thoughts[b][j] = integrated_thoughts[j_count]
                            j_count += 1
        return new_thoughts


class ATOCCriticNet(nn.Module):
    r"""
    .. note::

        "ATOC paper:The critic network has two hidden layers with 512 and 256 units respectively."
    """

    # note, the critic take the action as input
    def __init__(self, obs_dim: int, action_dim: int, embedding_dims: List[int] = [128, 64]):
        super(ATOCCriticNet, self).__init__()
        self._obs_dim = obs_dim
        self._act_dim = action_dim
        self._embedding_dims = embedding_dims
        cur_dim = self._obs_dim + self._act_dim
        self._main = nn.ModuleList()
        for dim in embedding_dims:
            self._main.append(nn.Linear(cur_dim, dim))
            self._main.append(nn.LayerNorm(dim))
            self._main.append(nn.ReLU())
            cur_dim = dim
        self._main.append(nn.Linear(cur_dim, 1))

    def forward(self, data: Dict) -> Dict:
        r"""
        shapes:
            data['obs']: :math:`(B, A, obs_dim + act_dim)`
            data['action']: :math:`(B, A)`
        """
        obs = data['obs']
        action = data['action']
        x = torch.cat([obs, action], -1)
        for m in self._main:
            x = m(x)
        data['q_value'] = x.squeeze(-1)
        return data


class ATOCQAC(QActorCriticBase):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            thought_dim: int,
            n_agent: int,
            m_group: int,
            T_initiate: int,
    ) -> None:
        super(ATOCQAC, self).__init__()

        def backward_hook(module, grad_input, grad_output):
            for p in module.parameters():
                p.requires_grad = True

        self._actor = ATOCActorNet(obs_dim, thought_dim, action_dim, n_agent, m_group, T_initiate)
        self._critic = ATOCCriticNet(obs_dim, action_dim)
        self._critic.register_backward_hook(backward_hook)

    def _critic_forward(self, x: Dict[str, torch.Tensor]) -> Union[List[torch.Tensor], torch.Tensor]:
        return self._critic(x)

    def _actor_forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._actor(x)

    def compute_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        if inputs.get('action') is None:
            inputs['action'] = self._actor_forward(inputs)['action']
        q = self._critic_forward(inputs)
        return q

    def compute_action(self, inputs: Dict[str, torch.Tensor], get_delta_q: bool = False) -> Dict[str, torch.Tensor]:
        r'''
        Overview:
            use call the actor_forward function to compute action

            in ATOC, not only the action is computed, but the groups, initiator_prob, thoughts, delta_q, etc
        '''
        outputs = self._actor_forward(inputs)
        if get_delta_q:
            delta_q = self._compute_delta_q(inputs['obs'], outputs)
            outputs['delta_q'] = delta_q
        return outputs

    def optimize_actor(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for p in self._critic.parameters():
            p.requires_grad = False  # will set True when backward_hook called
        for p in self._actor.parameters():
            p.requires_grad = True

        if inputs.get('action') is None:
            inputs['action'] = self._actor_forward(inputs)['action']
        q = self._critic_forward(inputs)

        return q

    def optimize_actor_attention(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO(nyz) multi optimizer
        delta_q = inputs['delta_q'].reshape(-1)
        init_prob = inputs['initiator_prob'].reshape(-1)
        is_init = inputs['is_initiator'].reshape(-1)
        delta_q = delta_q[is_init.nonzero()]
        init_prob = init_prob[is_init.nonzero()]

        actor_attention_loss = -delta_q * torch.log(init_prob) - (1 - delta_q) * torch.log(1 - init_prob)
        return {'actor_attention_loss': actor_attention_loss}

    def _compute_delta_q(self, obs: torch.Tensor, actor_outputs: dict) -> torch.Tensor:
        assert len(obs.shape) == 3
        new_thoughts, old_thoughts, group, is_initiator = actor_outputs['new_thoughts'], actor_outputs[
            'old_thoughts'], actor_outputs['group'], actor_outputs['is_initiator']
        B, A = new_thoughts.shape[:2]
        curr_delta_q = torch.zeros(B, A).to(new_thoughts.device)
        with torch.no_grad():
            for b in range(B):
                for i in range(A):
                    if not is_initiator[b][i]:
                        continue
                    q_group = []
                    actual_q_group = []
                    for j in range(A):
                        if not group[b][i][j]:
                            continue
                        before_update_action_j = self._actor.actor_2(
                            torch.cat([old_thoughts[b][j], old_thoughts[b][j]], dim=-1)
                        )
                        after_update_action_j = self._actor.actor_2(
                            torch.cat([old_thoughts[b][j], new_thoughts[b][j]], dim=-1)
                        )
                        before_update_Q_j = self._critic_forward({
                            'obs': obs[b][j],
                            'action': before_update_action_j
                        })['q_value']
                        after_update_Q_j = self._critic_forward({
                            'obs': obs[b][j],
                            'action': after_update_action_j
                        })['q_value']
                        q_group.append(before_update_Q_j)
                        actual_q_group.append(after_update_Q_j)
                    q_group = torch.stack(q_group)
                    actual_q_group = torch.stack(actual_q_group)
                    curr_delta_q[b][i] = actual_q_group.mean() - q_group.mean()
        return curr_delta_q

    def forward(self, inputs, mode=None, **kwargs):
        assert (
            mode in [
                'optimize_actor', 'optimize_actor_attention', 'compute_q', 'compute_action', 'compute_action_q',
                'compute_delta_q'
            ]
        ), mode
        f = getattr(self, mode)
        return f(inputs, **kwargs)
