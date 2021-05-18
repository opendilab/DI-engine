from typing import Union, List, Dict, Optional, Tuple, Callable
from copy import deepcopy

import queue
import torch
import torch.nn as nn

from nervex.utils import squeeze, MODEL_REGISTRY
from ..common import ActorCriticBase


class ATOCAttentionUnit(nn.Module):
    r"""
    Overview:
        the attention unit of the atoc network. We now implement it as two-layer MLP, same as the original paper

    Interface:
        __init__, forward

    .. note::

        "ATOC paper: We use two-layer MLP to implement the attention unit but it is also can be realized by RNN."

    """

    def __init__(self, thought_size: int, embedding_size: int) -> None:
        r"""
        Overview:
            init the attention unit according to the size of input args

        Arguments:
            - thought_size (:obj:`int`): the size of input thought
            - embedding_size (:obj:`int`): the size of hidden layers
        """
        super(ATOCAttentionUnit, self).__init__()
        self._thought_size = thought_size
        self._hidden_size = embedding_size
        self._output_size = 1
        self._act1 = nn.ReLU()
        self._fc1 = nn.Linear(self._thought_size, self._hidden_size, bias=True)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size, bias=True)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size, bias=True)
        self._act2 = nn.Sigmoid()

    def forward(self, data: Union[Dict, torch.Tensor]) -> torch.Tensor:
        r"""
        Overview:
            forward method take the thought of agents as input and output the prob of these agent\
                being initiator

        Arguments:
            - x (:obj:`Union[Dict, torch.Tensor`): the input tensor or dict contain the thoughts tensor
            - ret (:obj:`torch.Tensor`): the output initiator prob

        """
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
        atoc commnication net is a bi-direction LSTM, so it can integrate all the thoughts in the group

    Interface:
        __init__, forward
    """

    def __init__(self, thought_size: int) -> None:
        r"""
        Overview:
            init method of the commnunication network

        Arguments:
            - thought_size (:obj:`int`): the size of input thought

        .. note::

            communication hidden size should be half of the actor_hidden_size because of the bi-direction lstm
        """
        super(ATOCCommunicationNet, self).__init__()
        assert thought_size % 2 == 0
        self._thought_size = thought_size
        self._comm_hidden_size = thought_size // 2
        self._bi_lstm = nn.LSTM(self._thought_size, self._comm_hidden_size, bidirectional=True)

    def forward(self, data: Union[Dict, torch.Tensor]):
        r"""
        Overview:
            the forward method that integrate thoughts
        Arguments:
            - x (:obj:`Union[Dict, torch.Tensor`): the input tensor or dict contain the thoughts tensor
            - out (:obj:`torch.Tensor`): the integrated thoughts
        Shapes:
            - data['thoughts']: :math:`(M, B, N)`, M is the num of thoughts to integrate,\
                B is batch_size and N is thought size
        """
        self._bi_lstm.flatten_parameters()
        x = data
        if isinstance(data, Dict):
            x = data['thoughts']
        out, _ = self._bi_lstm(x)
        # return {'thoughts': out}
        return out


class ATOCActorNet(nn.Module):
    r"""
    Overview:
        the overall ATOC actor network

    Interface:
        __init__, forward

        .. note::
            "ATOC paper: The neural networks use ReLU and batch normalization for some hidden layers."
    """

    def __init__(
        self,
        obs_shape: Union[Tuple, int],
        thought_size: int,
        action_shape: int,
        n_agent: int,
        communication: bool = True,
        agent_per_group: int = 2,
        group_init_freq: int = 5,
        initiator_threshold: float = 0.5,
        attention_embedding_size: int = 64,
        actor_1_embedding_size: Union[int, None] = None,
        actor_2_embedding_size: Union[int, None] = None,
    ):
        r"""
        Overview:
            the init method of atoc actor network

        Arguments:
            - obs_shape(:obj:`Union[Tuple, int]`): the observation size
            - thought_size (:obj:`int`): the size of thoughts
            - action_shape (:obj:`int`): the action size
            - n_agent (:obj:`int`): the num of agents
            - agent_per_group (:obj:`int`): the num of agent in each group
            - group_init_freq (:obj:`int`): the time between group initiate
            - initiator_threshold (:obj:`float`): the threshold of becoming an initiator, default set to 0.5
            - attention_embedding_size (obj:`int`): the embedding size of attention unit, default set to 64
            - actor_1_embedding_size (:obj:`Union[int, None]`): the size of embedding size of actor network part1, \
                if None, then default set to thought size
            - actor_2_embedding_size (:obj:`Union[int, None]`): the size of embedding size of actor network part2, \
                if None, then default set to thought size
        """
        super(ATOCActorNet, self).__init__()
        # now only support obs_shape of shape (O_dim, )
        self._obs_shape = squeeze(obs_shape)
        self._thought_size = thought_size
        self._act_shape = action_shape
        self._n_agent = n_agent
        self._communication = communication
        self._agent_per_group = agent_per_group
        self._initiator_threshold = initiator_threshold
        if not actor_1_embedding_size:
            actor_1_embedding_size = self._thought_size
        if not actor_2_embedding_size:
            actor_2_embedding_size = self._thought_size

        # The actor network has four hidden layers, the second layer is the thought (128 units),
        # and the output layer is the tanh activation function

        # Actor Net(I)
        actor_1_layer = []

        actor_1_layer.append(nn.Linear(self._obs_shape, actor_1_embedding_size))
        actor_1_layer.append(nn.LayerNorm(actor_1_embedding_size))
        actor_1_layer.append(nn.ReLU())
        actor_1_layer.append(nn.Linear(actor_1_embedding_size, self._thought_size))
        # actor_1_layer.append(nn.LayerNorm(self._thought_size))

        self._actor_1 = nn.Sequential(*actor_1_layer)

        # Actor Net(II)
        actor_2_layer = []
        actor_2_layer.append(nn.ReLU())

        # note that there might not be integrated thought for some agent, so we should think of a way to
        # update the thoughts
        actor_2_layer.append(nn.Linear(self._thought_size * 2, actor_2_embedding_size))
        # actor_2_layer.append(nn.Linear(self._thought_size * 2, actor_2_embedding_size))

        # not sure if we should layer norm here
        # actor_2_layer.append(nn.LayerNorm(actor_2_embedding_size))
        actor_2_layer.append(nn.Linear(actor_2_embedding_size, self._act_shape, bias=False))
        # actor_2_layer.append(nn.LayerNorm(self._act_shape))
        actor_2_layer.append(nn.Tanh())
        self.actor_2 = nn.Sequential(*actor_2_layer)

        # Communication
        if self._communication:
            self._attention = ATOCAttentionUnit(self._thought_size, attention_embedding_size)
            self._comm_net = ATOCCommunicationNet(self._thought_size)

        self._get_group_freq = group_init_freq
        self._step_count = 0

    def forward(self, obs: Dict):
        r"""
        Overview:
            the forward method of actor network, take the input obs, and calculate the corresponding action, group, \
                initiator_prob, thoughts, etc...

        Arguments:
            - obs (:obj:`Dict`): the input obs containing the observation
            - ret (:obj:`Dict`): the returned output, including action, group, initiator_prob, is_initiator, \
                new_thoughts and old_thoughts
        """
        assert len(obs.shape) == 3
        self._cur_batch_size, n_agent, obs_shape = obs.shape
        B, A, N = obs.shape
        assert A == self._n_agent
        assert N == self._obs_shape

        current_thoughts = self._actor_1(obs)  # B, A, thought size

        if self._communication:
            if self._step_count % self._get_group_freq == 0:
                init_prob, is_initiator, group = self._get_initiate_group(current_thoughts)

            old_thoughts = current_thoughts.clone().detach()
            new_thoughts = self._update_current_thoughts(current_thoughts, group, is_initiator)
        else:
            new_thoughts = current_thoughts
        action = self.actor_2(torch.cat([current_thoughts, new_thoughts], dim=-1))

        if self._communication:
            return {
                'action': action,
                'group': group,
                'initiator_prob': init_prob,
                'is_initiator': is_initiator,
                'new_thoughts': new_thoughts,
                'old_thoughts': old_thoughts,
            }
        else:
            return {'action': action}

    def _get_initiate_group(self, current_thoughts):
        if not self._communication:
            raise NotImplementedError
        init_prob = self._attention(current_thoughts)  # B, A
        is_initiator = (init_prob > self._initiator_threshold)
        B, A = init_prob.shape[:2]

        thoughts_pair_dot = current_thoughts.bmm(current_thoughts.transpose(1, 2))
        thoughts_square = thoughts_pair_dot.diagonal(0, 1, 2)
        curr_thought_dists = thoughts_square.unsqueeze(1) - 2 * thoughts_pair_dot + thoughts_square.unsqueeze(2)

        group = torch.zeros(B, A, A).to(init_prob.device)

        # "considers the agents in its observable field"
        # "initiator first chooses collaborators from agents who have not been selected,
        #  then from agents selected by other initiators,
        #  finally from other initiators"
        # "all based on proximity"

        # roughly choose m closest as group
        for b in range(B):
            for i in range(A):
                if is_initiator[b][i]:
                    index_seq = curr_thought_dists[b][i].argsort()
                    index_seq = index_seq[:self._agent_per_group]
                    group[b][i][index_seq] = 1
        return init_prob, is_initiator, group

    def _update_current_thoughts(self, current_thoughts, group, is_initiator):
        """
        Shapes:
             current_thoughts (:obj:`torch.Tensor`): :math:`(B, A, M)`, where M is thought size
            - group: (:obj:`torch.Tensor`): :math:`(B, A, A)`
            - is_initiator (:obj:`torch.Tensor`): :math:`(B, A)`
        """
        if not self._communication:
            raise NotImplementedError
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
    Overview:
        the critic part network of atoc

    Interface:
        __init__, forward

    .. note::

        "ATOC paper:The critic network has two hidden layers with 512 and 256 units respectively."
    """

    # note, the critic take the action as input
    def __init__(self, obs_shape: int, action_shape: int, embedding_sizes: List[int] = [128, 64]):
        r"""
        Overview:
            the init method of atoc critic net work

        Arguments:
            - obs_shape(:obj:`Union[Tuple, int]`): the observation size
            - action_shape (:obj:`int`): the action size
            - embedding_sizes (:obj:`list` of :obj:`int`): the hidden layer's size
        """
        super(ATOCCriticNet, self).__init__()
        self._obs_shape = obs_shape
        self._act_shape = action_shape
        self._embedding_sizes = embedding_sizes
        cur_size = self._obs_shape + self._act_shape
        self._main = nn.ModuleList()
        for size in embedding_sizes:
            self._main.append(nn.Linear(cur_size, size))
            # self._main.append(nn.LayerNorm(size))
            self._main.append(nn.ReLU())
            cur_size = size
        self._main.append(nn.Linear(cur_size, 1))

    def forward(self, data: Dict) -> Dict:
        r"""
        Overview:
            take the input obs and action, calculate the corresponding q_value

        Arguments:
            - data (:obj:`Dict`): the input data containing obs and action
            - data['q_value'] : the calculated q_value is added to the input data dict
        """
        obs = data['obs']
        action = data['action']
        x = torch.cat([obs, action], -1)
        for m in self._main:
            x = m(x)
        x = x.squeeze(-1)
        return {'q_value': x}


@MODEL_REGISTRY.register('atoc')
class ATOCQAC(ActorCriticBase):
    r"""
    Overview:
        the QAC network of atoc

    Interface:
        __init__, forward, compute_q, compute_action, optimize_actor, optimize_actor_attention
    """

    def __init__(
            self,
            obs_shape: int,
            action_shape: int,
            thought_size: int,
            n_agent: int,
            communication: bool = True,
            agent_per_group: int = 2,
            group_init_freq: int = 5,
            actor_1_embedding_size: Union[int, None] = None,
            actor_2_embedding_size: Union[int, None] = None,
            critic_embedding_sizes: List[int] = [128, 64],
    ) -> None:
        r"""
        Overview:
            init the atoc QAC network

        Arguments:
            - obs_shape(:obj:`Union[Tuple, int]`): the observation space shape
            - thought_size (:obj:`int`): the size of thoughts
            - action_shape (:obj:`int`): the action space shape
            - n_agent (:obj:`int`): the num of agents
            - agent_per_group (:obj:`int`): the num of agent in each group
            - group_init_freq (:obj:`int`): the time between group initiate
        """
        super(ATOCQAC, self).__init__()
        self._communication = communication

        if self._communication:
            self.modes.append('optimize_actor_attention')

        self._actor = ATOCActorNet(
            obs_shape,
            thought_size,
            action_shape,
            n_agent,
            communication,
            agent_per_group,
            group_init_freq,
            actor_1_embedding_size=actor_1_embedding_size,
            actor_2_embedding_size=actor_2_embedding_size
        )
        self._critic = ATOCCriticNet(obs_shape, action_shape, embedding_sizes=critic_embedding_sizes)

    def _compute_delta_q(self, obs: torch.Tensor, actor_outputs: dict) -> torch.Tensor:
        r"""
        Overview:
            calculate the delta_q according to obs and actor_outputs

        Arguments:
            - obs (:obj:`torch.Tensor`): the observations
            - actor_outputs (:obj:`dict`): the output of actors
            - delta_q (:obj:`Dict`): the calculated delta_q
        """
        if not self._communication:
            raise NotImplementedError
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
                        before_update_Q_j = self._critic({
                            'obs': obs[b][j],
                            'action': before_update_action_j
                        })['q_value']
                        after_update_Q_j = self._critic({'obs': obs[b][j], 'action': after_update_action_j})['q_value']
                        q_group.append(before_update_Q_j)
                        actual_q_group.append(after_update_Q_j)
                    q_group = torch.stack(q_group)
                    actual_q_group = torch.stack(actual_q_group)
                    curr_delta_q[b][i] = actual_q_group.mean() - q_group.mean()
        return curr_delta_q

    def compute_critic(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        r"""
        Overview:
            compute the q value of the given obs (or obs and action)

        Arguments:
            - inputs (:obj:`Dict`): the inputs contain obs and action, if action is None then the network will \
                calculate the action according to the observation
            - q (:obj:`Dict`): the output of ciritic network
        """
        if inputs.get('action') is None:
            inputs['action'] = self._actor(inputs)['action']
        q = self._critic(inputs)
        return q

    def compute_actor(self, inputs: Dict[str, torch.Tensor], get_delta_q: bool = True) -> Dict[str, torch.Tensor]:
        r'''
        Overview:
            compute the action according to inputs, call the _compute_delta_q function to compute delta_q

        Arguments:
            - inputs (:obj:`Dict`): the inputs containing the observation
            - get_delta_q (:obj:`bool`) : whether need to get delta_q
            - outputs (:obj:`Dict`): the output of actor network and delta_q

        .. note::

            in ATOC, not only the action is computed, but the groups, initiator_prob, thoughts, delta_q, etc
        '''
        outputs = self._actor(inputs)
        if get_delta_q and self._communication:
            delta_q = self._compute_delta_q(inputs, outputs)
            outputs['delta_q'] = delta_q
        return outputs

    def optimize_actor_attention(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            return the actor attention loss

        Arguments:
            - inputs (:obj:`Dict`): the inputs contain the delta_q, initiator_prob, and is_initiator
            - actor_attention_loss (:obj:`Dict`): the loss of actor attention unit
        """
        if not self._communication:
            raise NotImplementedError
        delta_q = inputs['delta_q'].reshape(-1)
        init_prob = inputs['initiator_prob'].reshape(-1)
        is_init = inputs['is_initiator'].reshape(-1)
        delta_q = delta_q[is_init.nonzero()]
        init_prob = init_prob[is_init.nonzero()]
        init_prob = 0.9 * init_prob + 0.05

        # judge to avoid nan
        if init_prob.shape == (0, 1):
            actor_attention_loss = torch.Tensor([-0.0])
            actor_attention_loss.requires_grad = True
        else:
            actor_attention_loss = -delta_q * \
                torch.log(init_prob) - (1 - delta_q) * torch.log(1 - init_prob)
        return {'actor_attention_loss': actor_attention_loss}
