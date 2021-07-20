from typing import Union, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ding.utils import squeeze, MODEL_REGISTRY, SequenceType
from ding.torch_utils import MLP
from ..common import RegressionHead


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
            init method of the communication network

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
        initiator_threshold: float = 0.5,
        attention_embedding_size: int = 64,
        actor_1_embedding_size: Union[int, None] = None,
        actor_2_embedding_size: Union[int, None] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
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

        # Actor Net(I)
        self.actor_1 = MLP(
            self._obs_shape,
            actor_1_embedding_size,
            self._thought_size,
            layer_num=2,
            activation=activation,
            norm_type=norm_type
        )

        # Actor Net(II)
        self.actor_2 = nn.Sequential(
            nn.Linear(self._thought_size * 2, actor_2_embedding_size), activation,
            RegressionHead(
                actor_2_embedding_size, self._act_shape, 2, final_tanh=True, activation=activation, norm_type=norm_type
            )
        )

        # Communication
        if self._communication:
            self.attention = ATOCAttentionUnit(self._thought_size, attention_embedding_size)
            self.comm_net = ATOCCommunicationNet(self._thought_size)

    def forward(self, obs: torch.Tensor) -> Dict:
        r"""
        Overview:
            the forward method of actor network, take the input obs, and calculate the corresponding action, group, \
                initiator_prob, thoughts, etc...

        Arguments:
            - obs (:obj:`Dict`): the input obs containing the observation
        Returns:
            - ret (:obj:`Dict`): the returned output, including action, group, initiator_prob, is_initiator, \
                new_thoughts and old_thoughts
        ReturnsKeys:
            - necessary: ``action``
            - optional: ``group``, ``initiator_prob``, ``is_initiator``, ``new_thoughts``, ``old_thoughts``
        """
        assert len(obs.shape) == 3
        self._cur_batch_size = obs.shape[0]
        B, A, N = obs.shape
        assert A == self._n_agent
        assert N == self._obs_shape

        current_thoughts = self.actor_1(obs)  # B, A, thought size

        if self._communication:
            old_thoughts = current_thoughts.clone().detach()
            init_prob, is_initiator, group = self._get_initiate_group(old_thoughts)

            new_thoughts = self._get_new_thoughts(current_thoughts, group, is_initiator)
        else:
            new_thoughts = current_thoughts
        action = self.actor_2(torch.cat([current_thoughts, new_thoughts], dim=-1))['pred']

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
        init_prob = self.attention(current_thoughts)  # B, A
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

    def _get_new_thoughts(self, current_thoughts, group, is_initiator):
        """
        Shapes:
            - current_thoughts (:obj:`torch.Tensor`): :math:`(B, A, M)`, where M is thought size
            - group: (:obj:`torch.Tensor`): :math:`(B, A, A)`
            - is_initiator (:obj:`torch.Tensor`): :math:`(B, A)`
        """
        if not self._communication:
            raise NotImplementedError
        B, A = current_thoughts.shape[:2]
        new_thoughts = current_thoughts.detach().clone()
        if len(torch.nonzero(is_initiator)) == 0:
            return new_thoughts

        # TODO(nyz) execute communication serially for shared agent in different group
        thoughts_to_commute = []
        for b in range(B):
            for i in range(A):
                if is_initiator[b][i]:
                    tmp = []
                    for j in range(A):
                        if group[b][i][j]:
                            tmp.append(new_thoughts[b][j])
                    thoughts_to_commute.append(torch.stack(tmp, dim=0))
        thoughts_to_commute = torch.stack(thoughts_to_commute, dim=1)  # agent_per_group, B_, N
        integrated_thoughts = self.comm_net(thoughts_to_commute)
        b_count = 0
        for b in range(B):
            for i in range(A):
                if is_initiator[b][i]:
                    j_count = 0
                    for j in range(A):
                        if group[b][i][j]:
                            new_thoughts[b][j] = integrated_thoughts[j_count][b_count]
                            j_count += 1
                    b_count += 1
        return new_thoughts


@MODEL_REGISTRY.register('atoc')
class ATOC(nn.Module):
    r"""
    Overview:
        The QAC network of ATOC, a kind of extension of DDPG for MARL.

    Interface:
        __init__, forward, compute_critic, compute_actor, optimize_actor_attention
    """
    mode = ['compute_actor', 'compute_critic', 'optimize_actor_attention']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            thought_size: int,
            n_agent: int,
            communication: bool = True,
            agent_per_group: int = 2,
            actor_1_embedding_size: Union[int, None] = None,
            actor_2_embedding_size: Union[int, None] = None,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 2,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
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
        """
        super(ATOC, self).__init__()
        self._communication = communication

        self.actor = ATOCActorNet(
            obs_shape,
            thought_size,
            action_shape,
            n_agent,
            communication,
            agent_per_group,
            actor_1_embedding_size=actor_1_embedding_size,
            actor_2_embedding_size=actor_2_embedding_size
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_shape + action_shape, critic_head_hidden_size), activation,
            RegressionHead(
                critic_head_hidden_size,
                1,
                critic_head_layer_num,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type,
            )
        )

    def _compute_delta_q(self, obs: torch.Tensor, actor_outputs: Dict) -> torch.Tensor:
        r"""
        Overview:
            calculate the delta_q according to obs and actor_outputs

        Arguments:
            - obs (:obj:`torch.Tensor`): the observations
            - actor_outputs (:obj:`dict`): the output of actors
            - delta_q (:obj:`Dict`): the calculated delta_q
        ArgumentsKeys:
            - necessary: ``new_thoughts``, ``old_thoughts``, ``group``, ``is_initiator``
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
                        before_update_action_j = self.actor.actor_2(
                            torch.cat([old_thoughts[b][j], old_thoughts[b][j]], dim=-1)
                        )
                        after_update_action_j = self.actor.actor_2(
                            torch.cat([old_thoughts[b][j], new_thoughts[b][j]], dim=-1)
                        )
                        before_update_input = torch.cat([obs[b][j], before_update_action_j['pred']], dim=-1)
                        before_update_Q_j = self.critic(before_update_input)['pred']
                        after_update_input = torch.cat([obs[b][j], after_update_action_j['pred']], dim=-1)
                        after_update_Q_j = self.critic(after_update_input)['pred']
                        q_group.append(before_update_Q_j)
                        actual_q_group.append(after_update_Q_j)
                    q_group = torch.stack(q_group)
                    actual_q_group = torch.stack(actual_q_group)
                    curr_delta_q[b][i] = actual_q_group.mean() - q_group.mean()
        return curr_delta_q

    def compute_actor(self, obs: torch.Tensor, get_delta_q: bool = False) -> Dict[str, torch.Tensor]:
        r'''
        Overview:
            compute the action according to inputs, call the _compute_delta_q function to compute delta_q

        Arguments:
            - obs (:obj:`torch.Tensor`): observation
            - get_delta_q (:obj:`bool`) : whether need to get delta_q
        Returns:
            - outputs (:obj:`Dict`): the output of actor network and delta_q
        ReturnsKeys:
            - necessary: ``action``
            - optional: ``group``, ``initiator_prob``, ``is_initiator``, ``new_thoughts``, ``old_thoughts``, ``delta_q``

        '''
        outputs = self.actor(obs)
        if get_delta_q and self._communication:
            delta_q = self._compute_delta_q(obs, outputs)
            outputs['delta_q'] = delta_q
        return outputs

    def compute_critic(self, inputs: Dict) -> Dict:
        """
        ArgumentsKeys:
            - necessary: ``obs``, ``action``
        ReturnsKeys:
            - necessary: ``q_value``
        """
        obs, action = inputs['obs'], inputs['action']
        if len(action.shape) == 2:  # (B, A) -> (B, A, 1)
            action = action.unsqueeze(2)
        x = torch.cat([obs, action], dim=-1)
        x = self.critic(x)['pred']
        return {'q_value': x}

    def optimize_actor_attention(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
        Overview:
            return the actor attention loss

        Arguments:
            - inputs (:obj:`Dict`): the inputs contain the delta_q, initiator_prob, and is_initiator
        Returns
            - loss (:obj:`Dict`): the loss of actor attention unit
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
            actor_attention_loss = torch.FloatTensor([-0.0]).to(delta_q.device)
            actor_attention_loss.requires_grad = True
        else:
            actor_attention_loss = -delta_q * \
                torch.log(init_prob) - (1 - delta_q) * torch.log(1 - init_prob)
        return {'loss': actor_attention_loss.mean()}

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str, **kwargs) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs, **kwargs)
