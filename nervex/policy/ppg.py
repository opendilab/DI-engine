from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nervex.utils import squeeze
from nervex.torch_utils import Adam
from nervex.rl_utils import ppo_data, ppo_error, Adder, ppg_data, ppg_joint_error, value_error, ppg_aux_data
from nervex.model import FCValueAC, ConvValueAC
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy
from ..model.common import FCEncoder


class ExperienceDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return list(self.data.values())[0].shape[0]

    def __getitem__(self, ind):
        data = {}
        for key in self.data.keys():
            data[key] = self.data[key][ind]
        return data


def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class ValueNet(nn.Module):

    def __init__(
            self,
            obs_dim: tuple,
            embedding_dim: int,
            head_hidden_dim: int = 128,
    ) -> None:
        super(ValueNet, self).__init__()
        self._act = nn.ReLU()
        self._obs_dim = squeeze(obs_dim)
        self._embedding_dim = embedding_dim
        self._encoder = self._setup_encoder()
        self._head_layer_num = 2

        # critic head
        input_dim = embedding_dim
        layers = []
        for _ in range(self._head_layer_num):
            layers.append(nn.Linear(input_dim, head_hidden_dim))
            layers.append(self._act)
            input_dim = head_hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self._critic = nn.Sequential(*layers)

    def _setup_encoder(self) -> torch.nn.Module:
        r"""
        Overview:
            Setup an ``ConvEncoder`` to encode 2-dim observation
        Returns:
            - encoder (:obj:`torch.nn.Module`): ``ConvEncoder``
        """
        return FCEncoder(self._obs_dim, self._embedding_dim)

    def _critic_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Use critic head to output value of current state.
        Arguments:
            - x (:obj:`torch.Tensor`): embedding tensor after encoder
        """
        return self._critic(x).squeeze(1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
                Overview:
                    First encode raw observation, then output value and logit.
                    Normal reinforcement learning training, often called by learner.rst to optimize both critic and actor.
                Arguments:
                    - inputs (:obj:`Dict[str, torch.Tensor]`): embedding tensor after encoder
                Returns:
                    - ret (:obj:`Dict[str, torch.Tensor]`): a dict containing value and logit
                """
        # for compatible, but we recommend use dict as input format
        if isinstance(inputs, torch.Tensor):
            embedding = self._encoder(inputs)
        else:
            embedding = self._encoder(inputs['obs'])
        value = self._critic_forward(embedding)

        return {'value': value}


class PPGPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of PPG algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main agent.
        """
        # Optimizer
        self._optimizer_policy = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._agent = Agent(self._model)
        self._critic = ValueNet(self._cfg.model.obs_dim, self._cfg.model.embedding_dim)
        self._optimizer_value = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._clip_ratio = algo_cfg.clip_ratio
        self._use_adv_norm = algo_cfg.get('use_adv_norm', False)

        # Main agent
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.mode(train=True)
        self._agent.reset()
        self._learn_setting_set = {}

        # Auxiliary memories
        self._epochs_aux = algo_cfg.epochs_aux
        self._train_step = 0
        self._aux_memories = []
        self._beta_weight = algo_cfg.beta_weight

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        # ====================
        # PPG forward
        # ====================
        policy_output = self._agent.forward(data['obs'], param={'mode': 'compute_action_value'})
        value_output = self._critic.forward(data['obs'])
        adv = data['adv']
        if self._use_adv_norm:
            # Normalize advantage in a total train_batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return_ = data['value'] + adv
        # Calculate ppg error
        data_ = ppo_data(
            policy_output['logit'], data['logit'], data['action'], value_output['value'], data['value'], adv, return_,
            data['weight']
        )
        ppg_loss, ppg_info = ppo_error(data_, self._clip_ratio)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = ppg_loss.policy_loss + wv * ppg_loss.value_loss - we * ppg_loss.entropy_loss
        # ====================
        # PP update TODO(zym) update optimizer
        # TODO(zym) calculate value loss for update value network
        # use aux loss after iterations and reset aux_memories
        # ====================
        self._optimizer_policy.zero_grad()
        self._optimizer_value.zero_grad()
        total_loss.backward(retain_graph=True)
        self._optimizer_policy.step()
        self._optimizer_value.step()

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].detach()
        self._aux_memories.append(copy.deepcopy(data))
        if self._train_step < self._cfg.learn.train_step - 1:
            self._train_step += 1
        else:
            self.learn_aux()
        return {
            'cur_lr': self._optimizer_policy.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': ppg_loss.policy_loss.item(),
            'value_loss': ppg_loss.value_loss.item(),
            'entropy_loss': ppg_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'approx_kl': ppg_info.approx_kl,
            'clipfrac': ppg_info.clipfrac,
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect agent.
        """
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        if self._traj_len == 'inf':
            self._traj_len = float('inf')
        # GAE calculation needs v_t+1
        assert self._traj_len > 1, "ppg traj len should be greater than 1"
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'multinomial_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {}
        self._adder = Adder(self._use_cuda, self._unroll_len)
        algo_cfg = self._cfg.collect.algo
        self._gamma = algo_cfg.discount_factor
        self._gae_lambda = algo_cfg.gae_lambda

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        policy_output = self._collect_agent.forward(data, param={'mode': 'compute_action'})
        value_output = self._critic.forward(data)
        policy_output['value'] = value_output['value']
        for key in policy_output.keys():
            policy_output[key] = policy_output[key].detach()
        return policy_output

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - agent_output (:obj:`dict`): Output of collect agent, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'logit': agent_output['logit'],
            'action': agent_output['action'],
            'value': agent_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - traj_cache (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=1)
        if self._traj_len == float('inf'):
            assert data[-1]['done'], "episode must be terminated by done=True"
        data = self._adder.get_gae_with_default_last_value(
            data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
        )
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval agent with argmax strategy.
        """
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'argmax_sample')
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}
        self._eval_critic = self._critic

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        policy_output = self._eval_agent.forward(data, param={'mode': 'compute_action'})
        value_output = self._critic.forward(data)
        policy_output['value'] = value_output['value']
        return policy_output

    def _init_command(self) -> None:
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_vac', ['nervex.model.actor_critic.value_ac']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'approx_kl', 'clipfrac'
        ]

    def learn_aux(self):
        aux_memories = self._aux_memories
        # gather states and target values into one tensor
        data = {}
        states = []
        actions = []
        return_ = []
        old_values = []
        weights = []
        for memory in aux_memories:
            # for memory in memories:
            states.append(memory['obs'].clone().detach())
            actions.append(memory['action'].clone().detach())
            return_.append(memory['value'].clone().detach() + memory['adv'].clone().detach())
            old_values.append(memory['value'].clone().detach())
            if memory['weight'] is None:
                weight = torch.ones_like(memory['action'].clone().detach())
            else:
                weight = torch.tensor(memory['weight'])
            weights.append(weight)

        data['obs'] = torch.cat(states)
        data['action'] = torch.cat(actions)
        data['return_'] = torch.cat(return_)
        data['value'] = torch.cat(old_values)
        data['weight'] = torch.cat(weights)

        # get old action predictions for minimizing kl divergence and clipping respectively
        policy_output = self._collect_agent.forward(data, param={'mode': 'compute_action_value'})
        logit_old = policy_output['logit'].detach()
        action = policy_output['action'].detach()
        data['logit_old'] = logit_old

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader(data, self._cfg.learn.batch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)
        # TODO(zym) replace sample
        for epoch in range(self._epochs_aux):
            for data in dl:
                policy_output = self._agent.forward(data['obs'], param={'mode': 'compute_action_value'})

                # Calculate ppg error 'logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'
                data_ppg = ppg_data(
                    policy_output['logit'], data['logit_old'], data['action'], policy_output['value'], data['value'],
                    data['return_'], data['weight']
                )
                ppg_loss = ppg_joint_error(data_ppg, self._clip_ratio)
                wb = self._beta_weight
                total_loss = ppg_loss.auxiliary_loss + wb * ppg_loss.behavioral_cloning_loss

                # # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                # aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                # loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                # policy_loss = aux_loss + loss_kl

                self._optimizer_policy.zero_grad()
                total_loss.backward()
                self._optimizer_policy.step()

                # paper says it is important to train the value network extra during the auxiliary phase
                # TODO(zym) calculate value loss for update value network
                # Calculate ppg error 'logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'
                values = self._critic(data['obs'])['value']
                data_aux = ppg_aux_data(values, data['value'], data['return_'], data['weight'])

                value_loss = value_error(data_aux, self._clip_ratio)['value_loss']

                self._optimizer_value.zero_grad()
                value_loss.backward()
                self._optimizer_value.step()
        self._train_step = 0
        self._aux_memories = []


register_policy('ppg', PPGPolicy)
