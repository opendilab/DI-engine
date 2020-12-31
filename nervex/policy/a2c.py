from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch

from nervex.rl_utils import a2c_data, a2c_error, Adder, nstep_return_data, nstep_return
from nervex.torch_utils import Adam
from nervex.model import FCValueAC, ConvValueAC
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class A2CPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of A2C algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target agents.
        """
        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=self._cfg.learn.weight_decay
        )

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._learn_gamma = algo_cfg.get('discount_factor', 0.99)

        # Main and target agents
        self._agent = Agent(self._model)
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.mode(train=True)
        self._agent.reset()
        self._learn_setting_set = {}

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','adv']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        # forward
        output = self._agent.forward(data['obs'], param={'mode': 'compute_action_value'})
        adv = data['adv']

        # return = value + adv
        return_ = data['value'] + adv
        data = a2c_data(output['logit'], data['action'], output['value'], adv, return_, data['weight'])

        # Calculate A2C loss
        a2c_loss = a2c_error(data)
        wv, we = self._value_weight, self._entropy_weight
        total_loss = a2c_loss.policy_loss + wv * a2c_loss.value_loss - we * a2c_loss.entropy_loss

        # ====================
        # A2C-learning update
        # ====================

        self._optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self._model.parameters()),
            max_norm=0.5,
        )
        self._optimizer.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': a2c_loss.policy_loss.item(),
            'value_loss': a2c_loss.value_loss.item(),
            'entropy_loss': a2c_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
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
        # because gae calculation need v_t+1
        assert self._traj_len > 1, "a2c traj len should be greater than 1"
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
        return self._collect_agent.forward(data, param={'mode': 'compute_action_value'})

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - agent_output (:obj:`dict`): Output of collect agent, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """

        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': agent_output['action'],
            'value': agent_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
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
        return self._eval_agent.forward(data, param={'mode': 'compute_action'})

    def _init_command(self) -> None:
        pass

    def _create_model_from_cfg(self, cfg: dict, model_type: Optional[type] = None) -> torch.nn.Module:
        r"""
        Overview:
            Create a model according to input config. This policy will adopt DiscreteNet.
        Arguments:
            - cfg (:obj:`dict`): Config.
            - model_type (:obj:`Optional[type]`): If this is not None, this function will create \
                an instance of this.
        Returns:
            - model (:obj:`torch.nn.Module`): Generated model.
        """
        if model_type is None:
            return FCValueAC(**cfg.model)
        else:
            return model_type(**cfg.model)

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max']


register_policy('a2c', A2CPolicy)
