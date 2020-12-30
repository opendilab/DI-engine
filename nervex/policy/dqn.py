from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, q_nstep_td_data, q_nstep_td_error, epsilon_greedy, Adder
from nervex.model import FCDiscreteNet, ConvDiscreteNet
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class DQNPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of DQN algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target agents.
        """
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._nstep = algo_cfg.nstep
        self._gamma = algo_cfg.discount_factor

        # Main and target agents
        self._agent = Agent(self._model)
        self._agent.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._agent.add_plugin('main', 'argmax_sample')
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """

        # ====================
        # Q-learning forward
        # ====================
        # Reward reshaping for n-step
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        assert reward.shape == (self._cfg.learn.batch_size, self._nstep), reward.shape
        reward = reward.permute(1, 0).contiguous()
        # Current q value (main agent)
        q_value = self._agent.forward(data['obs'])['logit']
        # Target q value
        target_q_value = self._agent.target_forward(data['next_obs'])['logit']
        # Max q value action (main agent)
        target_q_action = self._agent.forward(data['next_obs'])['action']

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, reward, data['done'], data['weight']
        )
        loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep)

        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # =============
        # after update
        # =============
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect agent.
            Enable the eps_greedy_sample
        """
        self._traj_len = self._cfg.collect.traj_len
        if self._traj_len == "inf":
            self._traj_len == float("inf")
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_nstep = self._cfg.collect.algo.nstep
        self._collect_agent = Agent(self._model)
        self._collect_agent.add_plugin('main', 'eps_greedy_sample')
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {'eps'}

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        return self._collect_agent.forward(data, eps=self._eps)

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
        return_num = 0 if self._collect_nstep == 1 else self._collect_nstep
        data = self._adder.get_traj(traj_cache, self._traj_len, return_num=return_num)
        data = self._adder.get_nstep_return_data(data, self._collect_nstep, self._traj_len)
        return self._adder.get_train_sample(data)

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
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

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
        return self._eval_agent.forward(data)

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
            Set the eps_greedy rule according to the config for command
        """
        eps_cfg = self._cfg.command.eps
        self.epsilon_greedy = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _get_setting_collect(self, command_info: dict) -> dict:
        r"""
        Overview:
            Collect mode setting information including eps
        Arguments:
            - command_info (:obj:`dict`): Dict type, including at least ['learner_step']
        Returns:
           - collect_setting (:obj:`dict`): Including eps in collect mode.
        """
        learner_step = command_info['learner_step']
        return {'eps': self.epsilon_greedy(learner_step)}

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
            return FCDiscreteNet(**cfg.model)
        else:
            return model_type(**cfg.model)


register_policy('dqn', DQNPolicy)
