from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, q_nstep_td_data, q_nstep_td_error, Adder
from nervex.model import FCDiscreteNet, ConvDiscreteNet
from nervex.armor import Armor
from nervex.utils import POLICY_REGISTRY
from .common_policy import CommonPolicy


@POLICY_REGISTRY.register('dqn')
class DQNPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of DQN algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target armors.
        """
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._nstep = algo_cfg.nstep
        self._gamma = algo_cfg.discount_factor

        # Main and target armors
        self._armor = Armor(self._model)
        self._armor.add_model('target', update_type='assign', update_kwargs={'freq': algo_cfg.target_update_freq})
        self._armor.add_plugin('main', 'argmax_sample')
        self._armor.mode(train=True)
        self._armor.target_mode(train=True)
        self._armor.reset()
        self._armor.target_reset()

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
        # Current q value (main armor)
        q_value = self._armor.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._armor.target_forward(data['next_obs'])['logit']
            # Max q value action (main armor)
            target_q_action = self._armor.forward(data['next_obs'])['action']

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, reward, data['done'], data['weight']
        )
        loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep)

        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        if self._use_distributed:
            self.sync_gradients(self._armor.model)
        self._optimizer.step()

        # =============
        # after update
        # =============
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Enable the eps_greedy_sample
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._use_her = self._cfg.collect.algo.get('use_her', False)
        if self._use_her:
            her_strategy = self._cfg.collect.algo.get('her_strategy', 'future')
            her_replay_k = self._cfg.collect.algo.get('her_replay_k', 1)
            self._adder = Adder(self._use_cuda, self._unroll_len, her_strategy=her_strategy, her_replay_k=her_replay_k)
        else:
            self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_nstep = self._cfg.collect.algo.nstep
        self._collect_armor = Armor(self._model)
        self._collect_armor.add_plugin('main', 'eps_greedy_sample')
        self._collect_armor.mode(train=False)
        self._collect_armor.reset()

    def _forward_collect(self, data_id: List[int], data: dict, eps: float) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        with torch.no_grad():
            output = self._collect_armor.forward(data, eps=eps)
        return output

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
        Arguments:
            - data (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        data = self._adder.get_nstep_return_data(data, self._collect_nstep)
        if self._use_her:
            data = self._adder.get_her(data)
        return self._adder.get_train_sample(data)

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin('main', 'argmax_sample')
        self._eval_armor.mode(train=False)
        self._eval_armor.reset()

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
        with torch.no_grad():
            output = self._eval_armor.forward(data)
        return output

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_discrete_net', ['nervex.model.discrete_net.discrete_net']
