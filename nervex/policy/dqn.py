from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, q_nstep_td_data, q_nstep_td_error, Adder
from nervex.model import FCDiscreteNet, ConvDiscreteNet
from nervex.armor import Armor
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('dqn')
class DQNPolicy(Policy):
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
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.get('use_priority', False),
            ignore_done=self._cfg.learn.get('ignore_done', False),
            use_nstep=True
        )
        if self._use_cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._armor.model.train()
        self._armor.target_model.train()
        # Current q value (main armor)
        q_value = self._armor.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._armor.target_forward(data['next_obs'])['logit']
            # Max q value action (main armor)
            target_q_action = self._armor.forward(data['next_obs'])['action']

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
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
        self._collect_armor.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._collect_armor.model.eval()
        with torch.no_grad():
            output = self._collect_armor.forward(data, eps=eps)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
        self._eval_armor.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._eval_armor.model.eval()
        with torch.no_grad():
            output = self._eval_armor.forward(data)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_discrete_net', ['nervex.model.discrete_net.discrete_net']
