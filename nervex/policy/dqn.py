from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import copy
import torch
import logging
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import q_1step_td_data, q_1step_td_error, q_nstep_td_data, q_nstep_td_error, Adder
from nervex.model import FCDiscreteNet, ConvDiscreteNet, model_wrap
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('dqn')
class DQNPolicy(Policy):
    r"""
    Overview:
        Policy class of DQN algorithm.
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        return copy.deepcopy(EasyDict(cls.config))

    config = dict(
        # RL policy register name (refer to function "register_policy").
        policy_type='dqn',
        # Whether to use cuda for network.
        use_cuda=False,
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        learn=dict(
            # How many iterations to train after collector's one collection.
            # Bigger "train_iteration" means bigger off-policy.
            # collect data -> train fixed iterations -> collect data -> ...
            train_iteration=3,
            batch_size=64,
            learning_rate=0.001,
            # L2 norm weight for network parameters.
            weight_decay=0.0,
            algo=dict(
                # Frequence of target network update.
                target_update_freq=100,
                # Reward's future discount facotr, aka. gamma.
                discount_factor=0.97,
                # How many steps in td error.
                nstep=1,
            ),
        ),
        # collect_mode config
        collect=dict(
            n_sample=8,
            traj_len=1,
            # Cut trajectories into pieces with length "unrol_len".
            unroll_len=1,
            algo=dict(nstep=1, ),
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._nstep = algo_cfg.nstep
        self._gamma = algo_cfg.discount_factor

        # use wrapper instead of plugin
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': algo_cfg.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

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
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

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
            self.sync_gradients(self._learn_model)
        self._optimizer.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect model.
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
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

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
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
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

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

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
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fc_discrete_net', ['nervex.model.discrete_net.discrete_net']
