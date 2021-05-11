from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, get_epsilon_greedy_fn, Adder
from nervex.model import QMix, model_wrap
from nervex.data import timestep_collate, default_collate, default_decollate
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy


@POLICY_REGISTRY.register('qmix')
class QMIXPolicy(Policy):
    r"""
    Overview:
        Policy class of QMIX algorithm. QMIX is a multi model reinforcement learning algorithm, \
            you can view the paper in the following link <https://arxiv.org/abs/1803.11485>_
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='qmix',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether to use multi gpu
        multi_gpu=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        learn=dict(
            update_per_collect=20,
            batch_size=64,
            learning_rate=0.0005,
            weight_decay=0.0001,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Target network update momentum parameter.
            # in [0, 1].
            target_update_theta=0.001,
            # (float) The discount factor for future rewards,
            # in [0, 1].
            discount_factor=0.99,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            n_episode=8,
            # (int) Cut trajectories into pieces with length "unroll_len", the length of timesteps
            # in each forward when training. In qmix, it is greater than 1 because there is RNN.
            unroll_len=20,
        ),
        eval=dict(),
        other=dict(
            eps=dict(
                # (str) Type of epsilon decay
                type='exp',
                # (float) Start value for epsilon decay, in [0, 1].
                # 0 means not use epsilon decay.
                start=1,
                # (float) Start value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Decay length(env step)
                decay=20000,
            ),
            replay_buffer=dict(
                replay_buffer_size=5000,
                # (int) The maximum reuse times of each data
                max_reuse=10,
            ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner model of QMIXPolicy
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._priority = self._cfg.priority
        assert not self._priority, "not implemented priority in PPO"
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.learn.discount_factor

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, from \
                [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}
        """
        # data preprocess
        data = timestep_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['obs', 'next_obs', 'action', 'reward', 'next_obs', 'prev_state', 'done']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # Q-mix forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # for hidden_state plugin, we need to reset the main model and target model
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        total_q = self._learn_model.forward(inputs, single_step=False)['total_q']
        next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, single_step=False)['total_q']

        data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        loss, td_error_per_sample = v_1step_td_error(data, self._gamma)
        # ====================
        # Q-mix update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset(data_id=data_id)

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
            Enable the eps_greedy_sample and the hidden_state plugin.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._cuda, self._unroll_len)
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
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
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': model_output['prev_state'],
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy and the hidden_state plugin.
        """
        self._eval_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory from the adder.
        Arguments:
            - data (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        return self._adder.get_train_sample(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qmix', ['nervex.model.qmix.qmix']
