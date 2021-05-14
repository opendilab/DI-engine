from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, Adder
from nervex.model import model_wrap
from nervex.data import timestep_collate, default_collate, default_decollate
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy


@POLICY_REGISTRY.register('r2d2')
class R2D2Policy(Policy):
    r"""
    Overview:
        Policy class of R2D2, from paper `Recurrent Experience Replay in Distributed Reinforcement Learning` .

        R2D2 proposed that several tricks should be used to improve upon DRQN,
        namely some recurrent experience replay trick such as burn-in.
    """
    default_nstep = 3
    default_burnin_step = 2
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='r2d2',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether to use multi gpu
        multi_gpu=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        learn=dict(
            update_per_collect=1,
            batch_size=64,
            learning_rate=0.0001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.97,
            # (int) N-step reward for target q_value estimation
            nstep=default_nstep,
            # (int) the timestep of burnin operation, which is designed to RNN hidden state difference
            # caused by off-policy
            burnin_step=default_burnin_step,
            # (bool) whether use value_rescale function for predicted value
            value_rescale=True,
            ignore_done=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_step, n_episode] shoule be set
            n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len", r2d2 should be 2*nstep + buin
            unroll_len=(2 * default_nstep + default_burnin_step),
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) N-step reward for target q_value estimation
            nstep=default_nstep,
            # (int) the timestep of burnin operation, which is designed to RNN hidden state difference
            # caused by off-policy
            burnin_step=default_burnin_step,
            # `env_num` is used in hidden state, should equal to that one in env config.
            # User should specify this value in user config.
            env_num=None,
        ),
        eval=dict(
            # `env_num` is used in hidden state, should equal to that one in env config.
            # User should specify this value in user config.
            env_num=None,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner model of R2D2Policy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - nstep (:obj:`int`): The num of n step return
            - value_rescale (:obj:`bool`): Whether to use value rescaled loss in algorithm
            - burnin_step (:obj:`int`): The num of step of burnin
        """
        self._priority = self._cfg.priority
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.learn.nstep
        self._value_rescale = self._cfg.learn.value_rescale
        self._burnin_step = self._cfg.learn.burnin_step

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._target_model = model_wrap(
            self._target_model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size
        )
        self._learn_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.learn.batch_size)
        self._learn_model = model_wrap(self._learn_model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']
            - data_info (:obj:`dict`): the data info, such as replay_buffer_idx, replay_unique_id
        """
        # data preprocess
        data = timestep_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        assert len(data['obs']) == 2 * self._nstep + self._burnin_step, data['obs'].shape  # todo: why 2*a+b
        bs = self._burnin_step
        data['weight'] = data.get('weight', [None for _ in range(self._nstep)])
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = [None for _ in range(self._nstep)]
        else:
            data['done'] = data['done'][bs:bs + self._nstep].float()
        data['action'] = data['action'][bs:bs + self._nstep]
        data['reward'] = data['reward'][bs:]
        # split obs into three parts ['burnin_obs'(0~bs), 'main_obs'(bs~bs+nstep), 'target_obs'(bs+nstep~bss+2nstep)]
        data['burnin_obs'] = data['obs'][:bs]
        data['main_obs'] = data['obs'][bs:bs + self._nstep]
        data['target_obs'] = data['obs'][bs + self._nstep:]
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
            Acquire the data, calculate the loss and optimize learner model.

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        # forward
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset(data_id=None, state=data['prev_state'][0])
        self._target_model.reset(data_id=None, state=data['prev_state'][0])
        if len(data['burnin_obs']) != 0:
            with torch.no_grad():
                inputs = {'obs': data['burnin_obs'], 'enable_fast_timestep': True}
                _ = self._learn_model.forward(inputs)
                _ = self._target_model.forward(inputs)
        inputs = {'obs': data['main_obs'], 'enable_fast_timestep': True}
        q_value = self._learn_model.forward(inputs)['logit']
        next_inputs = {'obs': data['target_obs'], 'enable_fast_timestep': True}
        with torch.no_grad():
            target_q_value = self._target_model.forward(next_inputs)['logit']
            target_q_action = self._learn_model.forward(next_inputs)['action']

        action, reward, done, weight = data['action'], data['reward'], data['done'], data['weight']
        # T, B, nstep -> T, nstep, B
        reward = reward.permute(0, 2, 1).contiguous()
        loss = []
        td_error = []
        for t in range(self._nstep):
            td_data = q_nstep_td_data(
                q_value[t], target_q_value[t], action[t], target_q_action[t], reward[t], done[t], weight[t]
            )
            if self._value_rescale:
                l, e = q_nstep_td_error_with_rescale(td_data, self._gamma, self._nstep)
                loss.append(l)
                td_error.append(e.abs())
            else:
                l, e = q_nstep_td_error(td_data, self._gamma, self._nstep)
                loss.append(l)
                td_error.append(e.abs())
        loss = sum(loss) / (len(loss) + 1e-8)
        td_error_per_sample = sum(td_error) / (len(td_error) + 1e-8)
        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
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
        """
        self._collect_nstep = self._cfg.collect.nstep
        self._collect_burnin_step = self._cfg.collect.burnin_step
        self._unroll_len = self._cfg.collect.unroll_len
        assert self._unroll_len == self._collect_burnin_step + 2 * self._collect_nstep
        self._adder = Adder(self._cuda, self._unroll_len)
        self._collect_model = model_wrap(
            self._model, wrapper_name='hidden_state', state_num=self._cfg.collect.env_num, save_prev_state=True
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Collect output according to eps_greedy plugin

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
            output = self._collect_model.forward(data, data_id=data_id, eps=eps)
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
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'action': model_output['action'],
            'prev_state': model_output['prev_state'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - data (:obj:`deque`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = self._adder.get_nstep_return_data(data, self._collect_nstep)
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.eval.env_num)
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode, similar to ``self._forward_collect``.

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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'fcr_discrete_net', ['nervex.model.discrete_net.discrete_net']
