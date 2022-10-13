import copy
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union, Optional

import torch

from ding.model import model_wrap
from ding.rl_utils import q_nstep_td_error_with_rescale, get_nstep_return_data, \
    get_train_sample, dqfd_nstep_td_error, dqfd_nstep_td_error_with_rescale, dqfd_nstep_td_data
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('r2d3')
class R2D3Policy(Policy):
    r"""
    Overview:
        Policy class of r2d3, from paper `Making Efficient Use of Demonstrations to Solve Hard Exploration Problems` .

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn            | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.997,         | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  ``burnin_step``      int      2              | The timestep of burnin operation,
                                                        | which is designed to RNN hidden state
                                                        | difference caused by off-policy
        9  | ``learn.update``   int      1              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        10 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.value_``   bool     True           | Whether use value_rescale function for
           | ``rescale``                                | predicted value
        13 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        14 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        15 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        16 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='r2d3',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # ==============================================================
        # The following configs are algorithm-specific
        # ==============================================================
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.997,
        # (int) N-step reward for target q_value estimation
        nstep=5,
        # (int) the timestep of burnin operation, which is designed to RNN hidden state difference
        # caused by off-policy
        burnin_step=2,
        # (int) the trajectory length to unroll the RNN network minus
        # the timestep of burnin operation
        learn_unroll_len=80,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=1,
            batch_size=64,
            learning_rate=0.0001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            # target_update_freq=100,
            target_update_theta=0.001,
            # (bool) whether use value_rescale function for predicted value
            value_rescale=True,
            ignore_done=False,
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            # each_iter_n_sample=32,
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

    def default_model(self) -> Tuple[str, List[str]]:
        return 'drqn', ['ding.model.template.q_learning']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner model of r2d3Policy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - nstep (:obj:`int`): The num of n step return
            - value_rescale (:obj:`bool`): Whether to use value rescaled loss in algorithm
            - burnin_step (:obj:`int`): The num of step of burnin
        """
        self.lambda1 = self._cfg.learn.lambda1  # n-step return
        self.lambda2 = self._cfg.learn.lambda2  # supervised loss
        self.lambda3 = self._cfg.learn.lambda3  # L2
        self.lambda_one_step_td = self._cfg.learn.lambda_one_step_td  # 1-step return
        # margin function in JE, here we implement this as a constant
        self.margin_function = self._cfg.learn.margin_function

        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._optimizer = Adam(
            self._model.parameters(), lr=self._cfg.learn.learning_rate, weight_decay=self.lambda3, optim_type='adamw'
        )
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._burnin_step = self._cfg.burnin_step
        self._value_rescale = self._cfg.learn.value_rescale

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
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
        )
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

        if self._priority_IS_weight:
            assert self._priority, "Use IS Weight correction, but Priority is not used."
        if self._priority and self._priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)

        bs = self._burnin_step

        # data['done'], data['weight'], data['value_gamma'] is used in def _forward_learn() to calculate
        # the q_nstep_td_error, should be length of [self._sequence_len-self._burnin_step-self._nstep]
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = [None for _ in range(self._sequence_len - bs)]
        else:
            data['done'] = data['done'][bs:].float()
            # NOTE that after the proprocessing of  get_nstep_return_data() in _get_train_sample
            # the data['done'] [t] is already the n-step done

        # if the data don't include 'weight' or 'value_gamma' then fill in None in a list
        # with length of [self._sequence_len-self._burnin_step-self._nstep],
        # below is two different implementation ways
        if 'value_gamma' not in data:
            data['value_gamma'] = [None for _ in range(self._sequence_len - bs)]
        else:
            data['value_gamma'] = data['value_gamma'][bs:]

        if 'weight' not in data:
            data['weight'] = [None for _ in range(self._sequence_len - bs)]
        else:
            data['weight'] = data['weight'] * torch.ones_like(data['done'])
            # every timestep in sequence has same weight, which is the _priority_IS_weight in PER

        data['action'] = data['action'][bs:-self._nstep]
        data['reward'] = data['reward'][bs:-self._nstep]

        # the burnin_nstep_obs is used to calculate the init hidden state of rnn for the calculation of the q_value,
        # target_q_value, and target_q_action
        data['burnin_nstep_obs'] = data['obs'][:bs + self._nstep]
        # the main_obs is used to calculate the q_value, the [bs:-self._nstep] means using the data from
        # [bs] timestep to [self._sequence_len-self._nstep] timestep
        data['main_obs'] = data['obs'][bs:-self._nstep]
        # the target_obs is used to calculate the target_q_value
        data['target_obs'] = data['obs'][bs + self._nstep:]

        # TODO(pu)
        data['target_obs_one_step'] = data['obs'][bs + 1:]
        if ignore_done:
            data['done_one_step'] = [None for _ in range(self._sequence_len - bs)]
        else:
            data['done_one_step'] = data['done_one_step'][bs:].float()

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
        # take out the hidden state in timestep=0
        self._learn_model.reset(data_id=None, state=data['prev_state'][0])
        self._target_model.reset(data_id=None, state=data['prev_state'][0])

        if len(data['burnin_nstep_obs']) != 0:
            with torch.no_grad():
                inputs = {'obs': data['burnin_nstep_obs'], 'enable_fast_timestep': True}
                burnin_output = self._learn_model.forward(
                    inputs,
                    saved_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep, self._burnin_step + 1]
                )
                burnin_output_target = self._target_model.forward(
                    inputs,
                    saved_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep, self._burnin_step + 1]
                )

        self._learn_model.reset(data_id=None, state=burnin_output['saved_state'][0])
        inputs = {'obs': data['main_obs'], 'enable_fast_timestep': True}
        q_value = self._learn_model.forward(inputs)['logit']

        # n-step
        self._learn_model.reset(data_id=None, state=burnin_output['saved_state'][1])
        self._target_model.reset(data_id=None, state=burnin_output_target['saved_state'][1])

        next_inputs = {'obs': data['target_obs'], 'enable_fast_timestep': True}
        with torch.no_grad():
            target_q_value = self._target_model.forward(next_inputs)['logit']
            # argmax_action double_dqn
            target_q_action = self._learn_model.forward(next_inputs)['action']

        # one-step
        self._learn_model.reset(data_id=None, state=burnin_output['saved_state'][2])
        self._target_model.reset(data_id=None, state=burnin_output_target['saved_state'][2])

        next_inputs_one_step = {'obs': data['target_obs_one_step'], 'enable_fast_timestep': True}
        with torch.no_grad():
            target_q_value_one_step = self._target_model.forward(next_inputs_one_step)['logit']
            # argmax_action double_dqn
            target_q_action_one_step = self._learn_model.forward(next_inputs_one_step)['action']

        action, reward, done, weight = data['action'], data['reward'], data['done'], data['weight']
        value_gamma = data['value_gamma']
        done_one_step = data['done_one_step']
        # T, B, nstep -> T, nstep, B
        reward = reward.permute(0, 2, 1).contiguous()
        loss = []
        loss_nstep = []
        loss_1step = []
        loss_sl = []
        td_error = []
        for t in range(self._sequence_len - self._burnin_step - self._nstep):
            # here t=0 means timestep <self._burnin_step> in the original sample sequence, we minus self._nstep
            # because for the last <self._nstep> timestep in the sequence, we don't have their target obs
            td_data = dqfd_nstep_td_data(
                q_value[t],
                target_q_value[t],
                action[t],
                target_q_action[t],
                reward[t],
                done[t],
                done_one_step[t],
                weight[t],
                target_q_value_one_step[t],
                target_q_action_one_step[t],
                data['is_expert'][t],  # is_expert flag(expert 1, agent 0)
            )

            if self._value_rescale:
                l, e, loss_statistics = dqfd_nstep_td_error_with_rescale(
                    td_data,
                    self._gamma,
                    self.lambda1,
                    self.lambda2,
                    self.margin_function,
                    self.lambda_one_step_td,
                    self._nstep,
                    False,
                    value_gamma=value_gamma[t],
                )
                loss.append(l)
                # td_error.append(e.abs())  # first sum then abs
                td_error.append(e)  # first abs then sum
                # loss statistics for debugging
                loss_nstep.append(loss_statistics[0])
                loss_1step.append(loss_statistics[1])
                loss_sl.append(loss_statistics[2])

            else:
                l, e, loss_statistics = dqfd_nstep_td_error(
                    td_data,
                    self._gamma,
                    self.lambda1,
                    self.lambda2,
                    self.margin_function,
                    self.lambda_one_step_td,
                    self._nstep,
                    False,
                    value_gamma=value_gamma[t],
                )
                loss.append(l)
                # td_error.append(e.abs())  # first sum then abs
                td_error.append(e)  # first abs then sum
                # loss statistics for debugging
                loss_nstep.append(loss_statistics[0])
                loss_1step.append(loss_statistics[1])
                loss_sl.append(loss_statistics[2])

        loss = sum(loss) / (len(loss) + 1e-8)
        # loss statistics for debugging
        loss_nstep = sum(loss_nstep) / (len(loss_nstep) + 1e-8)
        loss_1step = sum(loss_1step) / (len(loss_1step) + 1e-8)
        loss_sl = sum(loss_sl) / (len(loss_sl) + 1e-8)

        # using the mixture of max and mean absolute n-step TD-errors as the priority of the sequence
        td_error_per_sample = 0.9 * torch.max(
            torch.stack(td_error), dim=0
        )[0] + (1 - 0.9) * (torch.sum(torch.stack(td_error), dim=0) / (len(td_error) + 1e-8))
        # td_error shape list(<self._sequence_len-self._burnin_step-self._nstep>, B), for example, (75,64)
        # torch.sum(torch.stack(td_error), dim=0) can also be replaced with sum(td_error)

        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._target_model.update(self._learn_model.state_dict())

        # the information for debug
        batch_range = torch.arange(action[0].shape[0])
        q_s_a_t0 = q_value[0][batch_range, action[0]]
        target_q_s_a_t0 = target_q_value[0][batch_range, target_q_action[0]]

        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            # loss statistics for debugging
            'nstep_loss': loss_nstep.item(),
            '1step_loss': loss_1step.item(),
            'sl_loss': loss_sl.item(),
            'priority': td_error_per_sample.abs().tolist(),
            # the first timestep in the sequence, may not be the start of episode
            'q_s_taken-a_t0': q_s_a_t0.mean().item(),
            'target_q_s_max-a_t0': target_q_s_a_t0.mean().item(),
            'q_s_a-mean_t0': q_value[0].mean().item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        assert 'unroll_len' not in self._cfg.collect, "r2d3 use default unroll_len"
        self._nstep = self._cfg.nstep
        self._burnin_step = self._cfg.burnin_step
        self._gamma = self._cfg.discount_factor
        self._sequence_len = self._cfg.learn_unroll_len + self._cfg.burnin_step
        self._unroll_len = self._sequence_len  # for compatibility

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
            # in collect phase, inference=True means that each time we only pass one timestep data,
            # so the we can get the hidden state of rnn: <prev_state> at each timestep.
            output = self._collect_model.forward(data, data_id=data_id, eps=eps, inference=True)
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
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - data (:obj:`list`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        from copy import deepcopy
        data_one_step = deepcopy(get_nstep_return_data(data, 1, gamma=self._gamma))
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        for i in range(len(data)):
            # here we record the one-step done, we don't need record one-step reward,
            # because the n-step reward in data already include one-step reward
            data[i]['done_one_step'] = data_one_step[i]['done']
        return get_train_sample(data, self._sequence_len)

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
            output = self._eval_model.forward(data, data_id=data_id, inference=True)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset(data_id=data_id)

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'total_loss', 'nstep_loss', '1step_loss', 'sl_loss', 'priority', 'q_s_taken-a_t0', 'target_q_s_max-a_t0',
            'q_s_a-mean_t0'
        ]
