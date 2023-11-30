import copy
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union, Optional

import torch

from ding.model import model_wrap
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, get_nstep_return_data, \
    get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('r2d2')
class R2D2Policy(Policy):
    """
    Overview:
        Policy class of R2D2, from paper `Recurrent Experience Replay in Distributed Reinforcement Learning` .
        R2D2 proposes that several tricks should be used to improve upon DRQN, namely some recurrent experience replay \
        tricks and the burn-in mechanism for off-policy training.
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      r2d2           | RL policy register name, refer to      | This arg is optional,
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
        type='r2d2',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether to use priority(priority sample, IS weight, update priority)
        priority=True,
        # (bool) Whether to use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.997,
        # (int) N-step reward for target q_value estimation
        nstep=5,
        # (int) the timestep of burnin operation, which is designed to RNN hidden state difference
        # caused by off-policy
        burnin_step=20,
        # (int) the trajectory length to unroll the RNN network minus
        # the timestep of burnin operation
        learn_unroll_len=80,
        # learn_mode config
        learn=dict(
            # (int) The number of training updates (iterations) to perform after each data collection by the collector.
            # A larger "update_per_collect" value implies a more off-policy approach.
            # The whole pipeline process follows this cycle: collect data -> update policy -> collect data -> ...
            update_per_collect=1,
            # (int) The number of samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent, determining the rate of learning.
            learning_rate=0.0001,
            # (int) Frequence of target network update.
            # target_update_freq=100,
            target_update_theta=0.001,
            # (bool) whether use value_rescale function for predicted value
            value_rescale=True,
            # (bool) Whether ignore done(usually for max step termination env).
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            # (bool) It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            traj_len_inf=True,
            # (int) `env_num` is used in hidden state, should equal to that one in env config (e.g. collector_env_num).
            # User should specify this value in user config. `None` is a placeholder.
            env_num=None,
        ),
        # eval_mode config
        eval=dict(
            # (int) `env_num` is used in hidden state, should equal to that one in env config (e.g. evaluator_env_num).
            # User should specify this value in user config.
            env_num=None,
        ),
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Type of decay. Supports either 'exp' (exponential) or 'linear'.
                type='exp',
                # (float) Initial value of epsilon at the start.
                start=0.95,
                # (float) Final value of epsilon after decay.
                end=0.05,
                # (int) The number of environment steps over which epsilon should decay.
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=10000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about R2D2, its registered name is ``drqn`` and the import_names is \
            ``ding.model.template.q_learning``.
        """
        return 'drqn', ['ding.model.template.q_learning']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including some attributes and modules. For R2D2, it mainly contains \
            optimizer, algorithm-specific arguments such as burnin_step, value_rescale and gamma, main and target \
            model. Because of the use of RNN, all the models should be wrappered with ``hidden_state`` which needs to \
            be initialized with proper size.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
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

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Preprocess the data to fit the required data format for learning
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): The data collected from collect function
        Returns:
            - data (:obj:`Dict[str, torch.Tensor]`): The processed data, including at least \
                ['main_obs', 'target_obs', 'burnin_obs', 'action', 'reward', 'done', 'weight']
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

        burnin_step = self._burnin_step

        # data['done'], data['weight'], data['value_gamma'] is used in def _forward_learn() to calculate
        # the q_nstep_td_error, should be length of [self._sequence_len-self._burnin_step]
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = [None for _ in range(self._sequence_len - burnin_step)]
        else:
            data['done'] = data['done'][burnin_step:].float()  # for computation of online model self._learn_model
            # NOTE that after the proprocessing of  get_nstep_return_data() in _get_train_sample
            # the data['done'] [t] is already the n-step done

        # if the data don't include 'weight' or 'value_gamma' then fill in None in a list
        # with length of [self._sequence_len-self._burnin_step],
        # below is two different implementation ways
        if 'value_gamma' not in data:
            data['value_gamma'] = [None for _ in range(self._sequence_len - burnin_step)]
        else:
            data['value_gamma'] = data['value_gamma'][burnin_step:]

        if 'weight' not in data or data['weight'] is None:
            data['weight'] = [None for _ in range(self._sequence_len - burnin_step)]
        else:
            data['weight'] = data['weight'] * torch.ones_like(data['done'])
            # every timestep in sequence has same weight, which is the _priority_IS_weight in PER

        # cut the seq_len from burn_in step to (seq_len - nstep) step
        data['action'] = data['action'][burnin_step:-self._nstep]
        # cut the seq_len from burn_in step to (seq_len - nstep) step
        data['reward'] = data['reward'][burnin_step:-self._nstep]

        # the burnin_nstep_obs is used to calculate the init hidden state of rnn for the calculation of the q_value,
        # target_q_value, and target_q_action

        # these slicing are all done in the outermost layer, which is the seq_len dim
        data['burnin_nstep_obs'] = data['obs'][:burnin_step + self._nstep]
        # the main_obs is used to calculate the q_value, the [bs:-self._nstep] means using the data from
        # [bs] timestep to [self._sequence_len-self._nstep] timestep
        data['main_obs'] = data['obs'][burnin_step:-self._nstep]
        # the target_obs is used to calculate the target_q_value
        data['target_obs'] = data['obs'][burnin_step + self._nstep:]

        return data

    def _forward_learn(self, data: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data (trajectory for R2D2) from the replay buffer and then \
            returns the output result, including various training information such as loss, q value, priority.
        Arguments:
            - data (:obj:`List[List[Dict[int, Any]]]`): The input data used for policy forward, including a batch of \
                training samples. For each dict element, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the time and \
                batch dimension by the utility functions ``self._data_preprocess_learn``. \
                For R2D2, each element in list is a trajectory with the length of ``unroll_len``, and the element in \
                trajectory list is a dict containing at least the following keys: ``obs``, ``action``, ``prev_state``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight`` \
                and ``value_gamma``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for R2D2Policy: ``ding.policy.tests.test_r2d2``.
        """
        # forward
        data = self._data_preprocess_learn(data)  # output datatype: Dict
        self._learn_model.train()
        self._target_model.train()
        # use the hidden state in timestep=0
        # note the reset method is performed at the hidden state wrapper, to reset self._state.
        self._learn_model.reset(data_id=None, state=data['prev_state'][0])
        self._target_model.reset(data_id=None, state=data['prev_state'][0])

        if len(data['burnin_nstep_obs']) != 0:
            with torch.no_grad():
                inputs = {'obs': data['burnin_nstep_obs'], 'enable_fast_timestep': True}
                burnin_output = self._learn_model.forward(
                    inputs, saved_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
                )  # keys include 'logit', 'hidden_state' 'saved_state', \
                # 'action', for their specific dim, please refer to DRQN model
                burnin_output_target = self._target_model.forward(
                    inputs, saved_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]
                )

        self._learn_model.reset(data_id=None, state=burnin_output['saved_state'][0])
        inputs = {'obs': data['main_obs'], 'enable_fast_timestep': True}
        q_value = self._learn_model.forward(inputs)['logit']
        self._learn_model.reset(data_id=None, state=burnin_output['saved_state'][1])
        self._target_model.reset(data_id=None, state=burnin_output_target['saved_state'][1])

        next_inputs = {'obs': data['target_obs'], 'enable_fast_timestep': True}
        with torch.no_grad():
            target_q_value = self._target_model.forward(next_inputs)['logit']
            # argmax_action double_dqn
            target_q_action = self._learn_model.forward(next_inputs)['action']

        action, reward, done, weight = data['action'], data['reward'], data['done'], data['weight']
        value_gamma = data['value_gamma']
        # T, B, nstep -> T, nstep, B
        reward = reward.permute(0, 2, 1).contiguous()
        loss = []
        td_error = []
        for t in range(self._sequence_len - self._burnin_step - self._nstep):
            # here t=0 means timestep <self._burnin_step> in the original sample sequence, we minus self._nstep
            # because for the last <self._nstep> timestep in the sequence, we don't have their target obs
            td_data = q_nstep_td_data(
                q_value[t], target_q_value[t], action[t], target_q_action[t], reward[t], done[t], weight[t]
            )
            if self._value_rescale:
                l, e = q_nstep_td_error_with_rescale(td_data, self._gamma, self._nstep, value_gamma=value_gamma[t])
                loss.append(l)
                td_error.append(e.abs())
            else:
                l, e = q_nstep_td_error(td_data, self._gamma, self._nstep, value_gamma=value_gamma[t])
                loss.append(l)
                # td will be a list of the length
                # <self._sequence_len - self._burnin_step - self._nstep>
                # and each value is a tensor of the size batch_size
                td_error.append(e.abs())
        loss = sum(loss) / (len(loss) + 1e-8)

        # using the mixture of max and mean absolute n-step TD-errors as the priority of the sequence
        td_error_per_sample = 0.9 * torch.max(
            torch.stack(td_error), dim=0
        )[0] + (1 - 0.9) * (torch.sum(torch.stack(td_error), dim=0) / (len(td_error) + 1e-8))
        # torch.max(torch.stack(td_error), dim=0) will return tuple like thing, please refer to torch.max
        # td_error shape list(<self._sequence_len-self._burnin_step-self._nstep>, B),
        # for example, (75,64)
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
            'priority': td_error_per_sample.tolist(),  # note abs operation has been performed above
            # the first timestep in the sequence, may not be the start of episode
            'q_s_taken-a_t0': q_s_a_t0.mean().item(),
            'target_q_s_max-a_t0': target_q_s_a_t0.mean().item(),
            'q_s_a-mean_t0': q_value[0].mean().item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        """
        Overview:
            Reset some stateful variables for learn mode when necessary, such as the hidden state of RNN or the \
            memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different trajectories in ``data_id`` will have different hidden state in RNN.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                (i.e. RNN hidden_state in R2D2) specified by ``data_id``.
        """

        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For R2D2, it contains the \
            collect_model to balance the exploration and exploitation with epsilon-greedy sample mechanism and \
            maintain the hidden state of rnn. Besides, there are some initialization operations about other \
            algorithm-specific arguments such as burnin_step, unroll_len and nstep.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and nstep in R2D2. This \
            design is for the convenience of parallel execution of different policy modes.
        """
        self._nstep = self._cfg.nstep
        self._burnin_step = self._cfg.burnin_step
        self._gamma = self._cfg.discount_factor
        self._sequence_len = self._cfg.learn_unroll_len + self._cfg.burnin_step
        self._unroll_len = self._sequence_len

        # for r2d2, this hidden_state wrapper is to add the 'prev hidden state' for each transition.
        # Note that collect env forms a batch and the key is added for the batch simultaneously.
        self._collect_model = model_wrap(
            self._model, wrapper_name='hidden_state', state_num=self._cfg.collect.env_num, save_prev_state=True
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs. Besides, this policy also needs ``eps`` argument for \
            exploration, i.e., classic epsilon-greedy exploration strategy.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
            - eps (:obj:`float`): The epsilon value for exploration.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data (prev_state) for learn mode defined in ``self._process_transition`` method. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            RNN's hidden states are maintained in the policy, so we don't need pass them into data but to reset the \
            hidden states with ``_reset_collect`` method when episode ends. Besides, the previous hidden states are \
            necessary for training, so we need to return them in ``_process_transition`` method.
        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for R2D2Policy: ``ding.policy.tests.test_r2d2``.
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
        """
        Overview:
            Reset some stateful variables for eval mode when necessary, such as the hidden state of RNN or the \
            memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different environments/episodes in evaluation in ``data_id`` will have different hidden state in RNN.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                (i.e., RNN hidden_state in R2D2) specified by ``data_id``.
        """
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For R2D2, it contains obs, action, prev_state, reward, and done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network given the observation \
                as input. For R2D2, it contains the action and the prev_state of RNN.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'action': policy_output['action'],
            'prev_state': policy_output['prev_state'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In R2D2, a train sample is processed transitions with unroll_len \
            length. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each sample is a fixed-length \
                trajectory, and each element in a sample is the similar format as input transitions, but may contain \
                more data for training, such as nstep reward and value_gamma factor.
        """
        transitions = get_nstep_return_data(transitions, self._nstep, gamma=self._gamma)
        return get_train_sample(transitions, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For R2D2, it contains the \
            eval model to greedily select action with argmax q_value mechanism and main the hidden state.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='hidden_state', state_num=self._cfg.eval.env_num)
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` often use argmax sample method to get actions that \
            q_value is the highest.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            RNN's hidden states are maintained in the policy, so we don't need pass them into data but to reset the \
            hidden states with ``_reset_eval`` method when the episode ends.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for R2D2Policy: ``ding.policy.tests.test_r2d2``.
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
        """
        Overview:
            Reset some stateful variables for eval mode when necessary, such as the hidden state of RNN or the \
            memory bank of some special algortihms. If ``data_id`` is None, it means to reset all the stateful \
            varaibles. Otherwise, it will reset the stateful variables according to the ``data_id``. For example, \
            different environments/episodes in evaluation in ``data_id`` will have different hidden state in RNN.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id of the data, which is used to reset the stateful variables \
                (i.e., RNN hidden_state in R2D2) specified by ``data_id``.
        """
        self._eval_model.reset(data_id=data_id)

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return super()._monitor_vars_learn() + [
            'total_loss', 'priority', 'q_s_taken-a_t0', 'target_q_s_max-a_t0', 'q_s_a-mean_t0'
        ]
