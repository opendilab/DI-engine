from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy

from ding.torch_utils import to_device, RMSprop
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('collaq')
class CollaQPolicy(Policy):
    r"""
    Overview:
        Policy class of CollaQ algorithm. CollaQ is a multi-agent reinforcement learning algorithm
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _reset_learn, _state_dict_learn, _load_state_dict_learn\
            _init_collect, _forward_collect, _reset_collect, _process_transition, _init_eval, _forward_eval\
            _reset_eval, _get_train_sample, default_model
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      collaq         | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     True           | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update_``  int      20             | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.target_``  float    0.001          | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        9  | ``learn.collaq``   float    1.0            | The weight of collaq MARA loss
           | ``_loss_weight``
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='collaq',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # (int) Collect n_episode data, update_model n_iteration times
            update_per_collect=20,
            # (int) The number of data for a train iteration
            batch_size=32,
            # (float) Gradient-descent step size
            learning_rate=0.0005,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Target network update weight, theta * new_w + (1 - theta) * old_w, defaults in [0, 0.1]
            target_update_theta=0.001,
            # (float) Discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) The weight of collaq MARA loss
            collaq_loss_weight=1.0,
            # (float)
            clip_value=100,
            # (bool) Whether to use double DQN mechanism(target q for surpassing over estimation)
            double_q=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_episode=32,
            # (int) Cut trajectories into pieces with length "unroll_len", the length of timesteps
            # in each forward when training. In qmix, it is greater than 1 because there is RNN.
            unroll_len=10,
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
                decay=200000,
            ),
            replay_buffer=dict(
                # (int) max size of replay buffer
                replay_buffer_size=5000,
                max_reuse=10,
            ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner model of CollaQPolicy
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - alpha (:obj:`float`): The collaQ loss factor, the weight for calculating MARL loss
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "not implemented priority in collaq"
        self._optimizer = RMSprop(
            params=self._model.parameters(), lr=self._cfg.learn.learning_rate, alpha=0.99, eps=0.00001
        )
        self._gamma = self._cfg.learn.discount_factor
        self._alpha = self._cfg.learn.collaq_loss_weight

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
            init_fn=lambda: [[None for _ in range(self._cfg.model.agent_num)] for _ in range(3)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [[None for _ in range(self._cfg.model.agent_num)] for _ in range(3)]
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
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``next_obs``, ``action``, ``reward``, ``weight``, ``prev_state``, ``done``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # CollaQ forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # for hidden_state plugin, we need to reset the main model and target model
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}
        ret = self._learn_model.forward(inputs, single_step=False)
        total_q = ret['total_q']
        agent_colla_alone_q = ret['agent_colla_alone_q'].sum(-1).sum(-1)

        if self._cfg.learn.double_q:
            next_inputs = {'obs': data['next_obs']}
            logit_detach = self._learn_model.forward(next_inputs, single_step=False)['logit'].clone().detach()
            next_inputs = {'obs': data['next_obs'], 'action': logit_detach.argmax(dim=-1)}
        else:
            next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, single_step=False)['total_q']

        # td_loss calculation
        td_data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        td_loss, _ = v_1step_td_error(td_data, self._gamma)
        # collaQ loss calculation
        colla_loss = (agent_colla_alone_q ** 2).mean()
        # combine loss with factor
        loss = colla_loss * self._alpha + td_loss
        # ====================
        # CollaQ update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._cfg.learn.clip_value)
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'colla_loss': colla_loss.item(),
            'td_loss': td_loss.item(),
            'grad_norm': grad_norm,
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        r"""
        Overview:
            Reset learn model to the state indicated by data_id
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\
                the model state to the state indicated by data_id
        """
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Enable the eps_greedy_sample and the hidden_state plugin.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [[None for _ in range(self._cfg.model.agent_num)] for _ in range(3)]
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
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
        r"""
        Overview:
            Reset collect model to the state indicated by data_id
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\
                the model state to the state indicated by data_id
        """
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least \
                ['action', 'prev_state', 'agent_colla_alone_q']
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
            'agent_colla_alone_q': model_output['agent_colla_alone_q'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

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
            init_fn=lambda: [[None for _ in range(self._cfg.model.agent_num)] for _ in range(3)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
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
        r"""
        Overview:
            Reset eval model to the state indicated by data_id
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): The id that store the state and we will reset\
                the model state to the state indicated by data_id
        """
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the train sample from trajectory.
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        return get_train_sample(data, self._unroll_len)

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For collaq, ``ding.model.qmix.qmix``
        """
        return 'collaq', ['ding.model.template.qmix']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return ['cur_lr', 'total_loss', 'colla_loss', 'td_loss', 'grad_norm']
