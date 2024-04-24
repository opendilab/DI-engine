from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy

from ding.torch_utils import RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .qmix import QMIXPolicy


@POLICY_REGISTRY.register('madqn')
class MADQNPolicy(QMIXPolicy):
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='madqn',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        nstep=3,
        learn=dict(
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            clip_value=100,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Target network update momentum parameter.
            # in [0, 1].
            target_update_theta=0.008,
            # (float) The discount factor for future rewards,
            # in [0, 1].
            discount_factor=0.99,
            # (bool) Whether to use double DQN mechanism(target q for surpassing over estimation)
            double_q=False,
            weight_decay=1e-5,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_episode=32,
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
                decay=50000,
            ),
            replay_buffer=dict(
                replay_buffer_size=5000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names
        """
        return 'madqn', ['ding.model.template.madqn']

    def _init_learn(self) -> None:
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in QMIX"
        self._optimizer_current = RMSprop(
            params=self._model.current.parameters(),
            lr=self._cfg.learn.learning_rate,
            alpha=0.99,
            eps=0.00001,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_cooperation = RMSprop(
            params=self._model.cooperation.parameters(),
            lr=self._cfg.learn.learning_rate,
            alpha=0.99,
            eps=0.00001,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.nstep
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
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
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
        # Q-mix forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # for hidden_state plugin, we need to reset the main model and target model
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        inputs = {'obs': data['obs'], 'action': data['action']}

        total_q = self._learn_model.forward(inputs, single_step=False)['total_q']

        if self._cfg.learn.double_q:
            next_inputs = {'obs': data['next_obs']}
            self._learn_model.reset(state=data['prev_state'][1])
            logit_detach = self._learn_model.forward(next_inputs, single_step=False)['logit'].clone().detach()
            next_inputs = {'obs': data['next_obs'], 'action': logit_detach.argmax(dim=-1)}
        else:
            next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            target_total_q = self._target_model.forward(next_inputs, cooperation=True, single_step=False)['total_q']

        if self._nstep == 1:

            v_data = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
            loss, td_error_per_sample = v_1step_td_error(v_data, self._gamma)
            # for visualization
            with torch.no_grad():
                if data['done'] is not None:
                    target_v = self._gamma * (1 - data['done']) * target_total_q + data['reward']
                else:
                    target_v = self._gamma * target_total_q + data['reward']
        else:
            data['reward'] = data['reward'].permute(0, 2, 1).contiguous()
            loss = []
            td_error_per_sample = []
            for t in range(self._cfg.collect.unroll_len):
                v_data = v_nstep_td_data(
                    total_q[t], target_total_q[t], data['reward'][t], data['done'][t], data['weight'], None
                )
                # calculate v_nstep_td critic_loss
                loss_i, td_error_per_sample_i = v_nstep_td_error(v_data, self._gamma, self._nstep)
                loss.append(loss_i)
                td_error_per_sample.append(td_error_per_sample_i)
            loss = sum(loss) / (len(loss) + 1e-8)
            td_error_per_sample = sum(td_error_per_sample) / (len(td_error_per_sample) + 1e-8)

        self._optimizer_current.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.current.parameters(), self._cfg.learn.clip_value)
        self._optimizer_current.step()

        # cooperation
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        cooperation_total_q = self._learn_model.forward(inputs, cooperation=True, single_step=False)['total_q']
        next_inputs = {'obs': data['next_obs']}
        with torch.no_grad():
            cooperation_target_total_q = self._target_model.forward(
                next_inputs, cooperation=True, single_step=False
            )['total_q']

        if self._nstep == 1:
            v_data = v_1step_td_data(
                cooperation_total_q, cooperation_target_total_q, data['reward'], data['done'], data['weight']
            )
            cooperation_loss, _ = v_1step_td_error(v_data, self._gamma)
        else:
            cooperation_loss_all = []
            for t in range(self._cfg.collect.unroll_len):
                v_data = v_nstep_td_data(
                    cooperation_total_q[t],
                    cooperation_target_total_q[t],
                    data['reward'][t],
                    data['done'][t],
                    data['weight'],
                    None,
                )
                cooperation_loss, _ = v_nstep_td_error(v_data, self._gamma, self._nstep)
                cooperation_loss_all.append(cooperation_loss)
            cooperation_loss = sum(cooperation_loss_all) / (len(cooperation_loss_all) + 1e-8)
        self._optimizer_cooperation.zero_grad()
        cooperation_loss.backward()
        cooperation_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.cooperation.parameters(), self._cfg.learn.clip_value
        )
        self._optimizer_cooperation.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer_current.defaults['lr'],
            'total_loss': loss.item(),
            'total_q': total_q.mean().item() / self._cfg.model.agent_num,
            'target_total_q': target_total_q.mean().item() / self._cfg.model.agent_num,
            'grad_norm': grad_norm,
            'cooperation_grad_norm': cooperation_grad_norm,
            'cooperation_loss': cooperation_loss.item(),
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
            'optimizer_current': self._optimizer_current.state_dict(),
            'optimizer_cooperation': self._optimizer_cooperation.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
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
        self._optimizer_current.load_state_dict(state_dict['optimizer_current'])
        self._optimizer_cooperation.load_state_dict(state_dict['optimizer_cooperation'])

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
            - transition (:obj:`dict`): Dict type transition data, including 'obs', 'next_obs', 'prev_state',\
                'action', 'reward', 'done'
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': model_output['prev_state'],
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the train sample from trajectory.
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        if self._cfg.nstep == 1:
            return get_train_sample(data, self._unroll_len)
        else:
            data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
            return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return [
            'cur_lr', 'total_loss', 'total_q', 'target_total_q', 'grad_norm', 'target_reward_total_q',
            'cooperation_grad_norm', 'cooperation_loss'
        ]
