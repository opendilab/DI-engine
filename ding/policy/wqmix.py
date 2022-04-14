from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy

from ding.torch_utils import RMSprop, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy
from ding.policy.qmix import QMIXPolicy


@POLICY_REGISTRY.register('wqmix')
class WQMIXPolicy(QMIXPolicy):
    r"""
    Overview:
        Policy class of WQMIX algorithm. WQMIX is a reinforcement learning algorithm modified from Qmix, \
            you can view the paper in the following link https://arxiv.org/abs/2006.10800
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _reset_learn, _state_dict_learn, _load_state_dict_learn\
            _init_collect, _forward_collect, _reset_collect, _process_transition, _init_eval, _forward_eval\
            _reset_eval, _get_train_sample, default_model
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      qmix           | RL policy register name, refer to      | this arg is optional,
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
        7  | ``learn.target_``   float    0.001         | Target network update momentum         | between[0,1]
           | ``update_theta``                           | parameter.
        8  | ``learn.discount`` float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``_factor``                                | gamma                                  | reward env
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='wqmix',
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
            w=0.5,  # for OW
            # w = 0.75, # for CW
            wqmix_ow=True,
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

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the learner model of WQMIXPolicy
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in WQMIX"
        self._optimizer = RMSprop(
            params=list(self._model._q_network.parameters()) + list(self._model._mixer.parameters()),
            lr=self._cfg.learn.learning_rate,
            alpha=0.99,
            eps=0.00001
        )
        self._gamma = self._cfg.learn.discount_factor
        self._optimizer_star = RMSprop(
            params=list(self._model._q_network_star.parameters()) + list(self._model._mixer_star.parameters()),
            lr=self._cfg.learn.learning_rate,
            alpha=0.99,
            eps=0.00001
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model.reset()

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
        #  forward
        # ====================
        self._learn_model.train()

        inputs = {'obs': data['obs'], 'action': data['action']}

        # for hidden_state plugin, we need to reset the main model and target model
        self._learn_model.reset(state=data['prev_state'][0])
        total_q = self._learn_model.forward(inputs, single_step=False, q_star=False)['total_q']

        self._learn_model.reset(state=data['prev_state'][0])
        total_q_star = self._learn_model.forward(inputs, single_step=False, q_star=True)['total_q']

        next_inputs = {'obs': data['next_obs']}
        self._learn_model.reset(state=data['prev_state'][1])  # TODO(pu)
        next_logit_detach = self._learn_model.forward(
            next_inputs, single_step=False, q_star=False
        )['logit'].clone().detach()

        next_inputs = {'obs': data['next_obs'], 'action': next_logit_detach.argmax(dim=-1)}
        with torch.no_grad():
            self._learn_model.reset(state=data['prev_state'][1])  # TODO(pu)
            target_total_q = self._learn_model.forward(next_inputs, single_step=False, q_star=True)['total_q']

        with torch.no_grad():
            if data['done'] is not None:
                target_v = self._gamma * (1 - data['done']) * target_total_q + data['reward']
            else:
                target_v = self._gamma * target_total_q + data['reward']

        td_error = (total_q - target_v).clone().detach()
        data_ = v_1step_td_data(total_q, target_total_q, data['reward'], data['done'], data['weight'])
        _, td_error_per_sample = v_1step_td_error(data_, self._gamma)

        data_star = v_1step_td_data(total_q_star, target_total_q, data['reward'], data['done'], data['weight'])
        loss_star, td_error_per_sample_star_ = v_1step_td_error(data_star, self._gamma)

        # our implemention is based on the https://github.com/oxwhirl/wqmix
        # Weighting
        alpha_to_use = self._cfg.learn.alpha
        if self._cfg.learn.wqmix_ow:  # Optimistically-Weighted
            ws = torch.full_like(td_error, alpha_to_use)
            # if td_error < 0, i.e. Q < y_i, then w =1; if not, w = alpha_to_use
            ws = torch.where(td_error < 0, torch.ones_like(td_error), ws)
        else:  # Centrally-Weighted
            inputs = {'obs': data['obs']}
            self._learn_model.reset(state=data['prev_state'][0])  # TODO(pu)
            logit_detach = self._learn_model.forward(inputs, single_step=False, q_star=False)['logit'].clone().detach()
            cur_max_actions = logit_detach.argmax(dim=-1)
            inputs = {'obs': data['obs'], 'action': cur_max_actions}
            self._learn_model.reset(state=data['prev_state'][0])  # TODO(pu)
            max_action_qtot = self._learn_model.forward(inputs, single_step=False, q_star=True)['total_q']  # Q_star
            # Only if the action of each agent is optimal, then the joint action is optimal
            is_max_action = (data['action'] == cur_max_actions).min(dim=2)[0]  # shape (H,B,N) -> (H,B)
            qtot_larger = target_v > max_action_qtot
            ws = torch.full_like(td_error, alpha_to_use)
            # if y_i > Q_star or u =  u_star,  then w =1; if not, w = alpha_to_use
            ws = torch.where(is_max_action | qtot_larger, torch.ones_like(td_error), ws)

        if data['weight'] is None:
            data['weight'] = torch.ones_like(data['reward'])
        loss_weighted = (ws.detach() * td_error_per_sample * data['weight']).mean()

        # ====================
        # Q and Q_star update
        # ====================
        self._optimizer.zero_grad()
        self._optimizer_star.zero_grad()
        loss_weighted.backward(retain_graph=True)
        loss_star.backward()
        grad_norm_q = torch.nn.utils.clip_grad_norm_(
            list(self._model._q_network.parameters()) + list(self._model._mixer.parameters()),
            self._cfg.learn.clip_value
        )  # Q
        grad_norm_q_star = torch.nn.utils.clip_grad_norm_(
            list(self._model._q_network_star.parameters()) + list(self._model._mixer_star.parameters()),
            self._cfg.learn.clip_value
        )  # Q_star
        self._optimizer.step()  # Q update
        self._optimizer_star.step()  # Q_star update

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss_weighted.item(),
            'total_q': total_q.mean().item() / self._cfg.model.agent_num,
            'target_reward_total_q': target_v.mean().item() / self._cfg.model.agent_num,
            'target_total_q': target_total_q.mean().item() / self._cfg.model.agent_num,
            'grad_norm_q': grad_norm_q,
            'grad_norm_q_star': grad_norm_q_star,
        }

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names
        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For WQMIX, ``ding.model.template.wqmix``
        """
        return 'wqmix', ['ding.model.template.wqmix']

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'optimizer_star': self._optimizer_star.state_dict(),
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
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._optimizer_star.load_state_dict(state_dict['optimizer_star'])
