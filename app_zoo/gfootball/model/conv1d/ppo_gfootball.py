import math
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
import torch.nn as nn
import numpy as np

from nervex.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, Adder
from nervex.torch_utils import Adam, to_device, one_hot
from nervex.model import model_wrap
from nervex.data import default_collate, default_decollate
from nervex.utils import POLICY_REGISTRY
from nervex.policy.base_policy import Policy
from nervex.policy.common_utils import default_preprocess_learn
try:
    from app_zoo.gfootball.model.bots import FootballRuleBaseModel, FootballKaggle5thPlaceModel
except ImportError:
    FootballRuleBaseModel, FootballKaggle5thPlaceModel = None, None


@POLICY_REGISTRY.register('PPO_Gfootball')
class PPOPolicy(Policy):
    r"""
    Overview:
        Policy class of PPO for Gfootball
    Interface:
        __init__, set_setting, __repr__, state_dict_handle
    Property:
        learn_mode, collect_mode, eval_mode
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to use nstep_return for value loss
        nstep_return=False,
        nstep=3,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=5,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0001,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            ignore_done=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
            collector=dict(type='sample', ),
        ),
        eval=dict(),
        # Although ppo is an on-policy algorithm, nervex reuses the buffer mechanism, and clear buffer after update.
        # Note replay_buffer_size must be greater than n_sample.
        other=dict(replay_buffer=dict(
            type='naive',
            replay_buffer_size=1000,
        ), ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init optimizers, algorithm config, main and target models.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPO"
        # Optimizer
        self._optimizer = Adam(self._model.parameters(),
                               lr=self._cfg.learn.learning_rate)
        self._learn_model = model_wrap(
            self._model, wrapper_name='hidden_state', state_num=256, save_prev_state=False)

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._nstep = self._cfg.nstep
        self._nstep_return = self._cfg.nstep_return
        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        data = default_preprocess_learn(
            data, ignore_done=self._cfg.learn.ignore_done, use_nstep=self._nstep_return)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================

        self._learn_model.train()
        # normal ppo
        if not self._nstep_return:
            output = self._learn_model.forward(
                data['obs'], mode='compute_actor_critic')
            adv = data['adv']
            if self._adv_norm:
                # Normalize advantage in a total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            return_ = data['value'] + adv
            # Calculate ppo error
            ppodata = ppo_data(
                output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, return_,
                data['weight']
            )
            ppo_loss, ppo_info = ppo_error(ppodata, self._clip_ratio)
            wv, we = self._value_weight, self._entropy_weight
            total_loss = ppo_loss.policy_loss + wv * \
                ppo_loss.value_loss - we * ppo_loss.entropy_loss

        else:
            raise NotImplementedError

        # ====================
        # PPO update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': ppo_loss.policy_loss.item(),
            'value_loss': ppo_loss.value_loss.item(),
            'entropy_loss': ppo_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'approx_kl': ppo_info.approx_kl,
            'clipfrac': ppo_info.clipfrac,
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
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(
            model_wrap(self._model, wrapper_name='hidden_state',
                       state_num=256, save_prev_state=False),
            wrapper_name='multinomial_sample')
        self._collect_model.reset()
        self._adder = Adder(self._cuda, self._unroll_len)
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._nstep = self._cfg.nstep
        self._nstep_return = self._cfg.nstep_return

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': model_output['logit'],
            'action': model_output['action'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        data = self._adder.get_gae_with_default_last_value(
            data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
        )
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(
            self._model, wrapper_name='hidden_state', state_num=256, save_prev_state=False)
        self._eval_model.train()
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
        if self._use_cuda:
            data = to_device(data, self._device)
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    # TODO different collect model and learn model
    def default_model(self) -> Tuple[str, List[str]]:
        return 'Conv1D', ['app_zoo.gfootball.model.conv1d.conv1d']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return super()._monitor_vars_learn() + [
            'policy_loss', 'value_loss', 'entropy_loss', 'adv_abs_max', 'approx_kl', 'clipfrac'
        ]
