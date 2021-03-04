from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import numpy as np

from nervex.torch_utils import Adam
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, Adder
from nervex.model import QAC
from nervex.armor import Armor
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class ATOCPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of ATOC algorithm.
    Interface:
        __init__, set_setting, __repr__, state_dict_handle
    Property:
        learn_mode, collect_mode, eval_mode, command_mode
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init actor and critic optimizers, algorithm config, main and target armors.
        """
        # algorithm config
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._use_communication = algo_cfg.use_communication
        self._gamma = algo_cfg.discount_factor
        self._actor_update_freq = algo_cfg.actor_update_freq
        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model._actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_critic = Adam(
            self._model._critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            weight_decay=self._cfg.learn.weight_decay
        )
        if self._use_communication:
            self._optimizer_actor_attention = Adam(
                self._model._actor._attention.parameters(),
                lr=self._cfg.learn.learning_rate_actor,
                weight_decay=self._cfg.learn.weight_decay
            )
        self._use_reward_batch_norm = self._cfg.get('use_reward_batch_norm', False)

        # main and target armors
        self._armor = Armor(self._model)
        self._armor.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_theta})
        if algo_cfg.use_noise:
            self._armor.add_plugin(
                'target',
                'action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': algo_cfg.noise_sigma
                },
                noise_range=algo_cfg.noise_range,
            )
        self._armor.add_plugin('main', 'grad', enable_grad=True)
        self._armor.add_plugin('target', 'grad', enable_grad=False)
        self._armor.mode(train=True)
        self._armor.target_mode(train=True)
        self._armor.reset()
        self._armor.target_reset()

        self._learn_setting_set = {}
        self._forward_learn_cnt = 0  # count iterations

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.
        """
        loss_dict = {}
        # ====================
        # critic learn forward
        # ====================
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._use_reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._armor.forward(data, param={'mode': 'compute_q'})['q_value']
        q_value_dict = {}
        q_value_dict['q_value'] = q_value.mean()
        # target q value. SARSA: first predict next action, then calculate next q value
        next_data = {'obs': next_obs}
        next_action = self._armor.target_forward(next_data, param={'mode': 'compute_action'})['action']
        next_data['action'] = next_action
        target_q_value = self._armor.target_forward(next_data, param={'mode': 'compute_q'})['q_value']
        # td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
        # TODO what should we do here to keep shape
        td_data = v_1step_td_data(q_value.mean(-1), target_q_value.mean(-1), reward, data['done'], data['weight'])
        critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
        loss_dict['critic_loss'] = critic_loss
        # ================
        # critic update
        # ================
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        # ===============================
        # actor learn forward and update
        # ===============================
        # actor updates every ``self._actor_update_freq`` iters
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            if self._use_communication:
                inputs = self._armor.forward(
                    {'obs': data['obs']}, param={
                        'mode': 'compute_action',
                        'get_delta_q': False
                    }
                )
                inputs['delta_q'] = data['delta_q']
                attention_loss = -self._armor.forward(
                    inputs, param={'mode': 'optimize_actor_attention'}
                )['actor_attention_loss'].mean()
                loss_dict['attention_loss'] = attention_loss
                self._optimizer_actor_attention.zero_grad()
                attention_loss.backward()
                self._optimizer_actor_attention.step()
            actor_loss = -self._armor.forward(data, param={'mode': 'optimize_actor'})['q_value'].mean()
            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        # =============
        # after update
        # =============
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr_actor': self._optimizer_actor.defaults['lr'],
            'cur_lr_critic': self._optimizer_critic.defaults['lr'],
            # 'q_value': np.array(q_value).mean(),
            'action': data.get('action').mean(),
            'priority': td_error_per_sample.abs().tolist(),
            **loss_dict,
            **q_value_dict,
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
        """
        self._traj_len = self._cfg.collect.traj_len
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        # collect armor
        self._collect_armor = Armor(self._model)
        algo_cfg = self._cfg.collect.algo
        self._collect_armor.add_plugin(
            'main',
            'action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': algo_cfg.noise_sigma
            },
            noise_range=None,  # no noise clip in actor
        )
        self._collect_armor.add_plugin('main', 'grad', enable_grad=False)
        self._collect_armor.mode(train=False)
        self._collect_armor.reset()
        self._collect_setting_set = {}

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        output = self._collect_armor.forward(data, param={'mode': 'compute_action'})
        return output

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        if self._use_communication:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': armor_output['action'],
                'delta_q': armor_output['delta_q'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        else:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': armor_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        return transition

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        # adder is defined in _init_collect
        data = self._adder.get_traj(traj_cache, self._traj_len)
        if self._use_communication:
            delta_q_batch = [d['delta_q'] for d in data]
            delta_min = torch.stack(delta_q_batch).min()
            delta_max = torch.stack(delta_q_batch).max()
            for i in range(len(data)):
                data[i]['delta_q'] = (data[i]['delta_q'] - delta_min) / (delta_max - delta_min + 1e-8)
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor. Unlike learn and collect armor, eval armor does not need noise.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin('main', 'grad', enable_grad=False)
        self._eval_armor.mode(train=False)
        self._eval_armor.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode, similar to ``self._forward_collect``.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        output = self._eval_armor.forward(data, param={'mode': 'compute_action'})
        return output

    def _init_command(self) -> None:
        r"""
        Overview:
            Command mode init method. Called by ``self.__init__``.
        """
        pass

    def default_model(self) -> Tuple[str, List[str]]:
        return 'atoc', ['nervex.model.atoc.atoc_network']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = [
            'cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'attention_loss', 'total_loss', 'q_value',
            'action'
        ]
        return ret


register_policy('atoc', ATOCPolicy)
