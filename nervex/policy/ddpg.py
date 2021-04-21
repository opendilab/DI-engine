from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import numpy as np

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, Adder
from nervex.model import QAC
from nervex.armor import Armor
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('ddpg')
class DDPGPolicy(Policy):
    r"""
    Overview:
        Policy class of DDPG and TD3 algorithm. Since DDPG and TD3 share many common things, this Policy supports
        both algorithms. You can change ``_actor_update_freq``, ``_use_twin_critic`` and noise in armor plugin to
        switch algorithm.
    Property:
        learn_mode, collect_mode, eval_mode
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init actor and critic optimizers, algorithm config, main and target armors.
        """
        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._use_reward_batch_norm = self._cfg.get('use_reward_batch_norm', False)

        # algorithm config
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._gamma = algo_cfg.discount_factor
        self._actor_update_freq = algo_cfg.actor_update_freq
        self._use_twin_critic = algo_cfg.use_twin_critic  # True for TD3, False for DDPG

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
        self._armor.reset()
        self._armor.target_reset()

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
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.get('use_priority', False),
            ignore_done=self._cfg.learn.get('ignore_done', False),
            use_nstep=False
        )
        if self._use_cuda:
            data = to_device(data, self._device)
        # ====================
        # critic learn forward
        # ====================
        self._armor.model.train()
        self._armor.target_model.train()
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._use_reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._armor.forward(data, param={'mode': 'compute_q'})['q_value']
        q_value_dict = {}
        if self._use_twin_critic:
            q_value_dict['q_value'] = q_value[0].mean()
            q_value_dict['q_value_twin'] = q_value[1].mean()
        else:
            q_value_dict['q_value'] = q_value.mean()
        # target q value. SARSA: first predict next action, then calculate next q value
        with torch.no_grad():
            next_action = self._armor.target_forward(next_obs, param={'mode': 'compute_action'})['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._armor.target_forward(next_data, param={'mode': 'compute_q'})['q_value']
        if self._use_twin_critic:
            # TD3: two critic networks
            target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
            # network1
            td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
            critic_loss, td_error_per_sample1 = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
            # network2(twin network)
            td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
            critic_twin_loss, td_error_per_sample2 = v_1step_td_error(td_data_twin, self._gamma)
            loss_dict['critic_twin_loss'] = critic_twin_loss
            td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
        else:
            # DDPG: single critic network
            td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
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
            actor_loss = -self._armor.forward(data['obs'], param={'mode': 'optimize_actor'})['q_value'].mean()
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

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._model.state_dict(),
            'optimizer_actor': self._optimizer_actor.state_dict(),
            'optimizer_critic': self._optimizer_critic.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._model.load_state_dict(state_dict['model'])
        self._optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self._optimizer_critic.load_state_dict(state_dict['optimizer_critic'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
        """
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
        self._collect_armor.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._use_cuda:
            data = to_device(data, self._device)
        self._collect_armor.model.eval()
        with torch.no_grad():
            output = self._collect_armor.forward(data, param={'mode': 'compute_action'})
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor. Unlike learn and collect armor, eval armor does not need noise.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.reset()

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
        self._eval_armor.model.eval()
        with torch.no_grad():
            output = self._eval_armor.forward(data, param={'mode': 'compute_action'})
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qac', ['nervex.model.qac.q_ac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = [
            'cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'q_value_twin',
            'action'
        ]
        if self._use_twin_critic:
            ret += ['critic_twin_loss']
        return ret
