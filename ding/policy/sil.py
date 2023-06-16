from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np

from ding.rl_utils import sil_data, sil_error, a2c_data, a2c_error, ppo_data, ppo_error, ppo_policy_error,\
    ppo_policy_data, get_gae_with_default_last_value, get_train_sample, gae, gae_data, ppo_error_continuous, get_gae
from ding.torch_utils import Adam, to_device, to_dtype, unsqueeze
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .ppo import PPOPolicy
from .a2c import A2CPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('sil_a2c')
class SILA2CPolicy(A2CPolicy):
    r"""
    Overview:
        Policy class of SIL algorithm combined with A2C, paper link: https://arxiv.org/abs/1806.05635
    """
    config = dict(
        # (string) RL policy register name (refer to function "register_policy").
        type='sil_a2c',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether to use on-policy training pipeline(behaviour policy and training policy are the same)
        on_policy=True,
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of epochs to use SIL loss to update the policy.
        sil_update_per_collect=1,
        learn=dict(
            update_per_collect=1,  # fixed value, this line should not be modified by users
            batch_size=64,
            learning_rate=0.001,
            # (List[float])
            betas=(0.9, 0.999),
            # (float)
            eps=1e-8,
            # (float)
            grad_norm=0.5,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            ignore_done=False,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            # n_sample=80,
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
        ),
        eval=dict(),
    )

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','adv']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        # Extract off-policy data
        data_sil = data['replay_data']
        data_sil = [
            default_preprocess_learn(data_sil[i], ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
            for i in range(len(data_sil))
        ]
        # Extract on-policy data
        data_onpolicy = data['new_data']
        for i in range(len(data_onpolicy)):
            data_onpolicy[i] = {k: data_onpolicy[i][k] for k in ['obs', 'adv', 'value', 'action', 'done']}
        data_onpolicy = default_preprocess_learn(
            data_onpolicy, ignore_done=self._cfg.learn.ignore_done, use_nstep=False
        )
        data_onpolicy['weight'] = None
        # Put data to correct device.
        if self._cuda:
            data_onpolicy = to_device(data_onpolicy, self._device)
            data_sil = to_device(data_sil, self._device)
        self._learn_model.train()

        for batch in split_data_generator(data_onpolicy, self._cfg.learn.batch_size, shuffle=True):
            # forward
            output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')

            adv = batch['adv']
            return_ = batch['value'] + adv
            if self._adv_norm:
                # norm adv in total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            error_data = a2c_data(output['logit'], batch['action'], output['value'], adv, return_, batch['weight'])

            # Calculate A2C loss
            a2c_loss = a2c_error(error_data)
            wv, we = self._value_weight, self._entropy_weight
            a2c_total_loss = a2c_loss.policy_loss + wv * a2c_loss.value_loss - we * a2c_loss.entropy_loss

            # ====================
            # A2C-learning update
            # ====================

            self._optimizer.zero_grad()
            a2c_total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self._learn_model.parameters()),
                max_norm=self._grad_norm,
            )
            self._optimizer.step()

        for batch in data_sil:
            # forward
            output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')

            adv = batch['adv']
            return_ = batch['value'] + adv
            if self._adv_norm:
                # norm adv in total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            error_data = sil_data(output['logit'], batch['action'], output['value'], adv, return_, batch['weight'])

            # Calculate SIL loss
            sil_loss, sil_info = sil_error(error_data)
            wv = self._value_weight
            sil_total_loss = sil_loss.policy_loss + wv * sil_loss.value_loss

            # ====================
            # SIL-learning update
            # ====================

            self._optimizer.zero_grad()
            sil_total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self._learn_model.parameters()),
                max_norm=self._grad_norm,
            )
            self._optimizer.step()

        # =============
        # after update
        # =============
        # only record last updates information in logger
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': sil_total_loss.item() + a2c_total_loss.item(),
            'sil_total_loss': sil_total_loss.item(),
            'a2c_total_loss': a2c_total_loss.item(),
            'sil_policy_loss': sil_loss.policy_loss.item(),
            'a2c_policy_loss': a2c_loss.policy_loss.item(),
            'sil_value_loss': sil_loss.value_loss.item(),
            'a2c_value_loss': a2c_loss.value_loss.item(),
            'a2c_entropy_loss': a2c_loss.entropy_loss.item(),
            'policy_clipfrac': sil_info.policy_clipfrac,
            'value_clipfrac': sil_info.value_clipfrac,
            'adv_abs_max': adv.abs().max().item(),
            'grad_norm': grad_norm,
        }

    def _monitor_vars_learn(self) -> List[str]:
        return list(
            set(
                super()._monitor_vars_learn() + [
                    'sil_policy_loss', 'sil_value_loss', 'a2c_total_loss', 'sil_total_loss', 'policy_clipfrac',
                    'value_clipfrac'
                ]
            )
        )


@POLICY_REGISTRY.register('sil_ppo')
class SILPPOPolicy(PPOPolicy):
    r"""
    Overview:
        Policy class of SIL algorithm combined with PPO, paper link: https://arxiv.org/abs/1806.05635
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='sil_ppo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update due to priority.
        # If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to recompurete advantages in each iteration of on-policy PPO
        recompute_adv=True,
        # (str) Which kind of action space used in PPOPolicy, ['discrete', 'continuous', 'hybrid']
        action_space='discrete',
        # (bool) Whether to use nstep return to calculate value target, otherwise, use return = adv + value
        nstep_return=False,
        # (bool) Whether to enable multi-agent training, i.e.: MAPPO
        multi_agent=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        # (int) Number of epochs to use SIL loss to update the policy.
        sil_update_per_collect=1,
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=True,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
            ignore_done=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),
    )

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
        # Extract off-policy data
        data_sil = data['replay_data']
        data_sil = [
            default_preprocess_learn(data_sil[i], ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
            for i in range(len(data_sil))
        ]
        # Extract on-policy data
        data_onpolicy = data['new_data']
        for i in range(len(data_onpolicy)):
            data_onpolicy[i] = {k: data_onpolicy[i][k] for k in ['obs', 'adv', 'value', 'action', 'done']}
        data_onpolicy = default_preprocess_learn(
            data_onpolicy, ignore_done=self._cfg.learn.ignore_done, use_nstep=False
        )
        data_onpolicy['weight'] = None
        # Put data to correct device.
        if self._cuda:
            data_onpolicy = to_device(data_onpolicy, self._device)
            data_sil = to_device(data_sil, self._device)
        self._learn_model.train()
        data_onpolicy['obs'] = to_dtype(data_onpolicy['obs'], torch.float32)
        if 'next_obs' in data_onpolicy:
            data_onpolicy['next_obs'] = to_dtype(data_onpolicy['next_obs'], torch.float32)
        data_sil['obs'] = to_dtype(data_sil['obs'], torch.float32)
        if 'next_obs' in data_sil:
            data_sil['next_obs'] = to_dtype(data_sil['next_obs'], torch.float32)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:  # calculate new value using the new updated value network
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    traj_flag = data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], traj_flag)
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                    unnormalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = unnormalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(unnormalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = unnormalized_returns

            else:  # don't recompute adv
                if self._value_norm:
                    unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = unnormalized_return / self._running_mean_std.std
                    self._running_mean_std.update(unnormalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                adv = batch['adv']
                if self._adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Calculate ppo error
                ppo_batch = ppo_data(
                    output['logit'], batch['logit'], batch['action'], output['value'], batch['value'], adv,
                    batch['return'], batch['weight']
                )
                ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)
                wv, we = self._value_weight, self._entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'ppo_total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clipfrac': ppo_info.clipfrac,
                }
                return_infos.append(return_info)

        return_info_real = {
            k: sum([return_infos[i][k]
                    for i in range(len(return_infos))]) / len([return_infos[i][k] for i in range(len(return_infos))])
            for k in return_infos[0].keys()
        }

        for batch in data_sil:
            # forward
            output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')

            adv = batch['adv']
            return_ = batch['value'] + adv
            if self._adv_norm:
                # norm adv in total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            error_data = sil_data(output['logit'], batch['action'], output['value'], adv, return_, batch['weight'])

            # Calculate SIL loss
            sil_loss, sil_info = sil_error(error_data)
            wv = self._value_weight
            sil_total_loss = sil_loss.policy_loss + wv * sil_loss.value_loss

            # ====================
            # SIL-learning update
            # ====================

            self._optimizer.zero_grad()
            sil_total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self._learn_model.parameters()),
                max_norm=self._grad_norm,
            )
            self._optimizer.step()

        sil_learn_info = {
            'total_loss': sil_total_loss.item() + return_info_real['ppo_total_loss'],
            'sil_total_loss': sil_total_loss.item(),
            'sil_policy_loss': sil_loss.policy_loss.item(),
            'sil_value_loss': sil_loss.value_loss.item(),
            'policy_clipfrac': sil_info.policy_clipfrac,
            'value_clipfrac': sil_info.value_clipfrac
        }

        return_info_real.update(sil_learn_info)
        return return_info_real

    def _monitor_vars_learn(self) -> List[str]:
        variables = list(
            set(
                super()._monitor_vars_learn() + [
                    'sil_policy_loss', 'sil_value_loss', 'ppo_total_loss', 'sil_total_loss', 'policy_clipfrac',
                    'value_clipfrac'
                ]
            )
        )
        return variables
