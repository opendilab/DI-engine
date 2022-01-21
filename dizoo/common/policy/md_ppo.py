from typing import List, Dict, Any, Tuple, Union
import torch

from ding.policy import PPOPolicy, PPOOffPolicy
from ding.rl_utils import ppo_data, ppo_error, gae, gae_data
from ding.utils import POLICY_REGISTRY, split_data_generator
from ding.torch_utils import to_device
from ding.policy.common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('md_ppo')
class MultiDiscretePPOPolicy(PPOPolicy):
    r"""
    Overview:
        Policy class of Multi-discrete action space PPO algorithm.
    """

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_max, adv_mean, value_max, value_mean, approx_kl, clipfrac
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], data['traj_flag'])
                    # GAE need (T, B) shape input and return (T, B) output
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)
                    # value = value[:-1]
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
                loss_list = []
                info_list = []
                action_num = len(batch['action'])
                for i in range(action_num):
                    ppo_batch = ppo_data(
                        output['logit'][i], batch['logit'][i], batch['action'][i], output['value'], batch['value'], adv,
                        batch['return'], batch['weight']
                    )
                    ppo_loss, ppo_info = ppo_error(ppo_batch, self._clip_ratio)
                    loss_list.append(ppo_loss)
                    info_list.append(ppo_info)
                avg_policy_loss = sum([item.policy_loss for item in loss_list]) / action_num
                avg_value_loss = sum([item.value_loss for item in loss_list]) / action_num
                avg_entropy_loss = sum([item.entropy_loss for item in loss_list]) / action_num
                avg_approx_kl = sum([item.approx_kl for item in info_list]) / action_num
                avg_clipfrac = sum([item.clipfrac for item in info_list]) / action_num

                wv, we = self._value_weight, self._entropy_weight
                total_loss = avg_policy_loss + wv * avg_value_loss - we * avg_entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'policy_loss': avg_policy_loss.item(),
                    'value_loss': avg_value_loss.item(),
                    'entropy_loss': avg_entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': avg_approx_kl,
                    'clipfrac': avg_clipfrac,
                }
                return_infos.append(return_info)
        return return_infos


@POLICY_REGISTRY.register('md_ppo_offpolicy')
class MultiDiscretePPOOffPolicy(PPOOffPolicy):
    r"""
    Overview:
        Policy class of Multi-discrete action space off-policy PPO algorithm.
    """

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
        assert not self._nstep_return
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=self._nstep_return)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================

        self._learn_model.train()
        # normal ppo
        output = self._learn_model.forward(data['obs'], mode='compute_actor_critic')
        adv = data['adv']
        return_ = data['value'] + adv
        if self._adv_norm:
            # Normalize advantage in a total train_batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Calculate ppo error
        loss_list = []
        info_list = []
        action_num = len(data['action'])
        for i in range(action_num):
            ppodata = ppo_data(
                output['logit'][i], data['logit'][i], data['action'][i], output['value'], data['value'], adv, return_,
                data['weight']
            )
            ppo_loss, ppo_info = ppo_error(ppodata, self._clip_ratio)
            loss_list.append(ppo_loss)
            info_list.append(ppo_info)
        avg_policy_loss = sum([item.policy_loss for item in loss_list]) / action_num
        avg_value_loss = sum([item.value_loss for item in loss_list]) / action_num
        avg_entropy_loss = sum([item.entropy_loss for item in loss_list]) / action_num
        avg_approx_kl = sum([item.approx_kl for item in info_list]) / action_num
        avg_clipfrac = sum([item.clipfrac for item in info_list]) / action_num

        wv, we = self._value_weight, self._entropy_weight
        total_loss = avg_policy_loss + wv * avg_value_loss - we * avg_entropy_loss

        # ====================
        # PPO update
        # ====================
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'adv_abs_max': adv.abs().max().item(),
            'approx_kl': avg_approx_kl,
            'clipfrac': avg_clipfrac,
        }
