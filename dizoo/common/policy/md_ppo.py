from typing import List, Dict, Any, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.rl_utils import ppo_data, ppo_error
from ding.policy import PPOPolicy
from ding.utils import POLICY_REGISTRY
from ding.model.actor_critic.value_ac import ValueAC


class SepValueAC(ValueAC):

    def _setup_encoder(self) -> torch.nn.Module:
        return nn.Identity()


@POLICY_REGISTRY.register('md_ppo')
class MultiDiscretePPOPolicy(PPOPolicy):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        output = self._armor.forward(data['obs'], param={'mode': 'compute_action_value'})
        adv = data['adv']
        if self._use_adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return_ = data['value'] + adv
        action_num = len(data['logit'])
        loss, info = [], []
        for i in range(action_num):
            data_ = ppo_data(
                output['logit'][i], data['logit'][i], data['action'][i], output['value'], data['value'], adv, return_,
                data['weight']
            )
            ppo_loss, ppo_info = ppo_error(data_, self._clip_ratio)
            loss.append(ppo_loss)
            info.append(ppo_info)
        policy_loss = sum([item.policy_loss for item in loss]) / action_num
        value_loss = sum([item.value_loss for item in loss]) / action_num
        entropy_loss = sum([item.entropy_loss for item in loss]) / action_num
        wv, we = self._value_weight, self._entropy_weight
        total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

        approx_kl = sum([item.approx_kl for item in info]) / action_num
        clipfrac = sum([item.clipfrac for item in info]) / action_num

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'approx_kl': approx_kl,
            'clipfrac': clipfrac,
        }
