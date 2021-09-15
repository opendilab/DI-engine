from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple

from ding.torch_utils import Adam, to_device
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, v_nstep_td_data, v_nstep_td_error
from ding.utils import POLICY_REGISTRY
from ding.policy import PPOOffPolicy
from ding.policy.common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('md_ppo_offpolicy')
class MultiDiscretePPOOffPolicy(PPOOffPolicy):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:

        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=self._nstep_return)
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # PPO forward
        # ====================

        self._learn_model.train()
        # normal ppo
        if not self._nstep_return:
            output = self._learn_model.forward(data['obs'], mode='compute_actor_critic')
            adv = data['adv']
            return_ = data['value'] + adv
            if self._adv_norm:
                # Normalize advantage in a total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            if isinstance(data['logit'], list):
                action_num = len(data['logit'])
                loss, info = [], []
                for i in range(action_num):
                    data_ = ppo_data(
                        output['logit'][i], data['logit'][i], data['action'][i], output['value'], data['value'], adv,
                        return_, data['weight']
                    )
                    ppo_loss, ppo_info = ppo_error(data_, self._clip_ratio)
                    loss.append(ppo_loss)
                    info.append(ppo_info)
                policy_loss = sum([item.policy_loss for item in loss]) / action_num
                value_loss = sum([item.value_loss for item in loss]) / action_num
                entropy_loss = sum([item.entropy_loss for item in loss]) / action_num
            else:
                # Calculate ppo error
                ppodata = ppo_data(
                    output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, return_,
                    data['weight']
                )
                ppo_loss, ppo_info = ppo_error(ppodata, self._clip_ratio)
            wv, we = self._value_weight, self._entropy_weight
            total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

        else:
            output = self._learn_model.forward(data['obs'], mode='compute_actor')
            adv = data['adv']
            if self._adv_norm:
                # Normalize advantage in a total train_batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            if isinstance(data['logit'], list):
                action_num = len(data['logit'])
                loss, info = [], []
                for i in range(action_num):
                    ppodata = ppo_policy_data(
                        output['logit'][i], data['logit'][i], data['action'][i], adv, data['weight']
                    )
                    ppo_policy_loss, ppo_info = ppo_policy_error(ppodata, self._clip_ratio)
                    loss.append(ppo_policy_loss)
                    info.append(ppo_info)
                ppo_policy_loss = sum([item.policy_loss for item in loss]) / action_num
                entropy_loss = sum([item.entropy_loss for item in loss]) / action_num

            else:
                # Calculate ppo error
                ppodata = ppo_policy_data(output['logit'], data['logit'], data['action'], adv, data['weight'])
                ppo_policy_loss, ppo_info = ppo_policy_error(ppodata, self._clip_ratio)
            wv, we = self._value_weight, self._entropy_weight
            next_obs = data.get('next_obs')
            value_gamma = data.get('value_gamma')
            reward = data.get('reward')
            # current value
            value = self._learn_model.forward(data['obs'], mode='compute_critic')
            # target value
            next_data = {'obs': next_obs}
            target_value = self._learn_model.forward(next_data['obs'], mode='compute_critic')
            # TODO what should we do here to keep shape
            assert self._nstep > 1
            td_data = v_nstep_td_data(
                value['value'], target_value['value'], reward.t(), data['done'], data['weight'], value_gamma
            )
            # calculate v_nstep_td critic_loss
            critic_loss, td_error_per_sample = v_nstep_td_error(td_data, self._gamma, self._nstep)
            ppo_loss_data = namedtuple('ppo_loss', ['policy_loss', 'value_loss', 'entropy_loss'])
            ppo_loss = ppo_loss_data(ppo_policy_loss.policy_loss, critic_loss, ppo_policy_loss.entropy_loss)
            total_loss = ppo_policy_loss.policy_loss + wv * critic_loss - we * ppo_policy_loss.entropy_loss

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
