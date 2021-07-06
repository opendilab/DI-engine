from typing import Dict, Any
import torch
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error
from ding.policy import DQNPolicy
from ding.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('md_dqn')
class MultiDiscreteDQNPolicy(DQNPolicy):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        assert reward.shape == (self._cfg.learn.batch_size, self._nstep), reward.shape
        reward = reward.permute(1, 0).contiguous()
        q_value = self._armor.forward(data['obs'])['logit']
        # target_q_value = self._armor.target_forward(data['next_obs'])['logit']
        target = self._armor.forward(data['next_obs'])
        target_q_value = target['logit']
        next_act = target['action']
        if isinstance(q_value, torch.Tensor):
            td_data = q_nstep_td_data(  # 'q', 'next_q', 'act', 'next_act', 'reward', 'done', 'weight'
                q_value, target_q_value, data['action'][0], next_act, reward, data['done'], data['weight']
            )
            loss, td_error_per_sample = q_nstep_td_error(td_data, self._gamma, nstep=self._nstep)
        else:
            tl_num = len(q_value)
            loss, td_error_per_sample = [], []
            for i in range(tl_num):
                td_data = q_nstep_td_data(
                    q_value[i], target_q_value[i], data['action'][i], next_act[i], reward, data['done'], data['weight']
                )
                loss_, td_error_per_sample_ = q_nstep_td_error(td_data, self._gamma, nstep=self._nstep)
                loss.append(loss_)
                td_error_per_sample.append(td_error_per_sample_.abs())
            loss = sum(loss) / (len(loss) + 1e-8)
            td_error_per_sample = sum(td_error_per_sample) / (len(td_error_per_sample) + 1e-8)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }
