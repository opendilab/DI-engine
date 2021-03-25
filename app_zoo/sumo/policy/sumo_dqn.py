from typing import Dict, Any
import torch
from nervex.rl_utils import q_1step_td_data, q_1step_td_error
from nervex.policy import DQNPolicy
from nervex.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('sumo_dqn')
class SumoDQNPolicy(DQNPolicy):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        q_value = self._armor.forward(data['obs'])['logit']
        # target_q_value = self._armor.target_forward(data['next_obs'])['logit']
        target = self._armor.forward(data['next_obs'])
        target_q_value = target['logit']
        next_act = target['action']
        if isinstance(q_value, torch.Tensor):
            td_data = q_1step_td_data(  # 'q', 'next_q', 'act', 'next_act', 'reward', 'done', 'weight'
                q_value, target_q_value, data['action'][0], next_act, data['reward'], data['done'], data['weight']
            )
            loss = q_1step_td_error(td_data, self._gamma)
        else:
            tl_num = len(q_value)
            loss = []
            for i in range(tl_num):
                td_data = q_1step_td_data(
                    q_value[i], target_q_value[i], data['action'][i], next_act[i], data['reward'], data['done'],
                    data['weight']
                )
                loss.append(q_1step_td_error(td_data, self._gamma))
            loss = sum(loss) / (len(loss) + 1e-8)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }
