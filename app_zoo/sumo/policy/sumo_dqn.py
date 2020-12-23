from typing import Dict, Any
import torch
from nervex.rl_utils import q_1step_td_data, q_1step_td_error
from nervex.policy import DQNPolicy, register_policy


class SumoDQNPolicy(DQNPolicy):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        q_value = self._agent.forward(data['obs'])['logit']
        target_q_value = self._agent.target_forward(data['next_obs'])['logit']
        if isinstance(q_value, torch.Tensor):
            td_data = q_1step_td_data(
                q_value, target_q_value, data['action'], data['reward'], data['done'], data['weight']
            )
            loss = q_1step_td_error(td_data, self._gamma)
        else:
            tl_num = len(q_value)
            loss = []
            for i in range(tl_num):
                td_data = q_1step_td_data(
                    q_value[i], target_q_value[i], data['action'][i], data['reward'], data['done'], data['weight']
                )
                loss.append(q_1step_td_error(td_data, self._gamma))
            loss = sum(loss) / (len(loss) + 1e-8)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
        }


register_policy('sumo_dqn', SumoDQNPolicy)
