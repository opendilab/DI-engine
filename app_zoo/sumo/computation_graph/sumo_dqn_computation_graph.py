import torch

from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import q_1step_td_data, q_1step_td_error
from nervex.worker import BaseAgent


class SumoDqnGraph(BaseCompGraph):
    """
    Overview: Double DQN with eps-greedy
    """

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.dqn.discount_factor

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data['reward']

        action = data['action']
        done = data['done'].float()
        weights = data.get('IS', None)

        q_value = agent.forward(obs)['logit']
        if agent.is_double:
            target_q_value = agent.target_forward(next_obs)['logit']
        else:
            target_q_value = agent.forward(next_obs)['logit']
        if isinstance(q_value, torch.Tensor):
            data = q_1step_td_data(q_value, target_q_value, action, reward, done)
            loss = q_1step_td_error(data, self._gamma, weights)
        else:
            tl_num = len(q_value)
            loss = []
            for i in range(tl_num):
                data = q_1step_td_data(q_value[i], target_q_value[i], action[i], reward, done)
                loss.append(q_1step_td_error(data, self._gamma, weights))
            loss = sum(loss) / (len(loss) + 1e-8)
        if agent.is_double:
            agent.target_update(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self):
        return "Double DQN for SUMOWJ# multi-traffic-light env"
