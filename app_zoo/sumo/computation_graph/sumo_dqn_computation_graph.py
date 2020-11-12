import torch

from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import td_data, one_step_td_error
from nervex.worker import BaseAgent


class SumoDqnGraph(BaseCompGraph):
    """
    Overview: Double DQN with eps-greedy
    """

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.dqn.discount_factor
        self._reward_weights = cfg.reward_weights

    def get_weighted_reward(self, reward: dict) -> torch.Tensor:
        if len(self._reward_weights) >= 2:
            reward = sum(map(lambda x: reward[x] * self._reward_weights[x], self._reward_weights.keys()))
        else:
            reward = reward[list(self._reward_weights.keys())[0]]
        return reward

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = self.get_weighted_reward(data['reward'])

        action = data['action']
        done = data['done'].float()
        weights = data.get('IS', None)

        q_value = agent.forward(obs)['logit']
        if agent.is_double:
            target_q_value = agent.target_forward(next_obs)['logit']
        else:
            target_q_value = agent.forward(next_obs)['logit']
        if isinstance(q_value, torch.Tensor):
            data = td_data(q_value, target_q_value, action, reward, done)
            loss = one_step_td_error(data, self._gamma, weights)
        else:
            tl_num = len(q_value)
            loss = []
            for i in range(tl_num):
                data = td_data(q_value[i], target_q_value[i], action[i], reward, done)
                loss.append(one_step_td_error(data, self._gamma, weights))
            loss = sum(loss) / (len(loss) + 1e-8)
        if agent.is_double:
            agent.update_target_network(agent.state_dict()['model'])
        return {'total_loss': loss}

    def register_stats(self, variable_record, tb_logger):
        variable_record.register_var('total_loss')
        tb_logger.register_var('total_loss')

    def __repr__(self):
        return "Double DQN for SUMOWJ# multi-traffic-light env"
