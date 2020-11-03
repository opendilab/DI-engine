import torch

from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import td_data, one_step_td_error
from nervex.worker import BaseAgent


class GfootballIqlGraph(BaseCompGraph):
    """
    Overview: Double DQN with eps-greedy
    """

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.dqn.discount_factor

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs_batch = data.get('obs')
        nextobs_batch = data.get('next_obs')
        reward = data['reward'].squeeze(1)

        action = data['action'].squeeze(1)
        terminate = data['done'].float()
        weights = data.get('IS', None)

        q_value = agent.forward(obs_batch)
        if agent.is_double:
            target_q_value = agent.target_forward(nextobs_batch)
        else:
            target_q_value = agent.forward(nextobs_batch)
        data = td_data(q_value, target_q_value, action, reward, terminate)
        loss = one_step_td_error(data, self._gamma, weights)
        if agent.is_double:
            agent.update_target_network(agent.state_dict()['model'])
        return {'total_loss': loss}

    def register_stats(self, variable_record, tb_logger):
        variable_record.register_var('total_loss')
        tb_logger.register_var('total_loss')

    def __repr__(self):
        return "IQL consisting of N_PLAYER Double DQNs for Gfootball env"
