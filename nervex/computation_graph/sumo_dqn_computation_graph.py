import numpy as np
from collections import namedtuple
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from nervex.worker import BaseAgent
from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import td_data, one_step_td_error


class SumoDqnGraph(BaseCompGraph):
    """
    Overview: Double DQN with eps-greedy
    """

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.dqn.discount_factor
        self._reward_weights = cfg.reward_weights

    def get_weighted_reward(self, reward: dict) -> torch.Tensor:
        if len(self._reward_weights) >= 2:
            reward = reduce(
                lambda x, y: reward[x] * self._reward_weights[x] + reward[y] * self._reward_weights[y],
                self._reward_weights.keys()
            )
        else:
            reward = reward[list(self._reward_weights.keys())[0]]
        return reward

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs_batch = data.get('obs')
        nextobs_batch = data.get('next_obs')
        reward = self.get_weighted_reward(data['reward']).squeeze(1)

        action = data['action']
        terminate = data['done'].float()
        weights = data.get('IS', None)

        q_value = agent.forward(obs_batch)
        if agent.is_double:
            target_q_value = agent.target_forward(nextobs_batch)
        else:
            target_q_value = agent.forward(nextobs_batch)

        tl_num = len(q_value)
        loss = []
        for i in range(tl_num):
            data = td_data(q_value[i], target_q_value[i], action[i], reward, terminate)
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
