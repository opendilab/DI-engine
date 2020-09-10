import numpy as np
from collections import namedtuple
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from nervex.model.sumo_dqn.sumo_dqn_network import FCDQN
from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.worker.agent.sumo_dqn_agent import SumoDqnLearnerAgent
from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import td_data, one_step_td_error


class SumoDqnGraph(BaseCompGraph):
    """
    Overview: Double DQN with eps-greedy
    """
    def __init__(self, cfg: dict) -> None:
        sumo_env = SumoWJ3Env({})
        model = FCDQN(sumo_env.info().obs_space.shape, [v for k, v in sumo_env.info().act_space.shape.items()])
        if cfg.use_cuda:
            model.cuda()
        self.is_double = cfg.dqn.is_double
        self.agent = SumoDqnLearnerAgent(model, plugin_cfg={'is_double': self.is_double})
        self.agent.mode(train=True)
        if self.is_double:
            self.agent.target_mode(train=True)

        self._gamma = cfg.dqn.discount_factor
        self.update_target_freq = cfg.dqn.update_target_freq
        self.iter_count = 0
        self.reward_weights = cfg.reward_weights

    def forward(self, data: dict) -> dict:
        self.iter_count += 1
        obs_batch = data.get('obs')
        nextobs_batch = data.get('next_obs')
        reward = data.get('reward')
        if len(self.reward_weights) >= 2:
            reward = reduce(
                lambda x, y: reward[x] * self.reward_weights[x] + reward[y] * self.reward_weights[y],
                self.reward_weights.keys()
            )
        else:
            reward = reward[list(self.reward_weights.keys())[0]]

        action = data.get('action')
        terminate = data.get('done')
        weights = data.get('weights', None)

        q_value = self.agent.forward(obs_batch)
        next_q_value = self.agent.forward(nextobs_batch)
        if self.is_double:
            target_q_value = self.agent.target_forward(nextobs_batch)
        else:
            target_q_value = next_q_value

        tl_num = len(q_value)
        loss = []
        for i in range(tl_num):
            data = td_data(q_value[i], target_q_value[i], action[i], reward, terminate)
            loss.append(one_step_td_error(data, self._gamma, weights))
        loss = sum(loss) / (len(loss) + 1e-8)
        if self.is_double and self.iter_count % self.update_target_freq == 0:
            self.agent.update_target_network(self.agent.state_dict()['model'])
        return {'total_loss': loss}

    def register_stats(self, variable_record, tb_logger):
        variable_record.register_var('total_loss')
        tb_logger.register_var('total_loss')

    def sync_gradients(self):
        pass

    def __repr__(self):
        return "Double DQN for SUMOWJ# multi-traffic-light env"
