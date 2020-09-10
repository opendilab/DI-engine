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


class SumoDqnGraph(BaseCompGraph):
    td_data = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'terminate'])

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
        self.q_function_criterion = nn.MSELoss(reduction='none')
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
            data = SumoDqnGraph.td_data(q_value[i], target_q_value[i], action[i], reward, terminate)
            loss.append(self._single_tl_dqn_loss(data, weights))
        loss = sum(loss) / (len(loss) + 1e-8)
        if self.is_double and self.iter_count % self.update_target_freq == 0:
            self.agent.update_target_network(self.agent.state_dict()['model'])
        return {'total_loss': loss}

    def _single_tl_dqn_loss(self, data, weights=None):
        q, next_q, act, reward, terminate = data
        batch_range = torch.arange(act.shape[0])
        if weights is None:
            weights = torch.ones_like(reward)

        q_s_a = q[batch_range, act]

        next_act = next_q.argmax(dim=1)
        target_q_s_a = next_q[batch_range, next_act]
        target_q_s_a = self._gamma * (1 - terminate) * target_q_s_a + reward

        return (self.q_function_criterion(q_s_a, target_q_s_a.detach()) * weights).mean()

    def register_stats(self, variable_record, tb_logger):
        variable_record.register_var('total_loss')
        tb_logger.register_var('total_loss')

    def sync_gradients(self):
        pass

    def __repr__(self):
        return "Double DQN for SUMOWJ# multi-traffic-light env"
