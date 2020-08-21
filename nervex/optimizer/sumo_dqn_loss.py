import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from nervex.optimizer.base_loss import BaseLoss


class SumoDqnLoss(BaseLoss):
    td_data = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'terminate'])

    # def __init__(
    #     self, agent, discount_factor: Optional[float] = 0.99, q_function_criterion=nn.MSELoss(reduction='none')
    # ):
    def __init__(
        self, agent, cfg, q_function_criterion=nn.MSELoss(reduction='none')
    ):
        self.agent = agent
        # self._gamma = discount_factor
        self._gamma = cfg.dqn.discount_factor
        self.q_function_criterion = q_function_criterion
        self.update_target_freq = cfg.dqn.update_target_freq
        self.iter_count = 0
        self.is_double = cfg.dqn.is_double

    def compute_loss(self, data: dict):
        self.iter_count += 1
        state_batch = data.get('state')
        nextstate_batch = data.get('next_state')
        reward = data.get('reward')
        action = data.get('action')
        terminate = data.get('terminate')

        weights = None

        q_value = self.agent.forward(state_batch)
        next_q_value = self.agent.forward(nextstate_batch)
        if self.is_double:
            target_q_value = self.agent.target_forward(nextstate_batch)
        else:
            target_q_value = next_q_value

        tl_num = len(q_value)
        loss = []
        for i in range(tl_num):
            data = SumoDqnLoss.td_data(q_value[i], next_q_value[i], action[i], reward, terminate)
            loss.append(self._single_tl_dqn_loss(data, weights))
        loss = sum(loss) / (len(loss) + 1e-8)
        if self.iter_count % self.update_target_freq == 0:
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

    def register_log(self, variable_record, tb_logger):
        pass
