import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from nervex.optimizer.base_loss import BaseLoss


class SumoDqnLoss(BaseLoss):
    td_data = namedtuple('td_data', ['q', 'next_q', 'act', 'reward', 'terminate'])

    def __init__(self, agent, discount_factor: Optional[float] = 0.99, q_function_criterion=nn.MSELoss(reduction='none')):
        self.agent = agent
        self._gamma = discount_factor
        self.q_function_criterion = q_function_criterion

    def compute_loss(self, data: dict):
        # assert('q_value' in data)
        # assert('next_q_value' in data)
        # assert('target_q_value' in data)


        # assert('obs' in data)
        # assert('nextobs' in data)
        # assert('action' in data)
        # assert('reward' in data)
        # assert('terminals' in data)

        # q_value = data.get('q_value')
        # next_q_value = data.get('next_q_value')
        # target_q_value = data.get('target_q_value')
        batchs = data

        state_batch = torch.cat([torch.Tensor([x['obs']]) for x in batchs], 0)
        nextstate_batch = torch.cat([torch.Tensor([x['next_obs']]) for x in batchs], 0)
        action_batch = torch.cat([torch.LongTensor([x['acts']]) for x in batchs])
        reward_batch = torch.cat([torch.Tensor([x['rewards']]) for x in batchs])
        terminate_batch = torch.cat([torch.Tensor([x['terminals']]) for x in batchs])
        unterminate = torch.cat([torch.Tensor([1 - x['terminals']]) for x in batchs])

        # obs = data.get('obs')
        # nextobs = data.get('nextobs')
        # action = data.get('action')
        # reward = data.get('reward')
        # terminals = data.get('terminals')
        # weights = data.get('weights', None)
        weights = None

        q_value = self.agent.forward(state_batch)
        next_q_value = self.agent.forward(nextstate_batch)
        # if self.is_dobule:
        if False:
            pass
            # target_q_value = self.target_q_fuction(nextstate_batch.to(self.device))
        else:
            target_q_value = next_q_value

        reward = reward_batch
        action = action_batch
        action = list(zip(*action))
        action = [torch.stack(t) for t in action]
        terminate = terminate_batch

        tl_num = 3
        loss = []
        for i in range(tl_num):
            data = SumoDqnLoss.td_data(q_value[i], next_q_value[i], action[i], reward, terminate)
            loss.append(self._single_tl_dqn_loss(data, weights))
        loss = sum(loss) / (len(loss) + 1e-8)
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

#        q_s_a = []
#        for i in range(len(action_batch)):
#            q = []
#            for j in range(len(action_batch[i])):
#                q.append(q_value[j][i][action_batch[i][j]])
#            q_s_a.append(q)
#
#        for i in range(len(q_s_a)):
#            q_s_a[i] = torch.cat([x.reshape(1) for x in q_s_a[i]])
#
#        # q_s_a = q_value.gather(1, action.unsqueeze(1).long()).squeeze(1)
#        # q_s_a = q_value[:, action.unsqueeze(1).long()].squeeze(1)
#
#        #Get target_q_s_a
#        best_actions = []
#        for i in range(len(action_batch)):
#            best_action = []
#            for j in range(len(action_batch[i])):
#                best_action.append(torch.max(next_q_value[j][i], 0)[1])
#            best_actions.append(best_action)
#        # print(best_actions)
#        tmp_q_s_a = []
#        for i in range(len(best_actions)):
#            q = []
#            for j in range(len(best_actions[i])):
#                q.append(next_q_value[j][i][best_actions[i][j]])
#            tmp_q_s_a.append(q)
#        for i in range(len(tmp_q_s_a)):
#            tmp_q_s_a[i] = torch.cat([x.reshape(1) for x in tmp_q_s_a[i]])
#
#        t_s_a = []
#        for i in range(len(q_s_a)):
#            t = self._gamma * unterminate[i] * tmp_q_s_a[i]
#            # print(terminals[i], "+" , q_s_a[i], "=", t)
#            t_s_a.append(t)
#        tmp_q_s_a = t_s_a
#
#        t_s_a = []
#        for i in range(len(q_s_a)):
#            t = rewards[i] + tmp_q_s_a[i]
#            t_s_a.append(t)
#
#        target_q_s_a = t_s_a
#
#        # target_q_s_a = rewards + self._gamma * (1 - terminals) * \
#        #     target_q_value.gather(1, torch.max(next_q_value, 1)[1].unsqueeze(1)).squeeze(1).to(self.device)
#        if weights is not None:
#            q_function_loss = self.q_function_criterion(torch.cat(q_s_a), torch.cat(target_q_s_a).detach()) * weights
#        else:
#            q_function_loss = self.q_function_criterion(torch.cat(q_s_a), torch.cat(target_q_s_a).detach())
#        var_items = {}
#        var_items['total_loss'] = q_function_loss
#        return var_items

    def register_log(self, variable_record, tb_logger):
        pass
