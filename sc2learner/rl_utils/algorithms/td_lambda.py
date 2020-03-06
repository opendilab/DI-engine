'''
Copyright 2020 Sensetime X-lab. All Rights Reserved
Main Function:
    1. Define TD(temporal difference) lambda loss
'''
import torch
import torch.nn.functional as F
from .base import BaseRLAlgorithm


class TDLambda(BaseRLAlgorithm):
    '''
        Overview: Temporal difference loss between state values and lambda returns
        Note: refer to https://github.com/deepmind/trfl/blob/2c07ac22512a16715cc759f0072be43a5d12ae45/trfl/value_ops.py
    '''
    # overwrite
    def __init__(self, cfg):
        self.lambda_ = cfg.lambda_

    # overwrite
    def __call__(self, inputs):
        '''
        Overview: TD lambda loss
        Arguments: (key-value of inputs)
            - state_values (:obj:`torch.Tensor`) shape[T, B], timestep and batch size
            - rewards (:obj:`torch.Tensor`) shape[T, B]
            - discounts (:obj:`torch.Tensor`) shape[T, B]
        Returns: (key-value of returns)
            - td_lambda_loss (:obj:`torch.Tensor`) td lambda loss tensor
        '''
        state_values = inputs['state_values']
        rewards = inputs['rewards']
        discounts = inputs['discounts']

        # use 1~T steps to calculate returns, and 0~T-1 as baselines
        returns = self._lambda_returns(state_values[1:], rewards, discounts)
        # detach from the computational graph, stop gradients flowing through returns parts
        returns = returns.detach()
        loss = 0.5 * F.mse_loss(state_values[:-1], returns)

        return {
            'td_lambda_loss': loss
        }

    # overwrite
    def __repr__(self):
        return "TDLambda (lambda={})\n".format(self.lambda_)

    def _lambda_returns(self, state_values, rewards, discounts):
        '''
        Note:
            result[last] = rewards[last] + discounts[last] * state_values[last]
            result[t] = rewards[t] + discounts[t] * (lambda_ * result[t+1] + (1 - lambda_) * state_values[t])
        '''
        # equivalent implementation for speeding up
        sequence = rewards + discounts * (1 - self.lambda_) * state_values
        decay = discounts * self.lambda_
        result = torch.zeros_like(state_values)
        # TODO (optimization, tf.span)
        for i in reversed(range(result.shape[0])):
            if i == result.shape[0] - 1:
                result[i] = sequence[i] + decay * state_values[-1]
            else:
                result[i] = sequence[i] + decay * result[i+1]
        return result
