"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/model.py
"""

import torch
import numpy as np
import torch.nn as nn
from typing import List, NamedTuple
from dataclasses import dataclass


@dataclass
class NetworkOutput:
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden: object


def concat_output_value(output_lst):
    # concat the values of the model output list
    value_lst = []
    for output in output_lst:
        value_lst.append(output.value)

    value_lst = np.concatenate(value_lst)

    return value_lst


def concat_output(output_lst):
    # concat the model output
    value_lst, reward_lst, policy_logits_lst, hidden_state_lst = [], [], [], []
    reward_hidden_c_lst, reward_hidden_h_lst = [], []
    for output in output_lst:
        value_lst.append(output.value)
        reward_lst.append(output.value_prefix)
        policy_logits_lst.append(output.policy_logits)
        hidden_state_lst.append(output.hidden_state)
        reward_hidden_c_lst.append(output.reward_hidden[0].squeeze(0))
        reward_hidden_h_lst.append(output.reward_hidden[1].squeeze(0))

    value_lst = np.concatenate(value_lst)
    reward_lst = np.concatenate(reward_lst)
    policy_logits_lst = np.concatenate(policy_logits_lst)
    hidden_state_lst = np.concatenate(hidden_state_lst)
    reward_hidden_c_lst = np.expand_dims(np.concatenate(reward_hidden_c_lst), axis=0)
    reward_hidden_h_lst = np.expand_dims(np.concatenate(reward_hidden_h_lst), axis=0)

    return value_lst, reward_lst, policy_logits_lst, hidden_state_lst, (reward_hidden_c_lst, reward_hidden_h_lst)


class BaseNet(nn.Module):

    def __init__(self, inverse_value_transform, inverse_reward_transform, lstm_hidden_size):
        """
        Overview:
            Base Network
            schedule_timesteps. After this many timesteps pass final_p is
            returned.
            # discrete support: [-300, 300] support of value to represent the value scalars
        Arguments
             - inverse_value_transform: Any
                A function that maps value supports into value scalars
             - inverse_reward_transform: Any
                A function that maps reward supports into value scalars
            - lstm_hidden_size: int
                dim of lstm hidden
        """
        super(BaseNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform
        self.lstm_hidden_size = lstm_hidden_size

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, reward_hidden, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        num = obs.size(0)

        state = self.representation(obs)
        policy_logits, value = self.prediction(state)

        if self.training:
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (torch.zeros(1, num, self.lstm_hidden_size), torch.zeros(1, num, self.lstm_hidden_size))
        else:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy()
            # zero initialization for reward (value prefix) hidden states
            reward_hidden = (
                torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy(),
                torch.zeros(1, num, self.lstm_hidden_size).detach().cpu().numpy()
            )

        return NetworkOutput(value, [0. for _ in range(num)], policy_logits, state, reward_hidden)

    def recurrent_inference(self, hidden_state, reward_hidden, action) -> NetworkOutput:
        state, reward_hidden, value_prefix = self.dynamics(hidden_state, reward_hidden, action)
        policy_logits, value = self.prediction(state)

        if not self.training:
            # if not in training, obtain the scalars of the value/reward
            value = self.inverse_value_transform(value).detach().cpu().numpy()
            value_prefix = self.inverse_reward_transform(value_prefix).detach().cpu().numpy()
            state = state.detach().cpu().numpy()
            reward_hidden = (reward_hidden[0].detach().cpu().numpy(), reward_hidden[1].detach().cpu().numpy())
            policy_logits = policy_logits.detach().cpu().numpy()

        return NetworkOutput(value, value_prefix, policy_logits, state, reward_hidden)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients: torch.Tensor):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g
