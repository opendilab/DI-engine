import torch
import torch.nn as nn


class ValueActorCriticBase(nn.Module):

    def forward(self, inputs, mode=None, **kwargs):
        """
        Note:
            mode:
                - compute_action_value: normal reinforcement learning training
                - compute_action: evaluate policy performance, only use the actor part
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in ['compute_action_value', 'compute_action', 'mimic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        torch.manual_seed(seed)

    def compute_action_value(self, inputs):
        raise NotImplementedError

    def compute_action(self, inputs, **kwargs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _critic_forward(self, inputs, **kwargs):
        raise NotImplementedError


class QActorCriticBase(nn.Module):

    def forward(self, inputs, mode=None, **kwargs):
        """
        Note:
            mode:
                - optimize_actor: optimize actor part, with critic part `no grad`, return q value
                - compute_q: evaluate q value based on state and action from buffer
                - compute_action: evaluate policy performance, only use the actor part
                - mimic: supervised learning, learn policy/value output label
        """
        assert (mode in ['optimize_actor', 'compute_q', 'compute_action', 'compute_action_q', 'mimic']), mode
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def seed(self, seed):
        torch.manual_seed(seed)

    def optimize_actor(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_q(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_action(self, inputs, **kwargs):
        raise NotImplementedError

    def compute_action_q(self, inputs, **kwargs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _critic_forward(self, inputs, **kwargs):
        raise NotImplementedError
