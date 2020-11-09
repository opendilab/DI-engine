import torch
import torch.nn as nn


class ActorCriticBase(nn.Module):

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
