import torch
import torch.nn as nn


class ActorCriticBase(nn.Module):
    def forward(self, inputs, mode=None, **kwargs):
        # FIXME(pzh) That's a quiet strange implementation ...
        assert (mode in ['step', 'value', 'evaluate', 'mimic'])
        f = getattr(self, mode)
        return f(inputs, **kwargs)

    def set_seed(self, seed):
        torch.manual_seed(seed)

    def step(self, inputs):
        raise NotImplementedError

    def value(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, **kwargs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs, **kwargs):
        raise NotImplementedError

    def _critic_forward(self, inputs, **kwargs):
        raise NotImplementedError
