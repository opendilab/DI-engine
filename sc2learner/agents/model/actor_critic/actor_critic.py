import torch.nn as nn


class ActorCriticBase(nn.Module):
    def forward(self, inputs, mode=None, **kwargs):
        assert(mode in ['step', 'value', 'evaluate', 'mimic'])
        f = getattr(self, mode, **kwargs)
        return f(inputs)

    def step(self, inputs):
        raise NotImplementedError

    def value(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs):
        raise NotImplementedError

    def mimic(self, inputs, **kwargs):
        raise NotImplementedError

    def _actor_forward(self, inputs):
        raise NotImplementedError

    def _critic_forward(self, inputs):
        raise NotImplementedError
