import torch.nn as nn


class ActorCriticBase(nn.Module):
    def forward(self, inputs, mode=None):
        assert(mode in ['step', 'value', 'evaluate'])
        f = getattr(self, mode)
        return f(inputs)

    def step(self, inputs):
        raise NotImplementedError

    def value(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs):
        raise NotImplementedError

    def _actor_forward(self, inputs):
        raise NotImplementedError

    def _critic_forward(self, inputs):
        raise NotImplementedError
