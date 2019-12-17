from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pd(object):

    def neglogp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        # for randomness
        raise NotImplementedError

    def mode(self):
        # for deterministic
        raise NotImplementedError


class CategoricalPd(Pd):

    def __init__(self, logits=None):
        self.update_logits(logits)

    def update_logits(self, logits):
        self.logits = logits

    def neglogp(self, x, reduction='mean'):
        return F.cross_entropy(self.logits, x, reduction=reduction)

    def entropy(self, reduction='mean'):
        a = self.logits - self.logits.max(dim=-1, keepdim=True)[0]
        ea = torch.exp(a)
        z = ea.sum(dim=-1, keepdim=True)
        p = ea / z
        entropy = (p * (torch.log(z) - a)).sum(dim=-1)
        assert(reduction in ['none', 'mean'])
        if reduction == 'none':
            return entropy
        elif reduction == 'mean':
            return entropy.mean()

    def sample(self):
        u = torch.rand_like(self.logits)
        u = self.logits - torch.log(-torch.log(u))
        return u.argmax(dim=-1)

    def mode(self):
        return self.logits.argmax(dim=-1)


class CategoricalPdPytorch(torch.distributions.Categorical):
    def __init__(self, logits=None):
        if logits is not None:
            self.update_logits(logits)

    def update_logits(self, logits):
        super(CategoricalPdPytorch, self).__init__(logits=logits)

    def sample(self):
        return super().sample()

    def neglogp(self, actions, reduction='mean'):
        neglogp = super().log_prob(actions)
        assert(reduction in ['none', 'mean'])
        if reduction == 'none':
            return neglogp
        elif reduction == 'mean':
            return neglogp.mean(dim=0)

    def mode(self):
        return self.probs.argmax(dim=-1)

    def entropy(self, reduction='none'):
        entropy = super().entropy()
        assert(reduction in ['none', 'mean'])
        if reduction == 'none':
            return entropy
        elif reduction == 'mean':
            return entropy.mean()
