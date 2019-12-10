from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pd(object):

    def neglogp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class CategoricalPd(Pd):

    def __init__(self, logits=None):
        self.update_logits(logits)

    def update_logits(self, logits):
        self.logits = logits

    def neglogp(self, x, reduction='mean'):
        return F.cross_entropy(self.logits, x, reduction=reduction)

    def entropy(self):
        a = self.logits - self.logits.max(dim=-1, keepdim=True)[0]
        ea = torch.exp(a)
        z = ea.sum(dim=-1, keepdim=True)
        p = ea / z
        ret = (p * (torch.log(z) - a)).sum(dim=-1)
        return ret

    def sample(self):
        u = torch.rand_like(self.logits)
        u = self.logits - torch.log(-torch.log(u))
        return u.argmax(dim=-1)
