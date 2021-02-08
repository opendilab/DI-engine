from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F


class Pd(object):
    r"""
    Overview:
        Abstract class for parameterizable probability distributions and sampling functions
    Interface:
        neglogp, entropy, noise_mode, mode, sample
    """

    def neglogp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def noise_mode(self):
        # for randomness
        raise NotImplementedError

    def mode(self):
        # for deterministic
        raise NotImplementedError

    def sample(self):
        # for multinomial
        raise NotImplementedError


class CategoricalPd(Pd):
    r"""
    Overview:
        Catagorical probility distribution sampler
    Interface:
        __init__, update_logits, neglogp, entropy, noise_mode, mode, sample
    """

    def __init__(self, logits=None):
        r"""
        Overview:
            init the Pd with logits
        Arguments:
            - logits (:obj:torch.Tensor): logits to sample from
        """
        self.update_logits(logits)

    def update_logits(self, logits):
        r"""
        Overview:
            updata logits
        Arguments:
            - logits (:obj:torch.Tensor): logits to update
        """
        self.logits = logits

    def neglogp(self, x, reduction='mean'):
        r"""
        Overview:
            calculate cross_entropy between input x and logits
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Return:
            - cross_entropy (:obj:`torch.Tensor`): the returned cross_entropy loss
        """
        return F.cross_entropy(self.logits, x, reduction=reduction)

    def entropy(self, reduction='mean'):
        r"""
        Overview:
            calculate the softmax entropy of logits
        Arguments:
            - reduction (:obj:`str`): support [None, 'mean'], default set to mean
        Returns:
            - entropy (:obj:`torch.Tensor`): the calculated entropy
        """
        a = self.logits - self.logits.max(dim=-1, keepdim=True)[0]
        ea = torch.exp(a)
        z = ea.sum(dim=-1, keepdim=True)
        p = ea / z
        entropy = (p * (torch.log(z) - a)).sum(dim=-1)
        assert (reduction in [None, 'mean'])
        if reduction is None:
            return entropy
        elif reduction == 'mean':
            return entropy.mean()

    def noise_mode(self, viz=False):
        r"""
        Overview:
            add noise to logits
        Arguments:
            - viz (:obj:`bool`): whether to return numpy from of logits, noise and noise_logits; \
                have absolutly no idea why it is called viz.
        Return:
            - result (:obj:`torch.Tensor`): noised logits
            - viz_feature (:obj:`dict` of :obj:`numpy.array`): appended viz feature
        """
        u = torch.rand_like(self.logits)
        u = -torch.log(-torch.log(u))
        noise_logits = self.logits + u
        result = noise_logits.argmax(dim=-1)
        if viz:
            viz_feature = {}
            viz_feature['logits'] = self.logits.data.cpu().numpy()
            viz_feature['noise'] = u.data.cpu().numpy()
            viz_feature['noise_logits'] = noise_logits.data.cpu().numpy()
            return result, viz_feature
        else:
            return result

    def mode(self, viz=False):
        r"""
        Overview:
            return logits argmax result
        Argiments:
            - viz (:obj:`bool`): whether to return numpy from of logits
        Return:
            - result (:obj:`torch.Tensor`): the logits argmax result
            - viz_feature (:obj:`dict` of :obj:`numpy.array`): appended viz feature
        """
        result = self.logits.argmax(dim=-1)
        if viz:
            viz_feature = {}
            viz_feature['logits'] = self.logits.data.cpu().numpy()
            return result, viz_feature
        else:
            return result

    def sample(self, viz=False):
        r"""
        Overview:
            sample from logits's distributin use softmax
        Arguments:
            - viz (:obj:`bool`): whether to return numpy from of logits
        Return:
            - result (:obj:`torch.Tensor`): the logits sampled result
            - viz_feature (:obj:`dict` of :obj:`numpy.array`): appended viz feature
        """
        p = torch.softmax(self.logits, dim=1)
        result = torch.multinomial(p, 1).squeeze(1)
        if viz:
            viz_feature = {}
            viz_feature['logits'] = self.logits.data.cpu().numpy()
            return result, viz_feature
        else:
            return result


class CategoricalPdPytorch(torch.distributions.Categorical):
    r"""
    Overview:
        Wrapped torch.distributions.Categorical
    Notes:
        Please refer to torch.distributions.Categorical doc: \
            https://pytorch.org/docs/stable/distributions.html?highlight=torch%20distributions#module-torch.distributions\
                Categorical
    Interface:
        __init__, update_logits, updata_probs, sample, neglogp, mode, entropy
    """

    def __init__(self, probs=None):
        if probs is not None:
            self.update_probs(probs)

    def update_logits(self, logits):
        super(CategoricalPdPytorch, self).__init__(logits=logits)

    def update_probs(self, probs):
        super(CategoricalPdPytorch, self).__init__(probs=probs)

    def sample(self):
        return super().sample()

    def neglogp(self, actions, reduction='mean'):
        neglogp = super().log_prob(actions)
        assert (reduction in ['none', 'mean'])
        if reduction == 'none':
            return neglogp
        elif reduction == 'mean':
            return neglogp.mean(dim=0)

    def mode(self):
        return self.probs.argmax(dim=-1)

    def entropy(self, reduction=None):
        entropy = super().entropy()
        assert (reduction in [None, 'mean'])
        if reduction is None:
            return entropy
        elif reduction == 'mean':
            return entropy.mean()
