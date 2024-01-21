from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Tuple, Dict

import torch
import numpy as np
import torch.nn.functional as F


class Pd(object):
    """
    Overview:
        Abstract class for parameterizable probability distributions and sampling functions.
    Interfaces:
        ``neglogp``, ``entropy``, ``noise_mode``, ``mode``, ``sample``

    .. tip::

        In dereived classes, `logits` should be an attribute member stored in class.
    """

    def neglogp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Calculate cross_entropy between input x and logits
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
        Return:
            - cross_entropy (:obj:`torch.Tensor`): the returned cross_entropy loss
        """
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        """
        Overview:
            Calculate the softmax entropy of logits
        Arguments:
            - reduction (:obj:`str`): support [None, 'mean'], default set to 'mean'
        Returns:
            - entropy (:obj:`torch.Tensor`): the calculated entropy
        """
        raise NotImplementedError

    def noise_mode(self):
        """
        Overview:
            Add noise to logits. This method is designed for randomness
        """
        raise NotImplementedError

    def mode(self):
        """
        Overview:
            Return logits argmax result. This method is designed for deterministic.
        """
        raise NotImplementedError

    def sample(self):
        """
        Overview:
            Sample from logits's distribution by using softmax. This method is designed for multinomial.
        """
        raise NotImplementedError


class CategoricalPd(Pd):
    """
    Overview:
        Catagorical probility distribution sampler
    Interfaces:
        ``__init__``, ``neglogp``, ``entropy``, ``noise_mode``, ``mode``, ``sample``
    """

    def __init__(self, logits: torch.Tensor = None) -> None:
        """
        Overview:
            Init the Pd with logits
        Arguments:
            - logits (:obj:torch.Tensor): logits to sample from
        """
        self.update_logits(logits)

    def update_logits(self, logits: torch.Tensor) -> None:
        """
        Overview:
            Updata logits
        Arguments:
            - logits (:obj:`torch.Tensor`): logits to update
        """
        self.logits = logits

    def neglogp(self, x, reduction: str = 'mean') -> torch.Tensor:
        """
        Overview:
            Calculate cross_entropy between input x and logits
        Arguments:
            - x (:obj:`torch.Tensor`): the input tensor
            - reduction (:obj:`str`): support [None, 'mean'], default set to mean
        Return:
            - cross_entropy (:obj:`torch.Tensor`): the returned cross_entropy loss
        """
        return F.cross_entropy(self.logits, x, reduction=reduction)

    def entropy(self, reduction: str = 'mean') -> torch.Tensor:
        """
        Overview:
            Calculate the softmax entropy of logits
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

    def noise_mode(self, viz: bool = False) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Overview:
            add noise to logits
        Arguments:
            - viz (:obj:`bool`): Whether to return numpy from of logits, noise and noise_logits; \
                Short for ``visualize`` . (Because tensor type cannot visualize in tb or text log)
        Returns:
            - result (:obj:`torch.Tensor`): noised logits
            - viz_feature (:obj:`Dict[str, np.ndarray]`): ndarray type data for visualization.
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

    def mode(self, viz: bool = False) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Overview:
            return logits argmax result
        Arguments:
            - viz (:obj:`bool`): Whether to return numpy from of logits, noise and noise_logits;
                Short for ``visualize`` . (Because tensor type cannot visualize in tb or text log)
        Returns:
            - result (:obj:`torch.Tensor`): the logits argmax result
            - viz_feature (:obj:`Dict[str, np.ndarray]`): ndarray type data for visualization.
        """
        result = self.logits.argmax(dim=-1)
        if viz:
            viz_feature = {}
            viz_feature['logits'] = self.logits.data.cpu().numpy()
            return result, viz_feature
        else:
            return result

    def sample(self, viz: bool = False) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """
        Overview:
            Sample from logits's distribution by using softmax
        Arguments:
            - viz (:obj:`bool`): Whether to return numpy from of logits, noise and noise_logits; \
                Short for ``visualize`` . (Because tensor type cannot visualize in tb or text log)
        Returns:
            - result (:obj:`torch.Tensor`): the logits sampled result
            - viz_feature (:obj:`Dict[str, np.ndarray]`): ndarray type data for visualization.
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
    """
    Overview:
        Wrapped ``torch.distributions.Categorical``

    Interfaces:
        ``__init__``, ``update_logits``, ``update_probs``, ``sample``, ``neglogp``, ``mode``, ``entropy``
    """

    def __init__(self, probs: torch.Tensor = None) -> None:
        """
        Overview:
            Initialize the CategoricalPdPytorch object.
        Arguments:
            - probs (:obj:`torch.Tensor`): The tensor of probabilities.
        """
        if probs is not None:
            self.update_probs(probs)

    def update_logits(self, logits: torch.Tensor) -> None:
        """
        Overview:
            Updata logits
        Arguments:
            - logits (:obj:`torch.Tensor`): logits to update
        """
        super().__init__(logits=logits)

    def update_probs(self, probs: torch.Tensor) -> None:
        """
        Overview:
            Updata probs
        Arguments:
            - probs (:obj:`torch.Tensor`): probs to update
        """
        super().__init__(probs=probs)

    def sample(self) -> torch.Tensor:
        """
        Overview:
            Sample from logits's distribution by using softmax
        Return:
            - result (:obj:`torch.Tensor`): the logits sampled result
        """
        return super().sample()

    def neglogp(self, actions: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Overview:
            Calculate cross_entropy between input x and logits
        Arguments:
            - actions (:obj:`torch.Tensor`): the input action tensor
            - reduction (:obj:`str`): support [None, 'mean'], default set to mean
        Return:
            - cross_entropy (:obj:`torch.Tensor`): the returned cross_entropy loss
        """
        neglogp = super().log_prob(actions)
        assert (reduction in ['none', 'mean'])
        if reduction == 'none':
            return neglogp
        elif reduction == 'mean':
            return neglogp.mean(dim=0)

    def mode(self) -> torch.Tensor:
        """
        Overview:
            Return logits argmax result
        Return:
            - result(:obj:`torch.Tensor`): the logits argmax result
        """
        return self.probs.argmax(dim=-1)

    def entropy(self, reduction: str = None) -> torch.Tensor:
        """
        Overview:
            Calculate the softmax entropy of logits
        Arguments:
            - reduction (:obj:`str`): support [None, 'mean'], default set to mean
        Returns:
            - entropy (:obj:`torch.Tensor`): the calculated entropy
        """
        entropy = super().entropy()
        assert (reduction in [None, 'mean'])
        if reduction is None:
            return entropy
        elif reduction == 'mean':
            return entropy.mean()
