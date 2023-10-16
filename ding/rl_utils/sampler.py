import torch
import treetensor.torch as ttorch
from torch.distributions import Normal, Independent


class ArgmaxSampler:
    '''
    Overview:
        Argmax sampler, return the index of the maximum value
    '''

    def __call__(self, logit: torch.Tensor) -> torch.Tensor:
        '''
        Overview:
            Return the index of the maximum value
        Arguments:
            - logit (:obj:`torch.Tensor`): The input tensor
        Returns:
            - action (:obj:`torch.Tensor`): The index of the maximum value
        '''
        return logit.argmax(dim=-1)


class MultinomialSampler:
    '''
    Overview:
        Multinomial sampler, return the index of the sampled value
    '''

    def __call__(self, logit: torch.Tensor) -> torch.Tensor:
        '''
        Overview:
            Return the index of the sampled value
        Arguments:
            - logit (:obj:`torch.Tensor`): The input tensor
        Returns:
            - action (:obj:`torch.Tensor`): The index of the sampled value
        '''
        dist = torch.distributions.Categorical(logits=logit)
        return dist.sample()


class MuSampler:
    '''
    Overview:
        Mu sampler, return the mu of the input tensor
    '''

    def __call__(self, logit: ttorch.Tensor) -> torch.Tensor:
        '''
        Overview:
            Return the mu of the input tensor
        Arguments:
            - logit (:obj:`ttorch.Tensor`): The input tensor
        Returns:
            - action (:obj:`torch.Tensor`): The mu of the input tensor
        '''
        return logit.mu


class ReparameterizationSampler:
    '''
    Overview:
        Reparameterization sampler, return the reparameterized value of the input tensor
    '''

    def __call__(self, logit: ttorch.Tensor) -> torch.Tensor:
        '''
        Overview:
            Return the reparameterized value of the input tensor
        Arguments:
            - logit (:obj:`ttorch.Tensor`): The input tensor
        Returns:
            - action (:obj:`torch.Tensor`): The reparameterized value of the input tensor
        '''
        dist = Normal(logit.mu, logit.sigma)
        dist = Independent(dist, 1)
        return dist.rsample()


class HybridStochasticSampler:
    '''
    Overview:
        Hybrid stochastic sampler, return the sampled action type and the reparameterized action args
    '''

    def __call__(self, logit: ttorch.Tensor) -> ttorch.Tensor:
        '''
        Overview:
            Return the sampled action type and the reparameterized action args
        Arguments:
            - logit (:obj:`ttorch.Tensor`): The input tensor
        Returns:
            - action (:obj:`ttorch.Tensor`): The sampled action type and the reparameterized action args
        '''
        dist = torch.distributions.Categorical(logits=logit.action_type)
        action_type = dist.sample()
        dist = Normal(logit.action_args.mu, logit.action_args.sigma)
        dist = Independent(dist, 1)
        action_args = dist.rsample()
        return ttorch.as_tensor({
            'action_type': action_type,
            'action_args': action_args,
        })


class HybridDeterminsticSampler:
    '''
    Overview:
        Hybrid deterministic sampler, return the argmax action type and the mu action args
    '''

    def __call__(self, logit: ttorch.Tensor) -> ttorch.Tensor:
        '''
        Overview:
            Return the argmax action type and the mu action args
        Arguments:
            - logit (:obj:`ttorch.Tensor`): The input tensor
        Returns:
            - action (:obj:`ttorch.Tensor`): The argmax action type and the mu action args
        '''
        action_type = logit.action_type.argmax(dim=-1)
        action_args = logit.action_args.mu
        return ttorch.as_tensor({
            'action_type': action_type,
            'action_args': action_args,
        })
