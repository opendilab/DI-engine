import math
from abc import ABC, abstractmethod
from typing import Callable, Union, Optional
from copy import deepcopy
from ding.torch_utils.data_helper import to_device

import torch


def get_epsilon_greedy_fn(start: float, end: float, decay: int, type_: str = 'exp') -> Callable:
    """
    Overview:
        Generate an epsilon_greedy function with decay, which inputs current timestep and outputs current epsilon.
    Arguments:
        - start (:obj:`float`): Epsilon start value. For 'linear', it should be 1.0.
        - end (:obj:`float`): Epsilon end value.
        - decay (:obj:`int`): Controls the speed that epsilon decreases from ``start`` to ``end``. \
            We recommend epsilon decays according to env step rather than iteration.
        - type (:obj:`str`): How epsilon decays, now supports ['linear', 'exp'(exponential)]
    Returns:
        - eps_fn (:obj:`function`): The epsilon greedy function with decay
    """
    assert type_ in ['linear', 'exp'], type_
    if type_ == 'exp':
        return lambda x: (start - end) * math.exp(-1 * x / decay) + end
    elif type_ == 'linear':

        def eps_fn(x):
            if x >= decay:
                return end
            else:
                return (start - end) * (1 - x / decay) + end

        return eps_fn


class BaseNoise(ABC):
    r"""
    Overview:
        Base class for action noise
    Interface:
        __init__, __call__
    Examples:
        >>> noise_generator = OUNoise()  # init one type of noise
        >>> noise = noise_generator(action.shape, action.device)  # generate noise
    """

    def __init__(self) -> None:
        """
        Overview:
            Initialization method
        """
        super().__init__()

    @abstractmethod
    def __call__(self, shape: tuple, device: str) -> torch.Tensor:
        """
        Overview:
            Generate noise according to action tensor's shape, device
        Arguments:
            - shape (:obj:`tuple`): size of the action tensor, output noise's size should be the same
            - device (:obj:`str`): device of the action tensor, output noise's device should be the same as it
        Returns:
            - noise (:obj:`torch.Tensor`): generated action noise, \
                have the same shape and device with the input action tensor
        """
        raise NotImplementedError


class GaussianNoise(BaseNoise):
    r"""
    Overview:
        Derived class for generating gaussian noise, which satisfies :math:`X \sim N(\mu, \sigma^2)`
    Interface:
        __init__, __call__
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        """
        Overview:
            Initialize :math:`\mu` and :math:`\sigma` in Gaussian Distribution
        Arguments:
            - mu (:obj:`float`):  :math:`\mu` , mean value
            - sigma (:obj:`float`): :math:`\sigma` , standard deviation, should be positive
        """
        super(GaussianNoise, self).__init__()
        self._mu = mu
        assert sigma >= 0, "GaussianNoise's sigma should be positive."
        self._sigma = sigma

    def __call__(self, shape: tuple, device: str) -> torch.Tensor:
        """
        Overview:
            Generate gaussian noise according to action tensor's shape, device
        Arguments:
            - shape (:obj:`tuple`): size of the action tensor, output noise's size should be the same
            - device (:obj:`str`): device of the action tensor, output noise's device should be the same as it
        Returns:
            - noise (:obj:`torch.Tensor`): generated action noise, \
                have the same shape and device with the input action tensor
        """
        noise = torch.randn(shape, device=device)
        noise = noise * self._sigma + self._mu
        return noise


class OUNoise(BaseNoise):
    r"""
    Overview:
        Derived class for generating Ornstein-Uhlenbeck process noise.
        Satisfies :math:`dx_t=\theta(\mu-x_t)dt + \sigma dW_t`,
        where :math:`W_t` denotes Weiner Process, acting as a random perturbation term.
    Interface:
        __init__, reset, __call__
    """

    def __init__(
            self,
            mu: float = 0.0,
            sigma: float = 0.3,
            theta: float = 0.15,
            dt: float = 1e-2,
            x0: Optional[Union[float, torch.Tensor]] = 0.0,
    ) -> None:
        """
        Overview:
            Initialize ``_alpha`` :math:`=\theta * dt\`,
            ``beta`` :math:`= \sigma * \sqrt{dt}`,  in Ornstein-Uhlenbeck process
        Arguments:
            - mu (:obj:`float`):  :math:`\mu` , mean value
            - sigma (:obj:`float`): :math:`\sigma` , standard deviation of the perturbation noise
            - theta (:obj:`float`): how strongly the noise reacts to perturbations, \
                greater value means stronger reaction
            - dt (:obj:`float`): derivative of time t
            - x0 (:obj:`float` or :obj:`torch.Tensor`): initial action
        """
        super().__init__()
        self._mu = mu
        self._alpha = theta * dt
        self._beta = sigma * math.sqrt(dt)
        self._x0 = x0
        self.reset()

    def reset(self) -> None:
        """
        Overview:
            Reset ``_x`` to the initial state ``_x0``
        """
        self._x = deepcopy(self._x0)

    def __call__(self, shape: tuple, device: str, mu: Optional[float] = None) -> torch.Tensor:
        """
        Overview:
            Generate gaussian noise according to action tensor's shape, device
        Arguments:
            - shape (:obj:`tuple`): size of the action tensor, output noise's size should be the same
            - device (:obj:`str`): device of the action tensor, output noise's device should be the same as it
            - mu (:obj:`float`): new mean value :math:`\mu`, you can set it to `None` if don't need it
        Returns:
            - noise (:obj:`torch.Tensor`): generated action noise, \
                have the same shape and device with the input action tensor
        """
        if self._x is None or \
                (isinstance(self._x, torch.Tensor) and self._x.shape != shape):
            self._x = torch.zeros(shape)
        if mu is None:
            mu = self._mu
        noise = self._alpha * (mu - self._x) + self._beta * torch.randn(shape)
        self._x += noise
        noise = to_device(noise, device)
        return noise

    @property
    def x0(self) -> Union[float, torch.Tensor]:
        return self._x0

    @x0.setter
    def x0(self, _x0: Union[float, torch.Tensor]) -> None:
        """
        Overview:
            Set ``self._x0`` and reset ``self.x`` to ``self._x0`` as well
        """
        self._x0 = _x0
        self.reset()


noise_mapping = {'gauss': GaussianNoise, 'ou': OUNoise}


def create_noise_generator(noise_type: str, noise_kwargs: dict) -> BaseNoise:
    """
    Overview:
        Given the key (noise_type), create a new noise generator instance if in noise_mapping's values,
        or raise an KeyError. In other words, a derived noise generator must first register,
        then call ``create_noise generator`` to get the instance object.
    Arguments:
        - noise_type (:obj:`str`): the type of noise generator to be created
    Returns:
        - noise (:obj:`BaseNoise`): the created new noise generator, should be an instance of one of \
            noise_mapping's values
    """
    if noise_type not in noise_mapping.keys():
        raise KeyError("not support noise type: {}".format(noise_type))
    else:
        return noise_mapping[noise_type](**noise_kwargs)
