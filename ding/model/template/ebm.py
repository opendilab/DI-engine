"""
Vanilla DFO and EBM are adapted from https://github.com/kevinzakka/ibc.
MCMC is adapted from https://github.com/google-research/ibc.
"""
from typing import Callable, Tuple
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from ding.utils import MODEL_REGISTRY, STOCHASTIC_OPTIMIZER_REGISTRY
from ding.torch_utils import unsqueeze_repeat
from ding.model.wrapper import IModelWrapper
from ding.model.common import RegressionHead


def create_stochastic_optimizer(device: str, stochastic_optimizer_config: dict):
    """
    Overview:
        Create stochastic optimizer.
    Arguments:
        - device (:obj:`str`): Device.
        - stochastic_optimizer_config (:obj:`dict`): Stochastic optimizer config.
    """
    return STOCHASTIC_OPTIMIZER_REGISTRY.build(
        stochastic_optimizer_config.pop("type"), device=device, **stochastic_optimizer_config
    )


def no_ebm_grad():
    """Wrapper that disables energy based model gradients"""

    def ebm_disable_grad_wrapper(func: Callable):

        @wraps(func)
        def wrapper(*args, **kwargs):
            ebm = args[-1]
            assert isinstance(ebm, (IModelWrapper, nn.Module)),\
                   'Make sure ebm is the last positional arguments.'
            ebm.requires_grad_(False)
            result = func(*args, **kwargs)
            ebm.requires_grad_(True)
            return result

        return wrapper

    return ebm_disable_grad_wrapper


class StochasticOptimizer(ABC):
    """
    Overview:
        Base class for stochastic optimizers.
    Interface:
        ``__init__``, ``_sample``, ``_get_best_action_sample``, ``set_action_bounds``, ``sample``, ``infer``
    """

    def _sample(self, obs: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Drawing action samples from the uniform random distribution \
                and tiling observations to the same shape as action samples.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observation.
            - num_samples (:obj:`int`): The number of negative samples.
        Returns:
            - tiled_obs (:obj:`torch.Tensor`): Observations tiled.
            - action (:obj:`torch.Tensor`): Action sampled.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - num_samples (:obj:`int`): :math:`N`.
            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> opt = StochasticOptimizer()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> tiled_obs, action = opt._sample(obs, 8)
        """
        size = (obs.shape[0], num_samples, self.action_bounds.shape[1])
        low, high = self.action_bounds[0, :], self.action_bounds[1, :]
        action_samples = low + (high - low) * torch.rand(size).to(self.device)
        tiled_obs = unsqueeze_repeat(obs, num_samples, 1)
        return tiled_obs, action_samples

    @staticmethod
    @torch.no_grad()
    def _get_best_action_sample(obs: torch.Tensor, action_samples: torch.Tensor, ebm: nn.Module):
        """
        Overview:
            Return one action for each batch with highest probability (lowest energy).
        Arguments:
            - obs (:obj:`torch.Tensor`): Observation.
            - action_samples (:obj:`torch.Tensor`): Action from uniform distributions.
        Returns:
            - best_action_samples (:obj:`torch.Tensor`): Best action.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - action_samples (:obj:`torch.Tensor`): :math:`(B, N, A)`.
            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> action_samples = torch.randn(2, 8, 5)
            >>> ebm = EBM(4, 5)
            >>> opt = StochasticOptimizer()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> best_action_samples = opt._get_best_action_sample(obs, action_samples, ebm)
        """
        # (B, N)
        energies = ebm.forward(obs, action_samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        # (B, )
        best_idxs = probs.argmax(dim=-1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs]

    def set_action_bounds(self, action_bounds: np.ndarray):
        """
        Overview:
            Set action bounds calculated from the dataset statistics.
        Arguments:
            - action_bounds (:obj:`np.ndarray`): Array of shape (2, A), \
                where action_bounds[0] is lower bound and action_bounds[1] is upper bound.
        Returns:
            - action_bounds (:obj:`torch.Tensor`): Action bounds.
        Shapes:
            - action_bounds (:obj:`np.ndarray`): :math:`(2, A)`.
            - action_bounds (:obj:`torch.Tensor`): :math:`(2, A)`.
        Examples:
            >>> opt = StochasticOptimizer()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
        """
        self.action_bounds = torch.as_tensor(action_bounds, dtype=torch.float32).to(self.device)

    @abstractmethod
    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Create tiled observations and sample counter-negatives for InfoNCE loss.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - tiled_obs (:obj:`torch.Tensor`): Tiled observations.
            - action (:obj:`torch.Tensor`): Actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.

        .. note:: In the case of derivative-free optimization, this function will simply call _sample.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Overview:
            Optimize for the best action conditioned on the current observation.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - best_action_samples (:obj:`torch.Tensor`): Best actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.
        """
        raise NotImplementedError


@STOCHASTIC_OPTIMIZER_REGISTRY.register('dfo')
class DFO(StochasticOptimizer):
    """
    Overview:
        Derivative-Free Optimizer in paper Implicit Behavioral Cloning.
        https://arxiv.org/abs/2109.00137
    Interface:
        ``init``, ``sample``, ``infer``
    """

    def __init__(
        self,
        noise_scale: float = 0.33,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 8,
        inference_samples: int = 16384,
        device: str = 'cpu',
    ):
        """
        Overview:
            Initialize the Derivative-Free Optimizer
        Arguments:
            - noise_scale (:obj:`float`): Initial noise scale.
            - noise_shrink (:obj:`float`): Noise scale shrink rate.
            - iters (:obj:`int`): Number of iterations.
            - train_samples (:obj:`int`): Number of samples for training.
            - inference_samples (:obj:`int`): Number of samples for inference.
            - device (:obj:`str`): Device.
        """
        self.action_bounds = None
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.device = device

    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Drawing action samples from the uniform random distribution \
                and tiling observations to the same shape as action samples.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - tiled_obs (:obj:`torch.Tensor`): Tiled observation.
            - action_samples (:obj:`torch.Tensor`): Action samples.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action_samples (:obj:`torch.Tensor`): :math:`(B, N, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> ebm = EBM(4, 5)
            >>> opt = DFO()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> tiled_obs, action_samples = opt.sample(obs, ebm)
        """
        return self._sample(obs, self.train_samples)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Overview:
            Optimize for the best action conditioned on the current observation.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - best_action_samples (:obj:`torch.Tensor`): Actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> ebm = EBM(4, 5)
            >>> opt = DFO()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> best_action_samples = opt.infer(obs, ebm)
        """
        noise_scale = self.noise_scale

        # (B, N, O), (B, N, A)
        obs, action_samples = self._sample(obs, self.inference_samples)

        for i in range(self.iters):
            # (B, N)
            energies = ebm.forward(obs, action_samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            action_samples = action_samples[torch.arange(action_samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            action_samples = action_samples + torch.randn_like(action_samples) * noise_scale
            action_samples = action_samples.clamp(min=self.action_bounds[0, :], max=self.action_bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        return self._get_best_action_sample(obs, action_samples, ebm)


@STOCHASTIC_OPTIMIZER_REGISTRY.register('ardfo')
class AutoRegressiveDFO(DFO):
    """
    Overview:
        AutoRegressive Derivative-Free Optimizer in paper Implicit Behavioral Cloning.
        https://arxiv.org/abs/2109.00137
    Interface:
        ``__init__``, ``infer``
    """

    def __init__(
        self,
        noise_scale: float = 0.33,
        noise_shrink: float = 0.5,
        iters: int = 3,
        train_samples: int = 8,
        inference_samples: int = 4096,
        device: str = 'cpu',
    ):
        """
        Overview:
            Initialize the AutoRegressive Derivative-Free Optimizer
        Arguments:
            - noise_scale (:obj:`float`): Initial noise scale.
            - noise_shrink (:obj:`float`): Noise scale shrink rate.
            - iters (:obj:`int`): Number of iterations.
            - train_samples (:obj:`int`): Number of samples for training.
            - inference_samples (:obj:`int`): Number of samples for inference.
            - device (:obj:`str`): Device.
        """
        super().__init__(noise_scale, noise_shrink, iters, train_samples, inference_samples, device)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Overview:
            Optimize for the best action conditioned on the current observation.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - best_action_samples (:obj:`torch.Tensor`): Actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> ebm = EBM(4, 5)
            >>> opt = AutoRegressiveDFO()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> best_action_samples = opt.infer(obs, ebm)
        """
        noise_scale = self.noise_scale

        # (B, N, O), (B, N, A)
        obs, action_samples = self._sample(obs, self.inference_samples)

        for i in range(self.iters):
            # j: action_dim index
            for j in range(action_samples.shape[-1]):
                # (B, N)
                energies = ebm.forward(obs, action_samples)[..., j]
                probs = F.softmax(-1.0 * energies, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
                action_samples = action_samples[torch.arange(action_samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                action_samples[..., j] = action_samples[..., j] + torch.randn_like(action_samples[..., j]) * noise_scale

                action_samples[..., j] = action_samples[..., j].clamp(
                    min=self.action_bounds[0, j], max=self.action_bounds[1, j]
                )

            noise_scale *= self.noise_shrink

        # (B, N)
        energies = ebm.forward(obs, action_samples)[..., -1]
        probs = F.softmax(-1.0 * energies, dim=-1)
        # (B, )
        best_idxs = probs.argmax(dim=-1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs]


@STOCHASTIC_OPTIMIZER_REGISTRY.register('mcmc')
class MCMC(StochasticOptimizer):
    """
    Overview:
        MCMC method as stochastic optimizers in paper Implicit Behavioral Cloning.
        https://arxiv.org/abs/2109.00137
    Interface:
        ``__init__``, ``sample``, ``infer``, ``grad_penalty``
    """

    class BaseScheduler(ABC):
        """
        Overview:
            Base class for learning rate scheduler.
        Interface:
            ``get_rate``
        """

        @abstractmethod
        def get_rate(self, index):
            """
            Overview:
                Abstract method for getting learning rate.
            """
            raise NotImplementedError

    class ExponentialScheduler:
        """
        Overview:
            Exponential learning rate schedule for Langevin sampler.
        Interface:
            ``__init__``, ``get_rate``
        """

        def __init__(self, init, decay):
            """
            Overview:
                Initialize the ExponentialScheduler.
            Arguments:
                - init (:obj:`float`): Initial learning rate.
                - decay (:obj:`float`): Decay rate.
            """
            self._decay = decay
            self._latest_lr = init

        def get_rate(self, index):
            """
            Overview:
                Get learning rate. Assumes calling sequentially.
            Arguments:
                - index (:obj:`int`): Current iteration.
            """
            del index
            lr = self._latest_lr
            self._latest_lr *= self._decay
            return lr

    class PolynomialScheduler:
        """
        Overview:
            Polynomial learning rate schedule for Langevin sampler.
        Interface:
            ``__init__``, ``get_rate``
        """

        def __init__(self, init, final, power, num_steps):
            """
            Overview:
                Initialize the PolynomialScheduler.
            Arguments:
                - init (:obj:`float`): Initial learning rate.
                - final (:obj:`float`): Final learning rate.
                - power (:obj:`float`): Power of polynomial.
                - num_steps (:obj:`int`): Number of steps.
            """
            self._init = init
            self._final = final
            self._power = power
            self._num_steps = num_steps

        def get_rate(self, index):
            """
            Overview:
                Get learning rate for index.
            Arguments:
                - index (:obj:`int`): Current iteration.
            """
            if index == -1:
                return self._init
            return (
                (self._init - self._final) * ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))
            ) + self._final

    def __init__(
        self,
        iters: int = 100,
        use_langevin_negative_samples: bool = True,
        train_samples: int = 8,
        inference_samples: int = 512,
        stepsize_scheduler: dict = dict(
            init=0.5,
            final=1e-5,
            power=2.0,
            # num_steps,
        ),
        optimize_again: bool = True,
        again_stepsize_scheduler: dict = dict(
            init=1e-5,
            final=1e-5,
            power=2.0,
            # num_steps,
        ),
        device: str = 'cpu',
        # langevin_step
        noise_scale: float = 0.5,
        grad_clip=None,
        delta_action_clip: float = 0.5,
        add_grad_penalty: bool = True,
        grad_norm_type: str = 'inf',
        grad_margin: float = 1.0,
        grad_loss_weight: float = 1.0,
        **kwargs,
    ):
        """
        Overview:
            Initialize the MCMC.
        Arguments:
            - iters (:obj:`int`): Number of iterations.
            - use_langevin_negative_samples (:obj:`bool`): Whether to use Langevin sampler.
            - train_samples (:obj:`int`): Number of samples for training.
            - inference_samples (:obj:`int`): Number of samples for inference.
            - stepsize_scheduler (:obj:`dict`): Step size scheduler for Langevin sampler.
            - optimize_again (:obj:`bool`): Whether to run a second optimization.
            - again_stepsize_scheduler (:obj:`dict`): Step size scheduler for the second optimization.
            - device (:obj:`str`): Device.
            - noise_scale (:obj:`float`): Initial noise scale.
            - grad_clip (:obj:`float`): Gradient clip.
            - delta_action_clip (:obj:`float`): Action clip.
            - add_grad_penalty (:obj:`bool`): Whether to add gradient penalty.
            - grad_norm_type (:obj:`str`): Gradient norm type.
            - grad_margin (:obj:`float`): Gradient margin.
            - grad_loss_weight (:obj:`float`): Gradient loss weight.
        """
        self.iters = iters
        self.use_langevin_negative_samples = use_langevin_negative_samples
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.stepsize_scheduler = stepsize_scheduler
        self.optimize_again = optimize_again
        self.again_stepsize_scheduler = again_stepsize_scheduler
        self.device = device

        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.delta_action_clip = delta_action_clip
        self.add_grad_penalty = add_grad_penalty
        self.grad_norm_type = grad_norm_type
        self.grad_margin = grad_margin
        self.grad_loss_weight = grad_loss_weight

    @staticmethod
    def _gradient_wrt_act(
            obs: torch.Tensor,
            action: torch.Tensor,
            ebm: nn.Module,
            create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Overview:
            Calculate gradient w.r.t action.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - action (:obj:`torch.Tensor`): Actions.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
            - create_graph (:obj:`bool`): Whether to create graph.
        Returns:
            - grad (:obj:`torch.Tensor`): Gradient w.r.t action.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - grad (:obj:`torch.Tensor`): :math:`(B, N, A)`.
        """
        action.requires_grad_(True)
        energy = ebm.forward(obs, action).sum()
        # `create_graph` set to `True` when second order derivative
        #  is needed i.e, d(de/da)/d_param
        grad = torch.autograd.grad(energy, action, create_graph=create_graph)[0]
        action.requires_grad_(False)
        return grad

    def grad_penalty(self, obs: torch.Tensor, action: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Overview:
            Calculate gradient penalty.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - action (:obj:`torch.Tensor`): Actions.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - loss (:obj:`torch.Tensor`): Gradient penalty.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N+1, O)`.
            - action (:obj:`torch.Tensor`): :math:`(B, N+1, A)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N+1, O)`.
            - loss (:obj:`torch.Tensor`): :math:`(B, )`.
        """
        if not self.add_grad_penalty:
            return 0.
        # (B, N+1, A), this gradient is differentiable w.r.t model parameters
        de_dact = MCMC._gradient_wrt_act(obs, action, ebm, create_graph=True)

        def compute_grad_norm(grad_norm_type, de_dact) -> torch.Tensor:
            # de_deact: B, N+1, A
            # return:   B, N+1
            grad_norm_type_to_ord = {
                '1': 1,
                '2': 2,
                'inf': float('inf'),
            }
            ord = grad_norm_type_to_ord[grad_norm_type]
            return torch.linalg.norm(de_dact, ord, dim=-1)

        # (B, N+1)
        grad_norms = compute_grad_norm(self.grad_norm_type, de_dact)
        grad_norms = grad_norms - self.grad_margin
        grad_norms = grad_norms.clamp(min=0., max=1e10)
        grad_norms = grad_norms.pow(2)

        grad_loss = grad_norms.mean()
        return grad_loss * self.grad_loss_weight

    # can not use @torch.no_grad() during the inference
    # because we need to calculate gradient w.r.t inputs as MCMC updates.
    @no_ebm_grad()
    def _langevin_step(self, obs: torch.Tensor, action: torch.Tensor, stepsize: float, ebm: nn.Module) -> torch.Tensor:
        """
        Overview:
            Run one langevin MCMC step.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - action (:obj:`torch.Tensor`): Actions.
            - stepsize (:obj:`float`): Step size.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - action (:obj:`torch.Tensor`): Actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.
            - stepsize (:obj:`float`): :math:`(B, )`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
        """
        l_lambda = 1.0
        de_dact = MCMC._gradient_wrt_act(obs, action, ebm)

        if self.grad_clip:
            de_dact = de_dact.clamp(min=-self.grad_clip, max=self.grad_clip)

        gradient_scale = 0.5
        de_dact = (gradient_scale * l_lambda * de_dact + torch.randn_like(de_dact) * l_lambda * self.noise_scale)

        delta_action = stepsize * de_dact
        delta_action_clip = self.delta_action_clip * 0.5 * (self.action_bounds[1] - self.action_bounds[0])
        delta_action = delta_action.clamp(min=-delta_action_clip, max=delta_action_clip)

        action = action - delta_action
        action = action.clamp(min=self.action_bounds[0], max=self.action_bounds[1])

        return action

    @no_ebm_grad()
    def _langevin_action_given_obs(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            ebm: nn.Module,
            scheduler: BaseScheduler = None
    ) -> torch.Tensor:
        """
        Overview:
            Run langevin MCMC for `self.iters` steps.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - action (:obj:`torch.Tensor`): Actions.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
            - scheduler (:obj:`BaseScheduler`): Learning rate scheduler.
        Returns:
            - action (:obj:`torch.Tensor`): Actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
        """
        if not scheduler:
            self.stepsize_scheduler['num_steps'] = self.iters
            scheduler = MCMC.PolynomialScheduler(**self.stepsize_scheduler)
        stepsize = scheduler.get_rate(-1)
        for i in range(self.iters):
            action = self._langevin_step(obs, action, stepsize, ebm)
            stepsize = scheduler.get_rate(i)
        return action

    @no_ebm_grad()
    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Create tiled observations and sample counter-negatives for InfoNCE loss.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - tiled_obs (:obj:`torch.Tensor`): Tiled observations.
            - action_samples (:obj:`torch.Tensor`): Action samples.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.
            - action_samples (:obj:`torch.Tensor`): :math:`(B, N, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> ebm = EBM(4, 5)
            >>> opt = MCMC()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> tiled_obs, action_samples = opt.sample(obs, ebm)
        """
        obs, uniform_action_samples = self._sample(obs, self.train_samples)
        if not self.use_langevin_negative_samples:
            return obs, uniform_action_samples
        langevin_action_samples = self._langevin_action_given_obs(obs, uniform_action_samples, ebm)
        return obs, langevin_action_samples

    @no_ebm_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """
        Overview:
            Optimize for the best action conditioned on the current observation.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observations.
            - ebm (:obj:`torch.nn.Module`): Energy based model.
        Returns:
            - best_action_samples (:obj:`torch.Tensor`): Actions.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.
            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.
            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.
        Examples:
            >>> obs = torch.randn(2, 4)
            >>> ebm = EBM(4, 5)
            >>> opt = MCMC()
            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))
            >>> best_action_samples = opt.infer(obs, ebm)
        """
        # (B, N, O), (B, N, A)
        obs, uniform_action_samples = self._sample(obs, self.inference_samples)
        action_samples = self._langevin_action_given_obs(
            obs,
            uniform_action_samples,
            ebm,
        )

        # Run a second optimization, a trick for more precise inference
        if self.optimize_again:
            self.again_stepsize_scheduler['num_steps'] = self.iters
            action_samples = self._langevin_action_given_obs(
                obs,
                action_samples,
                ebm,
                scheduler=MCMC.PolynomialScheduler(**self.again_stepsize_scheduler),
            )

        # action_samples: B, N, A
        return self._get_best_action_sample(obs, action_samples, ebm)


@MODEL_REGISTRY.register('ebm')
class EBM(nn.Module):
    """
    Overview:
        Energy based model.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size: int = 512,
        hidden_layer_num: int = 4,
        **kwargs,
    ):
        """
        Overview:
            Initialize the EBM.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape.
            - action_shape (:obj:`int`): Action shape.
            - hidden_size (:obj:`int`): Hidden size.
            - hidden_layer_num (:obj:`int`): Number of hidden layers.
        """
        super().__init__()
        input_size = obs_shape + action_shape
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            RegressionHead(
                hidden_size,
                1,
                hidden_layer_num,
                final_tanh=False,
            )
        )

    def forward(self, obs, action):
        """
        Overview:
            Forward computation graph of EBM.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observation of shape (B, N, O).
            - action (:obj:`torch.Tensor`): Action of shape (B, N, A).
        Returns:
            - pred (:obj:`torch.Tensor`): Energy of shape (B, N).
        Examples:
            >>> obs = torch.randn(2, 3, 4)
            >>> action = torch.randn(2, 3, 5)
            >>> ebm = EBM(4, 5)
            >>> pred = ebm(obs, action)
        """
        x = torch.cat([obs, action], -1)
        x = self.net(x)
        return x['pred']


@MODEL_REGISTRY.register('arebm')
class AutoregressiveEBM(nn.Module):
    """
    Overview:
        Autoregressive energy based model.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size: int = 512,
        hidden_layer_num: int = 4,
    ):
        """
        Overview:
            Initialize the AutoregressiveEBM.
        Arguments:
            - obs_shape (:obj:`int`): Observation shape.
            - action_shape (:obj:`int`): Action shape.
            - hidden_size (:obj:`int`): Hidden size.
            - hidden_layer_num (:obj:`int`): Number of hidden layers.
        """
        super().__init__()
        self.ebm_list = nn.ModuleList()
        for i in range(action_shape):
            self.ebm_list.append(EBM(obs_shape, i + 1, hidden_size, hidden_layer_num))

    def forward(self, obs, action):
        """
        Overview:
            Forward computation graph of AutoregressiveEBM.
        Arguments:
            - obs (:obj:`torch.Tensor`): Observation of shape (B, N, O).
            - action (:obj:`torch.Tensor`): Action of shape (B, N, A).
        Returns:
            - pred (:obj:`torch.Tensor`): Energy of shape (B, N, A).
        Examples:
            >>> obs = torch.randn(2, 3, 4)
            >>> action = torch.randn(2, 3, 5)
            >>> arebm = AutoregressiveEBM(4, 5)
            >>> pred = arebm(obs, action)
        """
        output_list = []
        for i, ebm in enumerate(self.ebm_list):
            output_list.append(ebm(obs, action[..., :i + 1]))
        return torch.stack(output_list, axis=-1)
