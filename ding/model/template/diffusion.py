from typing import Union, List, Dict
from collections import namedtuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import list_split, MODEL_REGISTRY, squeeze, SequenceType
from ding.torch_utils.network.diffusion import extract, cosine_beta_schedule, apply_conditioning, \
    DiffusionUNet1d, TemporalValue

Sample = namedtuple('Sample', 'trajectories values chains')


def default_sample_fn(model, x, cond, t):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x,
        cond=cond,
        t=t,
    )
    noise = 0.5 * torch.randn_like(x)
    # no noise when t == 0
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1, ) * (len(x.shape) - 1)))
    values = torch.zeros(len(x), device=device)
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, values


def get_guide_output(guide, x, cond, t, returns=None, is_dynamic=False, act_dim=6):
    x.requires_grad_()
    if returns is not None:
        if not is_dynamic:
            y = guide(x, cond, t, returns).squeeze(dim=-1)
        else:
            returns = returns.unsqueeze(1).repeat_interleave(x.shape[1],dim=1)
            input = torch.cat([x, returns], dim=-1)
            input = input.reshape(-1, input.shape[-1])
            y = guide(input)
            y = y.reshape(x.shape[0], x.shape[1], -1)
            y = F.mse_loss(x[:, 1:, act_dim:], y[:, :-1], reduction='none')
    else:
        y = guide(x, cond, t).squeeze(dim=-1)
    grad = torch.autograd.grad([y.sum()], [x])[0]
    x.detach()
    return y, grad


def n_step_guided_p_sample(
    model,
    x,
    cond,
    t,
    guide,
    scale=0.001,
    t_stopgrad=0,
    n_guide_steps=1,
    scale_grad_by_std=True,
):
    """
    Overview:
        Guidance fn for Diffusion
    Arguments:
        - model (obj: 'class') diffusion model
        - x (obj: 'tensor') input for guidance
        - cond (obj: 'tensor') cond of input
        - guide (obj: 'class') guide function
    """
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = get_guide_output(guide, x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

def free_guidance_sample(
    model,
    x,
    cond,
    t,
    guide1,
    guide2,
    returns=None,
    scale=1,
    t_stopgrad=0,
    n_guide_steps=1,
    scale_grad_by_std=True,
    
):
    """
    Overview:
        Guidance fn for MetaDiffusion
    Arguments:
        - model (obj: 'class') diffusion model
        - x (obj: 'tensor') input for guidance
        - cond (obj: 'tensor') cond of input
        - guide1 (obj: 'class') guide function. In MetaDiffusion is reward function
        - guide2 (obj: 'class') guide function. In MetaDiffusion is dynamic function
        - returns (obj: 'tensor') for MetaDiffusion, it is id for task.

    """
    weight = extract(model.sqrt_one_minus_alphas_cumprod, t, x.shape)
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)
    
    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y1, grad1 = get_guide_output(guide1, x, cond, t, returns) # get reward
            y2, grad2 = get_guide_output(guide2, x, cond, t, returns, is_dynamic=True, 
                                         act_dim=model.action_dim) # get state
            grad = grad1 + scale * grad2

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        if model.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = model.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = model.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + model.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = model.model(x, cond, t)
        epsilon -= weight * grad

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t, epsilon=epsilon)
    # model_std = torch.exp(0.5 * model_log_variance)
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y1

class GaussianDiffusion(nn.Module):
    """
    Overview:
            Gaussian diffusion model
    Arguments:
            - model (:obj:`str`): type of model
            - model_cfg (:obj:'dict') config of model
            - horizon (:obj:`int`): horizon of trajectory
            - obs_dim (:obj:`int`): Dim of the ovservation
            - action_dim (:obj:`int`): Dim of the ation
            - n_timesteps (:obj:`int`): Number of timesteps
            - predict_epsilon (:obj:'bool'): Whether predict epsilon
            - loss_discount (:obj:'float'): discount of loss
            - clip_denoised (:obj:'bool'): Whether use clip_denoised
            - action_weight (:obj:'float'): weight of action
            - loss_weights (:obj:'dict'): weight of loss
    """

    def __init__(
            self,
            model: str,
            model_cfg: dict,
            horizon: int,
            obs_dim: Union[int, SequenceType],
            action_dim: Union[int, SequenceType],
            n_timesteps: int = 1000,
            predict_epsilon: bool = True,
            loss_discount: float = 1.0,
            clip_denoised: bool = False,
            action_weight: float = 1.0,
            loss_weights: dict = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.transition_dim = obs_dim + action_dim
        if type(model) == str:
            model = eval(model)
        self.model = model(**model_cfg)
        self.predict_epsilon = predict_epsilon
        self.clip_denoised = clip_denoised

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        )

        self.loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)

    def get_loss_weights(self, action_weight: float, discount: float, weights_dict: dict):
        """
        Overview:
            sets loss coefficients for trajectory
        Arguments:
            - action_weight (:obj:'float') coefficient on first action loss
            - discount (:obj:'float') multiplies t^th timestep of trajectory loss by discount**t
            - weights_dict (:obj:'dict') { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        Overview:
            give noise and step, compute mean, variance.
        Arguments:
            x_start (:obj:'tensor') noise trajectory in timestep 0
            x_t (:obj:'tuple') noise trajectory in timestep t
            t (:obj:'int') timestep of diffusion step
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, return_chain=False, sample_fn=default_sample_fn, plan_size=1, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            if return_chain:
                chain.append(x)
        values = values.reshape(-1, plan_size, *values.shape[1:])
        x = x.reshape(-1, plan_size, *x.shape[1:])
        if plan_size > 1:
            inds = torch.argsort(values, dim=1, descending=True)
            x = x[torch.arange(x.size(0)).unsqueeze(1), inds]
            values = values[torch.arange(values.size(0)).unsqueeze(1), inds]
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    def q_sample(self, x_start, t, noise=None):
        """
        Arguments:
            conditions (:obj:'tuple') [ (time, state), ... ] conditions of diffusion
            t (:obj:'int') timestep of diffusion
            noise (:obj:'tensor.float') timestep's noise of diffusion
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction='none')
            a0_loss = (loss[:, 0, :self.action_dim] / self.loss_weights[0, :self.action_dim].to(loss.device)).mean()
            loss = (loss * self.loss_weights.to(loss.device)).mean()
        else:
            loss = F.mse_loss(x_recon, x_start, reduction='none')
            a0_loss = (loss[:, 0, :self.action_dim] / self.loss_weights[0, :self.action_dim].to(loss.device)).mean()
            loss = (loss * self.loss_weights.to(loss.device)).mean()
        return loss, a0_loss

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):
    """
    Overview:
            Gaussian diffusion model for value function.
    """

    def p_losses(self, x_start, cond, target, t, returns=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t, returns)
        loss = F.mse_loss(pred, target, reduction='none').mean()
        with torch.no_grad():
            r0_loss = F.mse_loss(pred[:, 0], target[:,0])
        log = {
            'mean_pred': pred.mean().item(),
            'max_pred': pred.max().item(),
            'min_pred': pred.min().item(),
            'r0_loss': r0_loss.mean().item(),
        }
        

        return loss, log

    def forward(self, x, cond, t, returns=None):
        return self.model(x, cond, t, returns)


@MODEL_REGISTRY.register('pd')
class PlanDiffuser(nn.Module):
    """
    Overview:
            Diffuser model for plan.
    Arguments:
            - diffuser_model (:obj:`str`): type of plan model
            - diffuser_model_cfg (:obj:'dict') config of diffuser_model
            - value_model (:obj:`str`): type of value model, if haven't use, set it as None
            - value_model_cfg (:obj:`int`): config of value_model
            - sample_kwargs : config of sample function
    """

    def __init__(
        self, diffuser_model: str, diffuser_model_cfg: dict, value_model: str, value_model_cfg: dict, **sample_kwargs
    ):
        super().__init__()
        diffuser_model = eval(diffuser_model)
        self.diffuser = diffuser_model(**diffuser_model_cfg)
        self.value = None
        if value_model:
            value_model = eval(value_model)
            self.value = value_model(**value_model_cfg)
        self.sample_kwargs = sample_kwargs

    def diffuser_loss(self, x_start, cond, t):
        return self.diffuser.p_losses(x_start, cond, t)

    def value_loss(self, x_start, cond, target, t):
        return self.value.p_losses(x_start, cond, target, t)

    def get_eval(self, cond, batch_size=1):
        cond = self.repeat_cond(cond, batch_size)
        if self.value:
            samples = self.diffuser(
                cond, sample_fn=n_step_guided_p_sample, plan_size=batch_size, guide=self.value, **self.sample_kwargs
            )
            # extract action [eval_num, batch_size, horizon, transition_dim]
            actions = samples.trajectories[:, :, :, :self.diffuser.action_dim]
            action = actions[:, 0, 0]
            return action
        else:
            samples = self.diffuser(cond, plan_size=batch_size)
            return samples.trajectories[:, :, :, self.diffuser.action_dim:].squeeze(1)

    def repeat_cond(self, cond, batch_size):
        for k, v in cond.items():
            cond[k] = v.repeat_interleave(batch_size, dim=0)
        return cond


@MODEL_REGISTRY.register('dd')
class GaussianInvDynDiffusion(nn.Module):
    """
    Overview:
            Gaussian diffusion model with Invdyn action model.
    Arguments:
            - model (:obj:`str`): type of model
            - model_cfg (:obj:'dict') config of model
            - horizon (:obj:`int`): horizon of trajectory
            - obs_dim (:obj:`int`): Dim of the ovservation
            - action_dim (:obj:`int`): Dim of the ation
            - n_timesteps (:obj:`int`): Number of timesteps
            - hidden_dim (:obj:'int'): hidden dim of inv_model
            - returns_condition (:obj:'bool'): Whether use returns condition
            - ar_inv (:obj:'bool'): Whether use inverse action learning
            - train_only_inv (:obj:'bool'): Whether train inverse action model only
            - predict_epsilon (:obj:'bool'): Whether predict epsilon
            - condition_guidance_w (:obj:'float'): weight of condition guidance
            - loss_discount (:obj:'float'): discount of loss
    """

    def __init__(
            self,
            model: str,
            model_cfg: dict,
            horizon: int,
            obs_dim: Union[int, SequenceType],
            action_dim: Union[int, SequenceType],
            n_timesteps: int = 1000,
            hidden_dim: int = 256,
            returns_condition: bool = False,
            ar_inv: bool = False,
            train_only_inv: bool = False,
            predict_epsilon: bool = True,
            condition_guidance_w: float = 0.1,
            loss_discount: float = 1.0,
            clip_denoised: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.transition_dim = obs_dim + action_dim
        if type(model) == str:
            model = eval(model)
        self.model = model(**model_cfg)
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        self.predict_epsilon = predict_epsilon
        self.condition_guidance_w = condition_guidance_w

        self.inv_model = nn.Sequential(
            nn.Linear(2 * self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

        self.returns_condition = returns_condition
        self.clip_denoised = clip_denoised

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        )

        self.loss_weights = self.get_loss_weights(loss_discount)

    def get_loss_weights(self, discount: int):
        self.action_weight = 1
        dim_weights = torch.ones(self.obs_dim, dtype=torch.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        Arguments:
            x_start (:obj:'tensor') noise trajectory in timestep 0
            x_t (:obj:'tuple') noise trajectory in timestep t
            t (:obj:'int') timestep of diffusion step
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        """
        Arguments:
            x (:obj:'tensor') noise trajectory in timestep t
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            t (:obj:'int') timestep of diffusion step
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
        returns:
            model_mean (:obj:'tensor.float')
            posterior_variance (:obj:'float')
            posterior_log_variance (:obj:'float')
        """
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        """
        Arguments:
            x (:obj:'tensor') noise trajectory in timestep t
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            t (:obj:'int') timestep of diffusion step
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        """
        Arguments:
            shape (:obj:'tuple') (batch_size, horizon, self.obs_dim)
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
            horizon (:obj:'int') horizon of trajectory
            verbose (:obj:'bool') whether log diffusion progress
            return_diffusion (:obj:'bool') whether use return diffusion
        """
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        # In this model, init state must be given by the env and without noise.
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion:
            diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        """
        Arguments:
            conditions (:obj:'tuple') [ (time, state), ... ] state is init state of env, time is timestep of trajectory
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
            horizon (:obj:'int') horizon of trajectory
        returns:
            x (:obj:'tensor') tarjctory of env
        """
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.obs_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def q_sample(self, x_start, t, noise=None):
        """
        Arguments:
            conditions (:obj:'tuple') [ (time, state), ... ] conditions of diffusion
            t (:obj:'int') timestep of diffusion
            noise (:obj:'tensor.float') timestep's noise of diffusion
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction='none')
            loss = (loss * self.loss_weights.to(loss.device)).mean()
        else:
            loss = F.mse_loss(x_recon, x_start, reduction='none')
            loss = (loss * self.loss_weights.to(loss.device)).mean()

        return loss

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

class GuidenceFreeDifffuser(GaussianDiffusion):
    """
    Overview:
            Gaussian diffusion model with guidence
    Arguments:
            - model (:obj:`str`): type of model
            - model_cfg (:obj:'dict') config of model
            - horizon (:obj:`int`): horizon of trajectory
            - obs_dim (:obj:`int`): Dim of the ovservation
            - action_dim (:obj:`int`): Dim of the ation
            - n_timesteps (:obj:`int`): Number of timesteps
            - predict_epsilon (:obj:'bool'): Whether predict epsilon
            - loss_discount (:obj:'float'): discount of loss
            - clip_denoised (:obj:'bool'): Whether use clip_denoised
            - action_weight (:obj:'float'): weight of action
            - loss_weights (:obj:'dict'): weight of loss
            - returns_condition (:obj:'bool') whether use additional condition
            - condition_guidance_w (:obj:'float') guidance weight
    """

    def __init__(
            self,
            model: str,
            model_cfg: dict,
            horizon: int,
            obs_dim: Union[int, SequenceType],
            action_dim: Union[int, SequenceType],
            n_timesteps: int = 1000,
            predict_epsilon: bool = True,
            loss_discount: float = 1.0,
            clip_denoised: bool = False,
            action_weight: float = 1.0,
            loss_weights: dict = None,
            returns_condition: bool = False,
            condition_guidance_w: float = 0.1,
    ):
        super().__init__(model, model_cfg, horizon, obs_dim, action_dim, n_timesteps, predict_epsilon,
                       loss_discount, clip_denoised, action_weight, loss_weights,)
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

    def p_mean_variance(self, x, cond, t, epsilon):
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, shape, cond, sample_fn=None, plan_size=1, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        assert sample_fn != None
        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

        values = values.reshape(-1, plan_size, *values.shape[1:])
        x = x.reshape(-1, plan_size, *x.shape[1:])
        if plan_size > 1:
            inds = torch.argsort(values, dim=1, descending=True)
            inds = inds.unsqueeze(-1).expand_as(x)
            x = x.gather(1, inds)
        return x[:,0]

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.obs_dim + self.action_dim)
        return self.p_sample_loop(shape, cond, **sample_kwargs)
    
    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)        

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction='none')
            a0_loss = (loss[:, 0, :self.action_dim] / self.loss_weights[0, :self.action_dim].to(loss.device)).mean()
            loss = (loss * self.loss_weights.to(loss.device)).mean()
        else:
            loss = F.mse_loss(x_recon, x_start, reduction='none')
            a0_loss = (loss[:, 0, :self.action_dim] / self.loss_weights[0, :self.action_dim].to(loss.device)).mean()
            loss = (loss * self.loss_weights.to(loss.device)).mean()
        return loss, a0_loss
        

@MODEL_REGISTRY.register('metadiffuser')
class MetaDiffuser(nn.Module):
    """
    Overview:
            MetaDiffusion model
    Arguments:
            - dim (:obj:`int`): dim of emb and dynamic model
            - obs_dim (:obj:`int`): Dim of the ovservation
            - action_dim (:obj:`int`): Dim of the ation
            - reward_cfg (:obj:'dict') config of reward model
            - diffuser_model_cfg (:obj:'dict') config of diffuser_model
            - horizon (:obj:`int`): horizon of trajectory
            - encoder_horizon (:obj:`int`): horizon of emb model
            - sample_kwargs : config of sample function
    """
    def __init__(
            self,
            dim: int,
            obs_dim: Union[int, SequenceType],
            action_dim: Union[int, SequenceType],
            reward_cfg: dict,
            diffuser_model_cfg: dict,
            horizon: int,
            encoder_horizon: int,
            **sample_kwargs,
            ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.sample_kwargs = sample_kwargs
        self.encoder_horizon = encoder_horizon

        self.embed = nn.Sequential(
            nn.Linear((obs_dim * 2 + action_dim + 1) * encoder_horizon, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.reward_model = ValueDiffusion(**reward_cfg)

        self.dynamic_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim + dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, obs_dim),
        )

        self.diffuser = GuidenceFreeDifffuser(**diffuser_model_cfg)

    def get_task_id(self, traj):
        """
        Overview:
            get task id for trajectory
        Arguments:
            - traj (:obj:'tensor') trajectory of env
        """
        input_emb = traj.reshape(traj.shape[0], -1)
        task_idx = self.embed(input_emb)
        return task_idx

    def diffuser_loss(self, x_start, cond, t, returns=None):
        return self.diffuser.p_losses(x_start, cond, t, returns)
    
    def pre_train_loss(self, traj, target, t, cond):
        """
        Overview:
            train dynamic, reward and embed model.
        Arguments:
            - traj (:obj:'tensor') traj for dataset, include: obs, next_obs, action, reward
            - target (:obj:'tensor') target obs and rerward
            - t (:obj:'int') step
            - cond (:obj:'tensor') condition of input
        """
        encoder_traj = traj[:, :self.encoder_horizon]
        input_emb = encoder_traj.reshape(target.shape[0], -1)
        task_idx = self.embed(input_emb)

        states = traj[:, :, self.action_dim:self.action_dim + self.obs_dim]
        actions = traj[:, :, :self.action_dim]
        input = torch.cat([actions, states], dim=-1)
        target_reward = target[:, :, -1]

        target_next_state = target[:, :, :self.obs_dim].reshape(-1, self.obs_dim)

        reward_loss, reward_log = self.reward_model.p_losses(input, cond, target_reward, t, task_idx)
        
        
        task_idxs = task_idx.unsqueeze(1).repeat_interleave(self.horizon, dim=1)

        input = torch.cat([input, task_idxs], dim=-1)
        input = input.reshape(-1, input.shape[-1])

        next_state = self.dynamic_model(input)
        state_loss = F.mse_loss(next_state, target_next_state, reduction='none').mean()
        
        return state_loss, reward_loss, reward_log

    def get_eval(self, cond, id = None, batch_size = 1):
        """
        Overview:
            get action
        Arguments:
            - cond (:obj:'tensor') condition for sample
            - id (:obj:'tensor') id for task.
        """
        id = torch.stack(id, dim=0)
        if batch_size > 1:
            cond = self.repeat_cond(cond, batch_size)
            id = id.unsqueeze(1).repeat_interleave(batch_size, dim=1)
            id = id.reshape(-1, id.shape[-1])
        
        samples = self.diffuser(cond, returns=id, sample_fn=free_guidance_sample, plan_size=batch_size,
                                guide1=self.reward_model, guide2=self.dynamic_model, **self.sample_kwargs)
        return samples[:, 0, :self.action_dim]

    def repeat_cond(self, cond, batch_size):
        for k, v in cond.items():
            cond[k] = v.repeat_interleave(batch_size, dim=0)
        return cond
