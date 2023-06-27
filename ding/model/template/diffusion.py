from typing import Union, List, Dict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import list_split, MODEL_REGISTRY, squeeze, SequenceType



def extract(a, t, x_shape):
    '''
    Overview:
        extract output from a through index t.
    '''
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype = torch.float32):
    '''
    Overview:
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Return:
        Tensor of beta [timesteps,], computing by cosine.
    '''
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    '''
    Overview:
        add condition into x
    '''
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x

class Mish(nn.Module):
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class conv1d(nn.Module):
    """
    Overview:
        conv1dblock network
    Interface:
        __init__, forward
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            activation: nn.Module = None,
            n_groups: int = 8
    ) -> None:
        """
        Overview:
            Create a 1-dim convlution layer with activation and normalization.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor
            - out_channels (:obj:`int`): Number of channels in the output tensor
            - kernel_size (:obj:`int`): Size of the convolving kernel
            - padding (:obj:`int`): Zero-padding added to both sides of the input
            - activation (:obj:`nn.Module`): the optional activation function
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = activation

    def forward(self, inputs):
        """
        Overview:
            compute conv1d for inputs.
        """
        x = self.conv1(inputs)
        # [batch, channels, horizon] -> [batch, channels, 1, horizon]
        x = x.unsqueeze(-2)
        x = self.norm(x)
        # [batch, channels, 1, horizon] -> [batch, channels, horizon]
        x = x.squeeze(-2)
        out = self.act(x)
        return out

class SinusoidalPosEmb(nn.Module):
    '''
    Overview:
        compute sin position embeding
    '''
    def __init__(self, dim: int,) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

class ResidualTemporalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            embed_dim: int,
            kernel_size: int = 5,
            mish: bool = True
    ) -> None:
        super().__init__()
        if mish:
            act = Mish()
        else:
            act = SiLU()
        self.blocks = nn.ModuleList([
            conv1d(in_channels, out_channels, kernel_size, kernel_size // 2, act),
            conv1d(out_channels, out_channels, kernel_size, kernel_size // 2, act),
        ])
        self.time_mlp = nn.Sequential(
            act,
            nn.Linear(embed_dim, out_channels),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t).unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
            self,
            transition_dim: int,
            dim: int = 128,
            dim_mults: SequenceType = [1, 2, 4, 8],
            returns_condition: bool = False,
            condition_dropout: float = 0.1,
            calc_energy: bool = False,
            kernel_size: int = 5,
    ) -> None:
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if calc_energy:
            mish = False
            act = SiLU()
        else:
            mish = True
            act = Mish()

        self.time_dim = dim
        self.returns_dim = dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.cale_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act,
                nn.Linear(dim, dim * 4),
                act,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = torch.distributions.Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolution = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolution - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim, kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim, kernel_size, mish=mish),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity()
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size, mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size, mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolution - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim, kernel_size, mish=mish),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        self.final_conv = nn.Sequential(
            conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, activation=act),
            nn.Conv1d(dim, transition_dim, 1),
        )
    
    def forward(self, x, cond, time, returns = None, use_dropout: bool = True, 
                force_dropout: bool = False):
        '''
        Arguments:
            x (:obj:'tensor') noise trajectory
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            time (:obj:'int') timestep of diffusion step
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
            use_dropout (:obj:'bool') Whether use returns condition mask
            force_dropout (:obj:'bool') Whether use returns condition
        '''
        if self.cale_energy:
            x_inp = x

        # [batch, horizon, transition ] -> [batch, transition , horizon]
        x = x.transpose(1, 2)
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        # [batch, transition , horizon] -> [batch, horizon, transition ]
        x = x.transpose(1, 2)

        if self.cale_energy:
            # Energy function 
            energy = ((x - x_inp) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x
        
    def get_pred(self, x, cond, time, returns: bool = None, use_dropout: bool = True, 
                 force_dropout: bool = False):
        # [batch, horizon, transition ] -> [batch, transition , horizon]
        x = x.transpose(1, 2)
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        # [batch, transition , horizon] -> [batch, horizon, transition ]
        x = x.transpose(1, 2)
        return x
    
class TemporalValue(nn.Module):
    def __init__(
            self,
            horizon: int,
            transition_dim: int,
            dim: int = 32,
            time_dim: int = None,
            out_dim: int = 1,
            dim_mults: SequenceType = [1, 2, 4, 8],
    ) -> None:
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.blocks = nn.ModuleList([])
        
        for dim_in, dim_out in in_out:
            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1)
            ]))
            horizon = horizon // 2
        
        fc_dim = dims[-1] * max(horizon, 1)
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        # [batch, horizon, transition ] -> [batch, transition , horizon]
        x = x.transpose(1, 2)
        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out

class MLPnet(nn.Module):
    def __init__(
            self,
            transition_dim: int,
            cond_dim: int,
            dim: int = 128,
            returns_condition: bool = True,
            condition_dropout: float = 0.1,
            calc_energy: bool = False,
    ) -> None:
        super().__init__()
        if calc_energy:
            act = SiLU()
        else:
            act = Mish()
        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act,
            nn.Linear(dim * 4, dim),
        )
        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act,
                nn.Linear(dim, dim * 4),
                act,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = torch.distributions.Bernoulli(probs=1 - self.condition_dropout)
            embed_dim = 2 * dim
        else:
            embed_dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + transition_dim, 1024),
            act,
            nn.Linear(1024, 1024),
            act,
            nn.Linear(1024, self.action_dim),
        )

    def forward(self, x, cond, time, returns=None, use_dropout: bool = True, force_dropout: bool = False):
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            else:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)
        
        inputs = torch.cat([t, cond, x], dim=-1)
        out = self.mlp(inputs)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out
        
class ARInvModel(nn.Module):
    '''
    Overview:
        Action model, return action by given state and next state
    '''
    def __init__(
            self,
            hidden_dim: int,
            obs_dim: int,
            action_dim: int,
            low_act: float = -1.0,
            up_act: float = 1.0
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic = False):
        state_inp = comb_state
        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()
        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()            
        action = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat[state_d, self.lin_mod[i - 1](torch.cat(action, dim=1))], dim=1)
            l_i = torch.distributions.Categorical(logits=lp_i).sample()
            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                self.low_act + (l_i + 1) * self.bin_size).sample()
            action.append(a_i.unsqueeze(1))
        return torch.cat(action, dim=1)
    
    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim

@MODEL_REGISTRY.register('dd')
class GaussianInvDynDiffusion(nn.Module):
    '''
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
    '''
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
        if ar_inv:
            self.inv_model = ARInvModel(hidden_dim, obs_dim, action_dim)
        else:
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

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        self.loss_weights = self.get_loss_weights(loss_discount)
    
    def get_loss_weights(self, discount: int):
        self.action_weight = 1
        dim_weights = torch.ones(self.obs_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        '''
        Arguments:
            x_start (:obj:'tensor') noise trajectory in timestep 0
            x_t (:obj:'tuple') noise trajectory in timestep t
            t (:obj:'int') timestep of diffusion step
        '''
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        '''
        Arguments:
            x (:obj:'tensor') noise trajectory in timestep t
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            t (:obj:'int') timestep of diffusion step
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
        returns:
            model_mean (:obj:'tensor.float') 
            posterior_variance (:obj:'float')
            posterior_log_variance (:obj:'float')
        '''
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        '''
        Arguments:
            x (:obj:'tensor') noise trajectory in timestep t
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            t (:obj:'int') timestep of diffusion step
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
        '''
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        '''
        Arguments:
            shape (:obj:'tuple') (batch_size, horizon, self.obs_dim)
            cond (:obj:'tuple') [ (time, state), ... ] state is init state of env, time = 0
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
            horizon (:obj:'int') horizon of trajectory
            verbose (:obj:'bool') whether log diffusion progress
            return_diffusion (:obj:'bool') whether use return diffusion
        '''
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        # In this model, init state must be given by the env and without noise.
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, 0)


            if return_diffusion: diffusion.append(x)


        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
        Arguments:
            conditions (:obj:'tuple') [ (time, state), ... ] state is init state of env, time is timestep of trajectory
            returns (:obj:'tensor') condition returns of trajectory, returns is normal return
            horizon (:obj:'int') horizon of trajectory
        returns:
            x (:obj:'tensor') tarjctory of env
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.obs_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    
    def q_sample(self, x_start, t, noise=None):
        '''
        Arguments:
            conditions (:obj:'tuple') [ (time, state), ... ] conditions of diffusion
            t (:obj:'int') timestep of diffusion
            noise (:obj:'tensor.float') timestep's noise of diffusion
        '''
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
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

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
