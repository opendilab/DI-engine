from typing import Union, List, Dict
from collections import namedtuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import list_split, MODEL_REGISTRY, squeeze, SequenceType


def extract(a, t, x_shape):
    """
    Overview:
        extract output from a through index t.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32):
    """
    Overview:
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Return:
        Tensor of beta [timesteps,], computing by cosine.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim, mask = None):
    """
    Overview:
        add condition into x
    """
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x

class DiffusionConv1d(nn.Module):

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
            Create a 1-dim convlution layer with activation and normalization. This Conv1d have GropuNorm.
            And need add 1-dim when compute norm
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
    """
    Overview:
        compute sin position embeding
    """

    def __init__(
            self,
            dim: int,
    ) -> None:
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


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *arg, **kwargs):
        return self.fn(x, *arg, **kwargs) + x


class LayerNorm(nn.Module):
    """
    Overview: LayerNorm, compute dim = 1, because Temporal input x [batch, dim, horizon]
    """

    def __init__(self, dim, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    """
    Overview:
        Linear Attention head
    Arguments:
        - dim (:obj:'int'): dim of input
        - heads (:obj:'int'): num of head
        - dim_head (:obj:'int'): dim of head
    """

    def __init__(self, dim, heads=4, dim_head=32) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], self.heads, -1, t.shape[-1]), qkv)
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.reshape(out.shape[0], -1, out.shape[-1])
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):
    """
    Overview:
        Residual block of temporal
    Arguments:
        - in_channels (:obj:'int'): dim of in_channels
        - out_channels (:obj:'int'): dim of out_channels
        - embed_dim (:obj:'int'): dim of embeding layer
        - kernel_size (:obj:'int'): kernel_size of conv1d
        - mish (:obj:'bool'): whether use mish as a activate function
    """

    def __init__(
            self, in_channels: int, out_channels: int, embed_dim: int, kernel_size: int = 5, mish: bool = True
    ) -> None:
        super().__init__()
        if mish:
            act = nn.Mish()
        else:
            act = nn.SiLU()
        self.blocks = nn.ModuleList(
            [
                DiffusionConv1d(in_channels, out_channels, kernel_size, kernel_size // 2, act),
                DiffusionConv1d(out_channels, out_channels, kernel_size, kernel_size // 2, act),
            ]
        )
        self.time_mlp = nn.Sequential(
            act,
            nn.Linear(embed_dim, out_channels),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        out = self.blocks[0](x)
        out += self.time_mlp(t).unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class DiffusionUNet1d(nn.Module):

    def __init__(
            self,
            transition_dim: int,
            dim: int = 32,
            returns_dim: int = 1,
            dim_mults: SequenceType = [1, 2, 4, 8],
            returns_condition: bool = False,
            condition_dropout: float = 0.1,
            calc_energy: bool = False,
            kernel_size: int = 5,
            attention: bool = False,
    ) -> None:
        """
        Overview:
            temporal net
        Arguments:
            - transition_dim (:obj:'int'): dim of transition, it is obs_dim + action_dim
            - dim (:obj:'int'): dim of layer
            - dim_mults (:obj:'SequenceType'): mults of dim
            - returns_condition (:obj:'bool'): whether use return as a condition
            - condition_dropout (:obj:'float'): dropout of returns condition
            - calc_energy (:obj:'bool'): whether use calc_energy
            - kernel_size (:obj:'int'): kernel_size of conv1d
            - attention (:obj:'bool'): whether use attention
        """
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if calc_energy:
            mish = False
            act = nn.SiLU()
        else:
            mish = True
            act = nn.Mish()

        self.time_dim = dim
        self.returns_dim = returns_dim

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
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim, kernel_size, mish=mish),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim, kernel_size, mish=mish),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity()
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size, mish)
        self.mid_atten = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim, kernel_size, mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolution - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim, kernel_size, mish=mish),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim, kernel_size, mish=mish),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, activation=act),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns = None, use_dropout: bool = True, force_dropout: bool = False):
        """
        Arguments:
            x (:obj:'tensor'): noise trajectory
            cond (:obj:'tuple'): [ (time, state), ... ] state is init state of env, time = 0
            time (:obj:'int'): timestep of diffusion step
            returns (:obj:'tensor'): condition returns of trajectory, returns is normal return
            use_dropout (:obj:'bool'): Whether use returns condition mask
            force_dropout (:obj:'bool'): Whether use returns condition
        """
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

        for resnet, resnet2, atten, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = atten(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_atten(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, atten, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = atten(x)
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

    def get_pred(self, x, cond, time, returns = None, use_dropout: bool = True, force_dropout: bool = False):
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
    """
    Overview:
        temporal net for value function
    Arguments:
        - horizon (:obj:'int'): horizon of trajectory
        - transition_dim (:obj:'int'): dim of transition, it is obs_dim + action_dim
        - dim (:obj:'int'): dim of layer
        - time_dim (:obj:'): dim of time
        - dim_mults (:obj:'SequenceType'): mults of dim
        - kernel_size (:obj:'int'): kernel_size of conv1d
        - returns_condition (:obj:'bool'): whether use an additionly condition
    """

    def __init__(
            self,
            horizon: int,
            transition_dim: int,
            dim: int = 32,
            time_dim: int = None,
            out_dim: int = 1,
            kernel_size: int = 5,
            dim_mults: SequenceType = [1, 2, 4, 8],
            returns_condition: bool = False,
            no_need_ret_sin: bool =False,
    ) -> None:
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        if returns_condition:
            time_dim += time_dim
            if not no_need_ret_sin:
                self.returns_mlp = nn.Sequential(
                    SinusoidalPosEmb(dim),
                    nn.Linear(dim, dim * 4),
                    nn.Mish(),
                    nn.Linear(dim * 4, dim),
                )
            else:
                self.returns_mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.Mish(),
                    nn.Linear(dim * 4, dim),
                )
        self.blocks = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.blocks.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, kernel_size=kernel_size, embed_dim=time_dim),
                        ResidualTemporalBlock(dim_out, dim_out, kernel_size=kernel_size, embed_dim=time_dim),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                    ]
                )
            )

            horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=kernel_size, embed_dim=time_dim)
        self.mid_down1 = nn.Conv1d(mid_dim_2, mid_dim_2, 3, 2, 1)

        horizon = horizon // 2
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=kernel_size, embed_dim=time_dim)
        self.mid_down2 = nn.Conv1d(mid_dim_3, mid_dim_3, 3, 2, 1)
        horizon = horizon // 2

        fc_dim = mid_dim_3 * max(horizon, 1)
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, returns=None, *args):
        # [batch, horizon, transition ] -> [batch, transition , horizon]
        x = x.transpose(1, 2)
        t = self.time_mlp(time)
        
        if returns is not None:
            returns_embed = self.returns_mlp(returns)
            t = torch.cat([t, returns_embed], dim=-1)

        for resnet, resnet2, downsample in self.blocks:
            # print('x:',x)
            x = resnet(x, t)
            # print('after res1 x:',x)
            x = resnet2(x, t)
            # print('after res2 x:',x)
            x = downsample(x)
            # print('after down x:',x)

        x = self.mid_block1(x, t)
        x = self.mid_down1(x)

        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
