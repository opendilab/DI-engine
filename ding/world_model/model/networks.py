import math
import numpy as np
from typing import Optional, Dict, Union, List

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
from ding.utils import SequenceType
from ding.torch_utils.network.dreamer import weight_init, uniform_weight_init, static_scan, \
    OneHotDist, ContDist, SymlogDist, DreamerLayerNorm


class RSSM(nn.Module):

    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        action_type=None,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act=nn.ELU,
        norm=nn.LayerNorm,
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="gru",
        unimix_ratio=0.01,
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._action_type = action_type
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        self._act = act
        self._norm = norm
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(self._norm(self._hidden, eps=1e-03))
            inp_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(weight_init)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
            self._cell.apply(weight_init)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
            self._cell.apply(weight_init)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(self._norm(self._hidden, eps=1e-03))
            img_out_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(self._norm(self._hidden, eps=1e-03))
            obs_out_layers.append(self._act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(weight_init)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._ims_stat_layer.apply(weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(weight_init)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._ims_stat_layer.apply(weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(weight_init)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(self._device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        return state

    def observe(self, embed, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))  # 交换前两维
        if state is None:
            state = self.initial(action.shape[0])  # {logit, stoch, deter}
        # (batch, time, ch) -> (time, batch, ch)
        embed, action = swap(embed), swap(action)
        post, prior = static_scan(
            lambda prev_state, prev_act, embed: self.obs_step(prev_state[0], prev_act, embed),
            (action, embed),
            (state, state),
        )

        # (time, batch, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)
        else:
            mean, std = state["mean"], state["std"]
            dist = ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        return dist

    def obs_step(self, prev_state, prev_action, embed, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        if self._action_type == 'continuous':
            prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self._obs_out_layers(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    # this is used for making future image
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        if self._action_type == 'continuous':
            prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape)
            # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_stoch, prev_action, embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._inp_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}  # {stoch, deter, logit}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, forward, free, lscale, rscale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        # forward == false -> (post, prior)
        lhs, rhs = (prior, post) if forward else (post, prior)

        # forward == false -> Lrep
        value_lhs = value = kld(
            dist(lhs) if self._discrete else dist(lhs)._dist,
            dist(sg(rhs)) if self._discrete else dist(sg(rhs))._dist,
        )
        # forward == false -> Ldyn
        value_rhs = kld(
            dist(sg(lhs)) if self._discrete else dist(sg(lhs))._dist,
            dist(rhs) if self._discrete else dist(rhs)._dist,
        )
        # free bits
        loss_lhs = torch.mean(torch.clip(value_lhs, min=free))
        loss_rhs = torch.mean(torch.clip(value_rhs, min=free))
        loss = lscale * loss_lhs + rscale * loss_rhs

        return loss, value, loss_lhs, loss_rhs


class ConvDecoder(nn.Module):

    def __init__(
        self,
        inp_depth,  # config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        depth=32,
        act=nn.ELU,
        norm=nn.LayerNorm,
        shape=(3, 64, 64),
        kernels=(3, 3, 3, 3),
        outscale=1.0,
    ):
        super(ConvDecoder, self).__init__()
        self._inp_depth = inp_depth
        self._act = act
        self._norm = norm
        self._depth = depth
        self._shape = shape
        self._kernels = kernels
        self._embed_size = ((64 // 2 ** (len(kernels))) ** 2 * depth * 2 ** (len(kernels) - 1))

        self._linear_layer = nn.Linear(inp_depth, self._embed_size)
        inp_dim = self._embed_size // 16  # 除以最后的4*4 feature map来得到channel数

        layers = []
        h, w = 4, 4
        for i, kernel in enumerate(self._kernels):
            depth = self._embed_size // 16 // (2 ** (i + 1))
            act = self._act
            bias = False
            initializer = weight_init
            if i == len(self._kernels) - 1:
                depth = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = uniform_weight_init(outscale)

            if i != 0:
                inp_dim = 2 ** (len(self._kernels) - (i - 1) - 2) * self._depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    inp_dim,
                    depth,
                    kernel,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(DreamerLayerNorm(depth))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def __call__(self, features):
        x = self._linear_layer(features)  # feature:[batch, time, stoch*discrete + deter]
        x = x.reshape([-1, 4, 4, self._embed_size // 16])
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        mean = x.reshape(list(features.shape[:-1]) + self._shape)
        #mean = mean.permute(0, 1, 3, 4, 2)
        return SymlogDist(mean)


class GRUCell(nn.Module):

    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size  # hidden
        self._size = size  # deter
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
