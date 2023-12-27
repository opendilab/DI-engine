#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

from easydict import EasyDict
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ding.torch_utils import MLP
from ding.torch_utils.diffusion_SDE import dpm_solver_pytorch
from ding.model.common.encoder import GaussianFourierProjectionTimeEncoder
from ding.torch_utils.network.res_block import TemporalSpatialResBlock


def marginal_prob_std(t, device):
    """
    Overview:
        Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """
    t = torch.tensor(t, device=device)
    beta_1 = 20.0
    beta_0 = 0.1
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std


class TwinQ(nn.Module):

    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.q1 = MLP(
            in_channels=state_dim + action_dim,
            hidden_channels=256,
            out_channels=1,
            activation=nn.ReLU(),
            layer_num=4,
            output_activation=False
        )
        self.q2 = MLP(
            in_channels=state_dim + action_dim,
            hidden_channels=256,
            out_channels=1,
            activation=nn.ReLU(),
            layer_num=4,
            output_activation=False
        )

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))


class GuidanceQt(nn.Module):

    def __init__(self, action_dim, state_dim, time_embed_dim=32):
        super().__init__()
        self.qt = MLP(
            in_channels=action_dim + time_embed_dim + state_dim,
            hidden_channels=256,
            out_channels=1,
            activation=torch.nn.SiLU(),
            layer_num=4,
            output_activation=False
        )
        self.embed = nn.Sequential(
            GaussianFourierProjectionTimeEncoder(embed_dim=time_embed_dim), nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, action, t, condition=None):
        embed = self.embed(t)
        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)


class Critic_Guide(nn.Module):

    def __init__(self, adim, sdim) -> None:
        super().__init__()
        # is sdim is 0  means unconditional guidance
        self.conditional_sampling = False if sdim == 0 else True
        self.q0 = None
        self.qt = None

    def forward(self, a, condition=None):
        return self.q0(a, condition)

    def calculate_guidance(self, a, t, condition=None):
        raise NotImplementedError

    def calculateQ(self, a, condition=None):
        return self(a, condition)

    def update_q0(self, data):
        raise NotImplementedError

    def update_qt(self, data):
        raise NotImplementedError


class QGPO_Critic(Critic_Guide):

    def __init__(self, device, cfg, adim, sdim) -> None:
        super().__init__(adim, sdim)
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.device = device
        self.cfg = cfg
        self.q0 = TwinQ(adim, sdim).to(self.device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(self.device)
        self.qt = GuidanceQt(adim, sdim).to(self.device)
        self.qt_update_momentum = 0.005
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4)
        self.discount = 0.99

        self.alpha = self.cfg.alpha
        self.guidance_scale = 1.0

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            guidance = self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def update_q0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]

        fake_a = data['fake_a']
        fake_a_ = data['fake_a_']
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = self.q0_target(fake_a_, torch.stack([s_] * fake_a_.shape[1],
                                                              axis=1)).detach().squeeze()  # <bz, 16>
            next_v = torch.sum(softmax(self.cfg.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True)

        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target
        for param, target_param in zip(self.q0.parameters(), self.q0_target.parameters()):
            target_param.data.copy_(
                self.qt_update_momentum * param.data + (1 - self.qt_update_momentum) * target_param.data
            )

        return q_loss.detach().cpu().numpy()

    def update_qt(self, data):
        # input  many s <bz, S>  anction <bz, M, A>,
        s = data['s']
        a = data['a']
        fake_a = data['fake_a']
        energy = self.q0_target(fake_a, torch.stack([s] * fake_a.shape[1], axis=1)).detach().squeeze()

        self.all_mean = torch.mean(energy, dim=-1).detach().cpu().squeeze().numpy()
        self.all_std = torch.std(energy, dim=-1).detach().cpu().squeeze().numpy()

        # CEP guidance method, as proposed in the paper
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)

        x0_data_energy = energy * self.alpha
        # random_t = torch.rand((fake_a.shape[0], fake_a.shape[1]), device=s.device) * (1. - 1e-3) + 1e-3
        random_t = torch.rand((fake_a.shape[0], ), device=self.device) * (1. - 1e-3) + 1e-3
        random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
        z = torch.randn_like(fake_a)
        alpha_t, std = marginal_prob_std(random_t, device=self.device)
        perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
        xt_model_energy = self.qt(perturbed_fake_a, random_t, torch.stack([s] * fake_a.shape[1], axis=1)).squeeze()
        p_label = softmax(x0_data_energy)
        self.debug_used = torch.flatten(p_label).detach().cpu().numpy()
        #  <bz,M>
        loss = -torch.mean(torch.sum(p_label * logsoftmax(xt_model_energy), axis=-1))

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()


class ScoreBase(nn.Module):

    def __init__(self, device, cfg, input_dim, output_dim, marginal_prob_std, embed_dim=32):
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        self.embed = nn.Sequential(
            GaussianFourierProjectionTimeEncoder(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
        )
        self.device = device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(
            self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True
        )
        # self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.q = []
        self.q.append(QGPO_Critic(device, cfg.qgpo_critic, adim=output_dim, sdim=input_dim - output_dim))

    def forward_dmp_wrapper_fn(self, x, t):
        score = self(x, t)
        result = -(score + self.q[0].calculate_guidance(x, t, self.condition)) * self.marginal_prob_std(
            t, device=self.device
        )[1][..., None]
        return result

    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def calculateQ(self, s, a, t=None):
        if s is None:
            if self.condition.shape[0] == a.shape[0]:
                s = self.condition
            elif self.condition.shape[0] == 1:
                s = torch.cat([self.condition] * a.shape[0])
            else:
                assert False
        return self.q[0](a, s)

    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def select_actions(self, states, diffusion_steps=15):
        self.eval()
        multiple_input = True
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            if states.dim == 1:
                states = states.unsqueeze(0)
                multiple_input = False
            num_states = states.shape[0]
            self.condition = states
            results = self.dpm_wrapper_sample(
                self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2
            )
            actions = results.reshape(num_states, self.output_dim).copy()  # <bz, A>
            self.condition = None
        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        self.train()
        return out_actions

    def sample(self, states, sample_per_state=16, diffusion_steps=15):
        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(
                self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2
            )
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim).copy()
            self.condition = None
        self.train()
        return actions


class ScoreNet(ScoreBase):

    def __init__(self, device, cfg, input_dim, output_dim, marginal_prob_std, embed_dim=32):
        super().__init__(device, cfg, input_dim, output_dim, marginal_prob_std, embed_dim)
        # The swish activation function
        self.device = device
        self.cfg = cfg
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(nn.Linear(input_dim - output_dim, 32), torch.nn.SiLU())
        self.sort_t = nn.Sequential(
            nn.Linear(64, 128),
            torch.nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.down_block1 = TemporalSpatialResBlock(output_dim, 512)
        self.down_block2 = TemporalSpatialResBlock(512, 256)
        self.down_block3 = TemporalSpatialResBlock(256, 128)
        self.middle1 = TemporalSpatialResBlock(128, 128)
        self.up_block3 = TemporalSpatialResBlock(256, 256)
        self.up_block2 = TemporalSpatialResBlock(512, 512)
        self.last = nn.Linear(1024, output_dim)

    def forward(self, x, t, condition=None):
        embed = self.embed(t)

        if condition is not None:
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition] * x.shape[0])
            else:
                assert False
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        embed = self.sort_t(embed)
        d1 = self.down_block1(x, embed)
        d2 = self.down_block2(d1, embed)
        d3 = self.down_block3(d2, embed)
        u3 = self.middle1(d3, embed)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed)
        u0 = torch.cat([d1, u1], dim=-1)
        h = self.last(u0)
        self.h = h
        # Normalize output
        return h / self.marginal_prob_std(t, device=self.device)[1][..., None]


class QGPO(nn.Module):

    def __init__(self, cfg: EasyDict) -> None:
        super(QGPO, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim

        #marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device)

        self.score_model = ScoreNet(
            device=self.device,
            cfg=cfg.score_net,
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.action_dim,
            marginal_prob_std=marginal_prob_std,
        )

    def loss_fn(self, x, marginal_prob_std, eps=1e-3):
        """
        Overview:
            The loss function for training score-based generative models.
        Arguments:
            model: A PyTorch model instance that represents a \
                time-dependent score-based model.
            x: A mini-batch of training data.
            marginal_prob_std: A function that gives the standard deviation of \
                the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)
        alpha_t, std = marginal_prob_std(random_t, device=x.device)
        perturbed_x = x * alpha_t[:, None] + z * std[:, None]
        score = self.score_model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None] + z) ** 2, dim=(1, )))
        return loss
