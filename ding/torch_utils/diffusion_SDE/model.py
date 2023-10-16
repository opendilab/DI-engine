import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from ding.torch_utils.diffusion_SDE import dpm_solver_pytorch
from ding.torch_utils.diffusion_SDE import schedule
from scipy.special import softmax


def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class SiLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class Residual_Block(nn.Module):

    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim), SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim), SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)


class TwinQ(nn.Module):

    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [state_dim + action_dim, 256, 256, 256, 1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))


class GuidanceQt(nn.Module):

    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [action_dim + 32 + state_dim, 256, 256, 256, 256, 1]
        self.qt = mlp(dims, activation=SiLU)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))

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

    def __init__(self, cfg, adim, sdim) -> None:
        super().__init__(adim, sdim)
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.cfg = cfg
        self.q0 = TwinQ(adim, sdim).to(self.cfg.device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(self.cfg.device)
        self.qt = GuidanceQt(adim, sdim).to(self.cfg.device)
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
        update_target(self.q0, self.q0_target, 0.005)

        return q_loss.detach().cpu().numpy()

    def update_qt(self, data):
        # input  many s <bz, S>  anction <bz, M, A>,
        s = data['s']
        a = data['a']
        fake_a = data['fake_a']
        energy = self.q0_target(fake_a, torch.stack([s] * fake_a.shape[1], axis=1)).detach().squeeze()

        self.all_mean = torch.mean(energy, dim=-1).detach().cpu().squeeze().numpy()
        self.all_std = torch.std(energy, dim=-1).detach().cpu().squeeze().numpy()

        if self.cfg.method == "mse":
            random_t = torch.rand(a.shape[0], device=s.device) * (1. - 1e-3) + 1e-3
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            # calculate sample based baselines
            # sample_based_baseline = torch.max(energy, dim=-1, keepdim=True)[0]  #<bz , 1>
            sample_based_baseline = 0.0
            self.debug_used = (self.q0_target(a, s).detach() * self.alpha -
                               sample_based_baseline * self.alpha).detach().cpu().squeeze().numpy()
            loss = torch.mean(
                (
                    self.qt(perturbed_a, random_t, s) - self.q0_target(a, s).detach() * self.alpha +
                    sample_based_baseline * self.alpha
                ) ** 2
            )
        elif self.cfg.method == "emse":
            random_t = torch.rand(a.shape[0], device=s.device) * (1. - 1e-3) + 1e-3
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            # calculate sample based baselines
            # sample_based_baseline = (torch.logsumexp(energy*self.alpha, dim=-1, keepdim=True)- np.log(energy.shape[1])) /self.alpha   #<bz , 1>
            sample_based_baseline = torch.max(energy, dim=-1, keepdim=True)[0]  #<bz , 1>
            self.debug_used = (self.q0_target(a, s).detach() * self.alpha -
                               sample_based_baseline * self.alpha).detach().cpu().squeeze().numpy()

            def unlinear_func(value, alpha, clip=False):
                if clip:
                    return torch.exp(torch.clamp(value * alpha, -100, 4.5))
                else:
                    return torch.exp(value * alpha)

            loss = torch.mean(
                (
                    unlinear_func(self.qt(perturbed_a, random_t, s), 1.0, clip=True) -
                    unlinear_func(self.q0_target(a, s).detach() - sample_based_baseline, self.alpha, clip=True)
                ) ** 2
            )
        elif self.cfg.method == "CEP":
            # CEP guidance method, as proposed in the paper
            logsoftmax = nn.LogSoftmax(dim=1)
            softmax = nn.Softmax(dim=1)

            x0_data_energy = energy * self.alpha
            # random_t = torch.rand((fake_a.shape[0], fake_a.shape[1]), device=s.device) * (1. - 1e-3) + 1e-3
            random_t = torch.rand((fake_a.shape[0], ), device=s.device) * (1. - 1e-3) + 1e-3
            random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
            z = torch.randn_like(fake_a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
            xt_model_energy = self.qt(perturbed_fake_a, random_t, torch.stack([s] * fake_a.shape[1], axis=1)).squeeze()
            p_label = softmax(x0_data_energy)
            self.debug_used = torch.flatten(p_label).detach().cpu().numpy()
            loss = -torch.mean(torch.sum(p_label * logsoftmax(xt_model_energy), axis=-1))  #  <bz,M>
        else:
            raise NotImplementedError

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()


class ScoreBase(nn.Module):

    def __init__(self, cfg, input_dim, output_dim, marginal_prob_std, embed_dim=32):
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.device = self.cfg.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(
            self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True
        )
        # self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.q = []
        self.q.append(QGPO_Critic(cfg.qgpo_critic, adim=output_dim, sdim=input_dim - output_dim))

    def forward_dmp_wrapper_fn(self, x, t):
        score = self(x, t)
        result = -(score + self.q[0].calculate_guidance(x, t, self.condition)) * self.marginal_prob_std(t)[1][..., None]
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

    def __init__(self, cfg, input_dim, output_dim, marginal_prob_std, embed_dim=32):
        super().__init__(cfg.score_base, input_dim, output_dim, marginal_prob_std, embed_dim)
        # The swish activation function
        self.cfg = cfg
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim - output_dim, 32), SiLU())
        self.sort_t = nn.Sequential(
            nn.Linear(64, 128),
            SiLU(),
            nn.Linear(128, 128),
        )
        self.down_block1 = Residual_Block(output_dim, 512)
        self.down_block2 = Residual_Block(512, 256)
        self.down_block3 = Residual_Block(256, 128)
        self.middle1 = Residual_Block(128, 128)
        self.up_block3 = Residual_Block(256, 256)
        self.up_block2 = Residual_Block(512, 512)
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
        return h / self.marginal_prob_std(t)[1][..., None]
