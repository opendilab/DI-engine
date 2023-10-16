import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ding.torch_utils.diffusion_SDE import dpm_solver_pytorch
from ding.torch_utils.diffusion_SDE import schedule
from ding.torch_utils.diffusion_SDE.model import GaussianFourierProjection, Dense, SiLU, mlp


class GuidanceQt(nn.Module):

    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [action_dim + 32 + state_dim, 512, 512, 512, 512, 1]
        self.qt = mlp(dims, activation=SiLU)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))

    def forward(self, action, t, condition=None):
        embed = self.embed(t)
        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)


class Bandit_Critic_Guide(nn.Module):

    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.qt = GuidanceQt(adim, sdim).to(args.device)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4)

        self.args = args
        self.alpha = args.alpha
        self.guidance_scale = 1.0

    def forward(self, a, condition=None):
        return self.qt(a, condition)

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            guidance = self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def update_qt(self, data):
        a = data['a']
        energy = data['e']  # <bz, 1>

        if self.args.method == "mse":
            random_t = torch.rand(a.shape[0], device=a.device) * (1. - 1e-5) + 1e-5
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            loss = torch.mean((self.qt(perturbed_a, random_t, None) - energy * self.alpha) ** 2)
        elif self.args.method == "emse":
            random_t = torch.rand(a.shape[0], device=a.device) * (1. - 1e-5) + 1e-5
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            def unlinear_func(value, alpha, clip=False):
                if clip:
                    return torch.exp(torch.clamp(value * alpha, -100, 4.0))
                else:
                    return torch.exp(value * alpha)

            loss = torch.mean(
                (
                    unlinear_func(self.qt(perturbed_a, random_t, None), 1.0) -
                    unlinear_func(energy - 1.0, self.alpha, clip=True)
                ) ** 2
            )
        elif self.args.method == "CEP":
            logsoftmax = nn.LogSoftmax(dim=0)
            softmax = nn.Softmax(dim=0)

            x0_data_energy = energy * self.alpha
            random_t = torch.rand((1, ), device=a.device) * (1. - 1e-5) + 1e-5
            random_t = torch.cat([random_t] * a.shape[0])
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]
            xt_model_energy = self.qt(perturbed_a, random_t, None)

            loss = -torch.sum(logsoftmax(xt_model_energy) * softmax(x0_data_energy))  #  <bz,1>
        else:
            raise NotImplementedError

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()


class Bandit_ScoreBase(nn.Module):

    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        assert input_dim == output_dim
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.device = args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.q = []
        self.q.append(Bandit_Critic_Guide(adim=output_dim, sdim=input_dim - output_dim, args=args))
        self.args = args

    def forward_dmp_wrapper_fn(self, x, t):
        score = self(x, t)
        result = -(score + self.q[0].calculate_guidance(x, t, None)) * self.marginal_prob_std(t)[1][..., None]
        return result

    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def sample(self, states=None, sample_per_state=16, diffusion_steps=15):
        self.eval()
        with torch.no_grad():
            results = self.dpm_wrapper_sample(
                self.output_dim, batch_size=sample_per_state, steps=diffusion_steps, order=2, method='multistep'
            )
            actions = results[:, :]
        self.train()
        return actions


class Bandit_MlpScoreNet(Bandit_ScoreBase):

    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.dense1 = Dense(embed_dim, 32)
        self.dense2 = Dense(output_dim, 256 - 32)
        self.block1 = nn.Sequential(
            nn.Linear(256, 512),
            SiLU(),
            nn.Linear(512, 512),
            SiLU(),
            nn.Linear(512, 512),
            SiLU(),
            nn.Linear(512, 512),
            SiLU(),
            nn.Linear(512, 256),
        )
        self.decoder = Dense(256, output_dim)

    def forward(self, x, t, condition=None):
        x = x
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h = torch.cat((self.dense2(x), self.dense1(embed)), dim=-1)

        h = self.block1(h)
        h = self.decoder(self.act(h))
        # Normalize output
        h = h / self.marginal_prob_std(t)[1][..., None]
        return h
