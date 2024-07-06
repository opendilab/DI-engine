#############################################################
# This QGPO model is a modification implementation from https://github.com/ChenDRAG/CEP-energy-guided-diffusion
#############################################################

from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ding.torch_utils import MLP
from ding.torch_utils.diffusion_SDE.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
from ding.model.common.encoder import GaussianFourierProjectionTimeEncoder
from ding.torch_utils.network.res_block import TemporalSpatialResBlock


def marginal_prob_std(t, device):
    """
    Overview:
        Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    Arguments:
        - t (:obj:`torch.Tensor`): The input time.
        - device (:obj:`torch.device`): The device to use.
    """

    t = torch.tensor(t, device=device)
    beta_1 = 20.0
    beta_0 = 0.1
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std


class TwinQ(nn.Module):
    """
    Overview:
        Twin Q network for QGPO, which has two Q networks.
    Interfaces:
        ``__init__``, ``forward``, ``both``
    """

    def __init__(self, action_dim, state_dim):
        """
        Overview:
            Initialization of Twin Q.
        Arguments:
            - action_dim (:obj:`int`): The dimension of action.
            - state_dim (:obj:`int`): The dimension of state.
        """
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
        """
        Overview:
            Return the output of two Q networks.
        Arguments:
            - action (:obj:`torch.Tensor`): The input action.
            - condition (:obj:`torch.Tensor`): The input condition.
        """
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        """
        Overview:
            Return the minimum output of two Q networks.
        Arguments:
            - action (:obj:`torch.Tensor`): The input action.
            - condition (:obj:`torch.Tensor`): The input condition.
        """
        return torch.min(*self.both(action, condition))


class GuidanceQt(nn.Module):
    """
    Overview:
        Energy Guidance Qt network for QGPO. \
            In the origin paper, the energy guidance is trained by CEP method.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, action_dim, state_dim, time_embed_dim=32):
        """
        Overview:
            Initialization of Guidance Qt.
        Arguments:
            - action_dim (:obj:`int`): The dimension of action.
            - state_dim (:obj:`int`): The dimension of state.
            - time_embed_dim (:obj:`int`): The dimension of time embedding. \
                The time embedding is a Gaussian Fourier Feature tensor.
        """
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
        """
        Overview:
            Return the output of Guidance Qt.
        Arguments:
            - action (:obj:`torch.Tensor`): The input action.
            - t (:obj:`torch.Tensor`): The input time.
            - condition (:obj:`torch.Tensor`): The input condition.
        """
        embed = self.embed(t)
        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)


class QGPOCritic(nn.Module):
    """
    Overview:
        QGPO critic network.
    Interfaces:
        ``__init__``, ``forward``, ``calculateQ``, ``calculate_guidance``
    """

    def __init__(self, device, cfg, action_dim, state_dim) -> None:
        """
        Overview:
            Initialization of QGPO critic.
        Arguments:
            - device (:obj:`torch.device`): The device to use.
            - cfg (:obj:`EasyDict`): The config dict.
            - action_dim (:obj:`int`): The dimension of action.
            - state_dim (:obj:`int`): The dimension of state.
        """

        super().__init__()
        # is state_dim is 0  means unconditional guidance
        assert state_dim > 0
        # only apply to conditional sampling here
        self.device = device
        self.q0 = TwinQ(action_dim, state_dim).to(self.device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(self.device)
        self.qt = GuidanceQt(action_dim, state_dim).to(self.device)

        self.alpha = cfg.alpha
        self.q_alpha = cfg.q_alpha

    def calculate_guidance(self, a, t, condition=None, guidance_scale=1.0):
        """
        Overview:
            Calculate the guidance for conditional sampling.
        Arguments:
            - a (:obj:`torch.Tensor`): The input action.
            - t (:obj:`torch.Tensor`): The input time.
            - condition (:obj:`torch.Tensor`): The input condition.
            - guidance_scale (:obj:`float`): The scale of guidance.
        """

        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            guidance = guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def forward(self, a, condition=None):
        """
        Overview:
            Return the output of QGPO critic.
        Arguments:
            - a (:obj:`torch.Tensor`): The input action.
            - condition (:obj:`torch.Tensor`): The input condition.
        """

        return self.q0(a, condition)

    def calculateQ(self, a, condition=None):
        """
        Overview:
            Return the output of QGPO critic.
        Arguments:
            - a (:obj:`torch.Tensor`): The input action.
            - condition (:obj:`torch.Tensor`): The input condition.
        """

        return self(a, condition)


class ScoreNet(nn.Module):
    """
    Overview:
        Score-based generative model for QGPO.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, device, input_dim, output_dim, embed_dim=32):
        """
        Overview:
            Initialization of ScoreNet.
        Arguments:
            - device (:obj:`torch.device`): The device to use.
            - input_dim (:obj:`int`): The dimension of input.
            - output_dim (:obj:`int`): The dimension of output.
            - embed_dim (:obj:`int`): The dimension of time embedding. \
                The time embedding is a Gaussian Fourier Feature tensor.
        """

        super().__init__()

        # origin score base
        self.output_dim = output_dim
        self.embed = nn.Sequential(
            GaussianFourierProjectionTimeEncoder(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
        )

        self.device = device
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

    def forward(self, x, t, condition):
        """
        Overview:
            Return the output of ScoreNet.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
            - t (:obj:`torch.Tensor`): The input time.
            - condition (:obj:`torch.Tensor`): The input condition.
        """

        embed = self.embed(t)
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
        return h / marginal_prob_std(t, device=self.device)[1][..., None]


class QGPO(nn.Module):
    """
    Overview:
        Model of QGPO algorithm.
    Interfaces:
        ``__init__``, ``calculateQ``, ``select_actions``, ``sample``, ``score_model_loss_fn``, ``q_loss_fn``, \
            ``qt_loss_fn``
    """

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialization of QGPO.
        Arguments:
            - cfg (:obj:`EasyDict`): The config dict.
        """

        super(QGPO, self).__init__()
        self.device = cfg.device
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim

        self.noise_schedule = NoiseScheduleVP(schedule='linear')

        self.score_model = ScoreNet(
            device=self.device,
            input_dim=self.obs_dim + self.action_dim,
            output_dim=self.action_dim,
        )

        self.q = QGPOCritic(self.device, cfg.qgpo_critic, action_dim=self.action_dim, state_dim=self.obs_dim)

    def calculateQ(self, s, a):
        """
        Overview:
            Calculate the Q value.
        Arguments:
            - s (:obj:`torch.Tensor`): The input state.
            - a (:obj:`torch.Tensor`): The input action.
        """

        return self.q(a, s)

    def select_actions(self, states, diffusion_steps=15, guidance_scale=1.0):
        """
        Overview:
            Select actions for conditional sampling.
        Arguments:
            - states (:obj:`list`): The input states.
            - diffusion_steps (:obj:`int`): The diffusion steps.
            - guidance_scale (:obj:`float`): The scale of guidance.
        """

        def forward_dpm_wrapper_fn(x, t):
            score = self.score_model(x, t, condition=states)
            result = -(score +
                       self.q.calculate_guidance(x, t, states, guidance_scale=guidance_scale)) * marginal_prob_std(
                           t, device=self.device
                       )[1][..., None]
            return result

        self.eval()
        multiple_input = True
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            if states.dim == 1:
                states = states.unsqueeze(0)
                multiple_input = False
            num_states = states.shape[0]

            init_x = torch.randn(states.shape[0], self.action_dim, device=self.device)
            results = DPM_Solver(
                forward_dpm_wrapper_fn, self.noise_schedule, predict_x0=True
            ).sample(
                init_x, steps=diffusion_steps, order=2
            ).cpu().numpy()

            actions = results.reshape(num_states, self.action_dim).copy()  # <bz, A>

        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        self.train()
        return out_actions

    def sample(self, states, sample_per_state=16, diffusion_steps=15, guidance_scale=1.0):
        """
        Overview:
            Sample actions for conditional sampling.
        Arguments:
            - states (:obj:`list`): The input states.
            - sample_per_state (:obj:`int`): The number of samples per state.
            - diffusion_steps (:obj:`int`): The diffusion steps.
            - guidance_scale (:obj:`float`): The scale of guidance.
        """

        def forward_dpm_wrapper_fn(x, t):
            score = self.score_model(x, t, condition=states)
            result = -(score + self.q.calculate_guidance(x, t, states, guidance_scale=guidance_scale)) \
                * marginal_prob_std(t, device=self.device)[1][..., None]
            return result

        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)

            init_x = torch.randn(states.shape[0], self.action_dim, device=self.device)
            results = DPM_Solver(
                forward_dpm_wrapper_fn, self.noise_schedule, predict_x0=True
            ).sample(
                init_x, steps=diffusion_steps, order=2
            ).cpu().numpy()

            actions = results[:, :].reshape(num_states, sample_per_state, self.action_dim).copy()

        self.train()
        return actions

    def score_model_loss_fn(self, x, s, eps=1e-3):
        """
        Overview:
            The loss function for training score-based generative models.
        Arguments:
            model: A PyTorch model instance that represents a \
                time-dependent score-based model.
            x: A mini-batch of training data.
            eps: A tolerance value for numerical stability.
        """

        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)
        alpha_t, std = marginal_prob_std(random_t, device=x.device)
        perturbed_x = x * alpha_t[:, None] + z * std[:, None]
        score = self.score_model(perturbed_x, random_t, condition=s)
        loss = torch.mean(torch.sum((score * std[:, None] + z) ** 2, dim=(1, )))
        return loss

    def q_loss_fn(self, a, s, r, s_, d, fake_a_, discount=0.99):
        """
        Overview:
            The loss function for training Q function.
        Arguments:
            - a (:obj:`torch.Tensor`): The input action.
            - s (:obj:`torch.Tensor`): The input state.
            - r (:obj:`torch.Tensor`): The input reward.
            - s\_ (:obj:`torch.Tensor`): The input next state.
            - d (:obj:`torch.Tensor`): The input done.
            - fake_a (:obj:`torch.Tensor`): The input fake action.
            - discount (:obj:`float`): The discount factor.
        """

        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = self.q.q0_target(fake_a_, torch.stack([s_] * fake_a_.shape[1], axis=1)).detach().squeeze()
            next_v = torch.sum(softmax(self.q.q_alpha * next_energy) * next_energy, dim=-1, keepdim=True)
        # Update Q function
        targets = r + (1. - d.float()) * discount * next_v.detach()
        qs = self.q.q0.both(a, s)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        return q_loss

    def qt_loss_fn(self, s, fake_a):
        """
        Overview:
            The loss function for training Guidance Qt.
        Arguments:
            - s (:obj:`torch.Tensor`): The input state.
            - fake_a (:obj:`torch.Tensor`): The input fake action.
        """

        # input  many s <bz, S>  anction <bz, M, A>,
        energy = self.q.q0_target(fake_a, torch.stack([s] * fake_a.shape[1], axis=1)).detach().squeeze()

        # CEP guidance method, as proposed in the paper
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)

        x0_data_energy = energy * self.q.alpha
        random_t = torch.rand((fake_a.shape[0], ), device=self.device) * (1. - 1e-3) + 1e-3
        random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
        z = torch.randn_like(fake_a)
        alpha_t, std = marginal_prob_std(random_t, device=self.device)
        perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
        xt_model_energy = self.q.qt(perturbed_fake_a, random_t, torch.stack([s] * fake_a.shape[1], axis=1)).squeeze()
        p_label = softmax(x0_data_energy)

        #  <bz,M>
        qt_loss = -torch.mean(torch.sum(p_label * logsoftmax(xt_model_energy), axis=-1))
        return qt_loss
