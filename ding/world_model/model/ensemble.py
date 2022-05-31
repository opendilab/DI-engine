import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.torch_utils import Swish


class StandardScaler(nn.Module):

    def __init__(self, input_size: int):
        super(StandardScaler, self).__init__()
        self.register_buffer('std', torch.ones(1, input_size))
        self.register_buffer('mu', torch.zeros(1, input_size))

    def fit(self, data: torch.Tensor):
        std, mu = torch.std_mean(data, dim=0, keepdim=True)
        std[std < 1e-12] = 1
        self.std.data.mul_(0.0).add_(std)
        self.mu.data.mul_(0.0).add_(mu)

    def transform(self, data: torch.Tensor):
        return (data - self.mu) / self.std

    def inverse_transform(self, data: torch.Tensor):
        return self.std * data + self.mu


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0.) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.zeros(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[0] == self.ensemble_size and len(input.shape) == 3
        return torch.bmm(input, self.weight) + self.bias  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, ensemble_size={}, weight_decay={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.weight_decay
        )


class EnsembleModel(nn.Module):

    def __init__(
        self,
        state_size,
        action_size,
        reward_size,
        ensemble_size,
        hidden_size=200,
        learning_rate=1e-3,
        use_decay=False
    ):
        super(EnsembleModel, self).__init__()

        self.use_decay = use_decay
        self.hidden_size = hidden_size
        self.output_dim = state_size + reward_size

        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        self.max_logvar = nn.Parameter(torch.ones(1, self.output_dim).float() * 0.5, requires_grad=False)
        self.min_logvar = nn.Parameter(torch.ones(1, self.output_dim).float() * -10, requires_grad=False)
        self.swish = Swish()

        def init_weights(m: nn.Module):

            def truncated_normal_init(t, mean: float = 0.0, std: float = 0.01):
                torch.nn.init.normal_(t, mean=mean, std=std)
                while True:
                    cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
                    if not torch.sum(cond):
                        break
                    t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
                return t

            if isinstance(m, nn.Linear) or isinstance(m, EnsembleFC):
                input_dim = m.in_features
                truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
                m.bias.data.fill_(0.0)

        self.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor, ret_log_var: bool = False):
        x = self.swish(self.nn1(x))
        x = self.swish(self.nn2(x))
        x = self.swish(self.nn3(x))
        x = self.swish(self.nn4(x))
        x = self.nn5(x)

        mean, logvar = x.chunk(2, dim=2)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.modules():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, labels: torch.Tensor):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: Ensemble_size x N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss_inv = (torch.pow(mean - labels, 2) * inv_var).mean(dim=(1, 2))
        var_loss = logvar.mean(dim=(1, 2))
        with torch.no_grad():
            # Used only for logging.
            mse_loss = torch.pow(mean - labels, 2).mean(dim=(1, 2))
        total_loss = mse_loss_inv.sum() + var_loss.sum()
        return total_loss, mse_loss

    def train(self, loss: torch.Tensor):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()

        self.optimizer.step()
