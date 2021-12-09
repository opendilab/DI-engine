import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):

    def truncated_normal_init(t, mean=0.0, std=0.01):
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
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, 1, out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input, self.weight) + self.bias  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class EnsembleTransition(nn.Module):

    def __init__(
        self,
        state_size: int,
        action_size: int,
        ensemble_size: int,
        hidden_size=512,
        learning_rate=3e-4,
        use_decay=False
    ):
        super(EnsembleTransition, self).__init__()

        self.use_decay = use_decay
        self.hidden_size = hidden_size
        self.output_dim = state_size

        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)  # weight_decay helps network generalization.
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn5 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn6 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn7 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn8 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        self.max_logvar = nn.Parameter(torch.ones(1, self.output_dim).float() * 0.5, requires_grad=False)
        self.min_logvar = nn.Parameter(torch.ones(1, self.output_dim).float() * -10, requires_grad=False)
        self.relu = nn.ReLU()

        self.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor, ret_log_var=False) -> tuple:
        nn1_output = self.relu(self.nn1(x))
        nn2_output = self.relu(self.nn2(nn1_output))
        nn3_output = self.relu(self.nn3(nn2_output))
        nn4_output = self.relu(self.nn4(nn3_output))
        nn5_output = self.relu(self.nn5(nn4_output))
        nn6_output = self.relu(self.nn6(nn5_output))
        nn7_output = self.relu(self.nn7(nn6_output))
        nn8_output = self.nn8(nn7_output)

        mean, logvar = nn8_output.chunk(2, dim=2)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        mean, logvar: Ensemble_size x N x dim
        labels: Ensemble_size x N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss_inv = (torch.pow(mean - labels, 2) * inv_var).mean(dim=(1, 2))
        var_loss = logvar.mean(dim=(1, 2))
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


class EnsembleReward(nn.Module):

    def __init__(
        self,
        state_size: int,
        action_size: int,
        reward_size: int,
        ensemble_size: int,
        hidden_size=128,
        learning_rate=3e-4,
        use_decay=False
    ):
        super(EnsembleReward, self).__init__()

        self.use_decay = use_decay
        self.hidden_size = hidden_size
        self.output_dim = reward_size

        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, self.output_dim, ensemble_size, weight_decay=0.0001)
        self.relu = nn.ReLU()

        self.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nn1_output = self.relu(self.nn1(x))
        nn2_output = self.relu(self.nn2(nn1_output))
        nn3_output = self.relu(self.nn3(nn2_output))
        return self.nn4(nn3_output)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, reward: torch.Tensor, labels: torch.Tensor):
        """
        reward: Ensemble_size x N x dim
        labels: Ensemble_size x N x dim
        """
        mse_loss = torch.pow(reward - labels, 2).mean(dim=(1, 2))
        return mse_loss

    def train(self, loss: torch.Tensor):
        self.optimizer.zero_grad()

        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()

        self.optimizer.step()


CE_Loss = nn.CrossEntropyLoss()


class EnsembleDone(nn.Module):

    def __init__(
        self,
        state_size: int,
        action_size: int,
        done_size: int,
        ensemble_size: int,
        hidden_size=512,
        learning_rate=3e-4,
        use_decay=False
    ):
        super(EnsembleDone, self).__init__()

        self.use_decay = use_decay
        self.hidden_size = hidden_size
        self.output_dim = done_size

        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, self.output_dim, ensemble_size, weight_decay=0.0001)
        self.relu = nn.ReLU()

        self.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nn1_output = self.relu(self.nn1(x))
        nn2_output = self.relu(self.nn2(nn1_output))
        nn3_output = self.relu(self.nn3(nn2_output))
        nn4_output = self.nn4(nn3_output)
        return nn4_output

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, done, labels) -> torch.Tensor:
        """
        done: Ensemble_size x N x dim
        labels: Ensemble_size x N x dim
        """
        cross_entropy_loss = CE_Loss(done, labels)
        return cross_entropy_loss

    def train(self, loss: torch.Tensor):
        self.optimizer.zero_grad()

        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()
        self.optimizer.step()
