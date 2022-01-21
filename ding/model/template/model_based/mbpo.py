import math
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.utils import MODEL_REGISTRY
from ding.torch_utils import Swish
from ding.utils.data import default_collate


class StandardScaler(nn.Module):

    def __init__(self, input_size):
        super(StandardScaler, self).__init__()
        self.register_buffer('std', torch.ones(1, input_size))
        self.register_buffer('mu', torch.zeros(1, input_size))

    def fit(self, data):
        std, mu = torch.std_mean(data, dim=0, keepdim=True)
        std[std < 1e-12] = 1
        self.std.data.mul_(0.0).add_(std)
        self.mu.data.mul_(0.0).add_(mu)

    def transform(self, data):
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        return self.std * data + self.mu


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
        assert input.shape[0] == self.ensemble_size and len(input.shape) == 3
        return torch.bmm(input, self.weight) + self.bias  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


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

        self.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean, logvar = nn5_output.chunk(2, dim=2)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, mean, logvar, labels):
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

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()

        self.optimizer.step()


@MODEL_REGISTRY.register('mbpo')
class EnsembleDynamicsModel(nn.Module):

    def __init__(
        self,
        network_size,
        elite_size,
        state_size,
        action_size,
        reward_size=1,
        hidden_size=200,
        use_decay=False,
        batch_size=256,
        holdout_ratio=0.2,
        max_epochs_since_update=5,
        train_freq=250,
        eval_freq=20,
        cuda=True,
        tb_logger=None
    ):
        super(EnsembleDynamicsModel, self).__init__()
        self._cuda = cuda
        self.tb_logger = tb_logger

        self.network_size = network_size
        self.elite_size = elite_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        self.ensemble_model = EnsembleModel(
            state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay
        )
        self.scaler = StandardScaler(state_size + action_size)
        if self._cuda:
            self.cuda()

        self.last_train_step = 0
        self.last_eval_step = 0
        self.train_freq = train_freq
        self.eval_freq = eval_freq
        self.ensemble_mse_losses = []
        self.model_variances = []

        self._max_epochs_since_update = max_epochs_since_update
        self.batch_size = batch_size
        self.holdout_ratio = holdout_ratio
        self.elite_model_idxes = []

    def should_eval(self, envstep):
        """
        Overview:
            Determine whether you need to start the evaluation mode, if the number of training has reached\
                the maximum number of times to start the evaluator, return True
        """
        if (envstep - self.last_eval_step) < self.eval_freq or self.last_train_step == 0:
            return False
        return True

    def should_train(self, envstep):
        """
        Overview:
            Determine whether you need to start the evaluation mode, if the number of training has reached\
                the maximum number of times to start the evaluator, return True
        """
        if (envstep - self.last_train_step) < self.train_freq:
            return False
        return True

    def eval(self, data, envstep):
        # load data
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)

        # build eval samples
        inputs = torch.cat([obs, action], dim=1)
        labels = torch.cat([reward, next_obs - obs], dim=1)
        if self._cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # normalize
        inputs = self.scaler.transform(inputs)

        # repeat for ensemble
        inputs = inputs[None, :, :].repeat(self.network_size, 1, 1)
        labels = labels[None, :, :].repeat(self.network_size, 1, 1)

        # eval
        with torch.no_grad():
            mean, logvar = self.ensemble_model(inputs, ret_log_var=True)
            loss, mse_loss = self.ensemble_model.loss(mean, logvar, labels)
            ensemble_mse_loss = torch.pow(mean.mean(0) - labels[0], 2)
            model_variance = mean.var(0)
            self.tb_logger.add_scalar('env_model_step/eval_mse_loss', mse_loss.mean().item(), envstep)
            self.tb_logger.add_scalar('env_model_step/eval_ensemble_mse_loss', ensemble_mse_loss.mean().item(), envstep)
            self.tb_logger.add_scalar('env_model_step/eval_model_variances', model_variance.mean().item(), envstep)

        self.last_eval_step = envstep

    def train(self, buffer, train_iter, envstep):
        # load data
        data = buffer.sample(buffer.count(), train_iter)
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        # build train samples
        inputs = torch.cat([obs, action], dim=1)
        labels = torch.cat([reward, next_obs - obs], dim=1)
        if self._cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        # train
        logvar = self._train(inputs, labels)
        self.last_train_step = envstep
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                self.tb_logger.add_scalar('env_model_step/' + k, v, envstep)

    def _train(self, inputs, labels):
        #split
        num_holdout = int(inputs.shape[0] * self.holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        #normalize
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        #repeat for ensemble
        holdout_inputs = holdout_inputs[None, :, :].repeat(self.network_size, 1, 1)
        holdout_labels = holdout_labels[None, :, :].repeat(self.network_size, 1, 1)

        self._epochs_since_update = 0
        self._snapshots = {i: (-1, 1e10) for i in range(self.network_size)}
        self._save_states()
        for epoch in itertools.count():

            train_idx = torch.stack([torch.randperm(train_inputs.shape[0])
                                     for _ in range(self.network_size)]).to(train_inputs.device)
            self.mse_loss = []
            for start_pos in range(0, train_inputs.shape[0], self.batch_size):
                idx = train_idx[:, start_pos:start_pos + self.batch_size]
                train_input = train_inputs[idx]
                train_label = train_labels[idx]
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, mse_loss = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                self.mse_loss.append(mse_loss.mean().item())
            self.mse_loss = sum(self.mse_loss) / len(self.mse_loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_loss = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels)
                self.curr_holdout_mse_loss = holdout_mse_loss.mean().item()
                break_train = self._save_best(epoch, holdout_mse_loss)
                if break_train:
                    break

        self._load_states()
        with torch.no_grad():
            holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
            _, holdout_mse_loss = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels)
            sorted_loss, sorted_loss_idx = holdout_mse_loss.sort()
            sorted_loss = sorted_loss.detach().cpu().numpy().tolist()
            sorted_loss_idx = sorted_loss_idx.detach().cpu().numpy().tolist()
            self.elite_model_idxes = sorted_loss_idx[:self.elite_size]
            self.top_holdout_mse_loss = sorted_loss[0]
            self.middle_holdout_mse_loss = sorted_loss[self.network_size // 2]
            self.bottom_holdout_mse_loss = sorted_loss[-1]
            self.best_holdout_mse_loss = holdout_mse_loss.mean().item()
        return {
            'mse_loss': self.mse_loss,
            'curr_holdout_mse_loss': self.curr_holdout_mse_loss,
            'best_holdout_mse_loss': self.best_holdout_mse_loss,
            'top_holdout_mse_loss': self.top_holdout_mse_loss,
            'middle_holdout_mse_loss': self.middle_holdout_mse_loss,
            'bottom_holdout_mse_loss': self.bottom_holdout_mse_loss,
        }

    def _save_states(self, ):
        self._states = copy.deepcopy(self.state_dict())

    def _save_state(self, id):
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                self._states[k].data[id] = copy.deepcopy(v.data[id])

    def _load_states(self):
        self.load_state_dict(self._states)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                self._save_state(i)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def batch_predict(self, obs, action):
        # to predict a batch
        # norm and repeat for ensemble
        inputs = self.scaler.transform(torch.cat([obs, action], dim=-1)).unsqueeze(0).repeat(self.network_size, 1, 1)
        # predict
        outputs, _ = self.ensemble_model(inputs, ret_log_var=False)
        outputs = outputs.mean(0)
        return outputs[:, 0], outputs[:, 1:] + obs

    def predict(self, obs, act, batch_size=8192, deterministic=True):
        # to predict the whole buffer and return cpu tensor
        # form inputs
        if self._cuda:
            obs = obs.cuda()
            act = act.cuda()
        inputs = torch.cat([obs, act], dim=1)
        inputs = self.scaler.transform(inputs)
        # predict
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = inputs[i:i + batch_size].unsqueeze(0).repeat(self.network_size, 1, 1)
            b_mean, b_var = self.ensemble_model(input, ret_log_var=False)
            ensemble_mean.append(b_mean)
            ensemble_var.append(b_var)
        ensemble_mean = torch.cat(ensemble_mean, 1)
        ensemble_var = torch.cat(ensemble_var, 1)
        ensemble_mean[:, :, 1:] += obs.unsqueeze(0)
        ensemble_std = ensemble_var.sqrt()
        # sample from the predicted distribution
        if deterministic:
            ensemble_sample = ensemble_mean
        else:
            ensemble_sample = ensemble_mean + torch.randn(**ensemble_mean.shape).to(ensemble_mean) * ensemble_std
        # sample from ensemble
        model_idxes = torch.from_numpy(np.random.choice(self.elite_model_idxes, size=len(obs))).to(inputs.device)
        batch_idxes = torch.arange(len(obs)).to(inputs.device)
        sample = ensemble_sample[model_idxes, batch_idxes]
        rewards, next_obs = sample[:, :1], sample[:, 1:]

        return rewards.detach().cpu(), next_obs.detach().cpu()
