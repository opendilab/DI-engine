import itertools
import numpy as np
import copy
import torch
from torch import nn

from ding.utils import WORLD_MODEL_REGISTRY
from ding.utils.data import default_collate
from ding.world_model.base_world_model import HybridWorldModel
from ding.world_model.model.ensemble import EnsembleModel, StandardScaler
from ding.torch_utils import fold_batch, unfold_batch, unsqueeze_repeat


@WORLD_MODEL_REGISTRY.register('mbpo')
class MBPOWorldModel(HybridWorldModel, nn.Module):
    config = dict(
        model=dict(
            ensemble_size=7,
            elite_size=5,
            state_size=None,
            action_size=None,
            reward_size=1,
            hidden_size=200,
            use_decay=False,
            batch_size=256,
            holdout_ratio=0.2,
            max_epochs_since_update=5,
            deterministic_rollout=True,
        ),
    )

    def __init__(self, cfg, env, tb_logger):
        HybridWorldModel.__init__(self, cfg, env, tb_logger)
        nn.Module.__init__(self)

        cfg = cfg.model
        self.ensemble_size = cfg.ensemble_size
        self.elite_size = cfg.elite_size
        self.state_size = cfg.state_size
        self.action_size = cfg.action_size
        self.reward_size = cfg.reward_size
        self.hidden_size = cfg.hidden_size
        self.use_decay = cfg.use_decay
        self.batch_size = cfg.batch_size
        self.holdout_ratio = cfg.holdout_ratio
        self.max_epochs_since_update = cfg.max_epochs_since_update
        self.deterministic_rollout = cfg.deterministic_rollout

        self.ensemble_model = EnsembleModel(
            self.state_size,
            self.action_size,
            self.reward_size,
            self.ensemble_size,
            self.hidden_size,
            use_decay=self.use_decay
        )
        self.scaler = StandardScaler(self.state_size + self.action_size)

        if self._cuda:
            self.cuda()

        self.ensemble_mse_losses = []
        self.model_variances = []
        self.elite_model_idxes = []

    def step(self, obs, act, batch_size=8192, keep_ensemble=False):
        if len(act.shape) == 1:
            act = act.unsqueeze(1)
        if self._cuda:
            obs = obs.cuda()
            act = act.cuda()
        inputs = torch.cat([obs, act], dim=-1)
        if keep_ensemble:
            inputs, dim = fold_batch(inputs, 1)
            inputs = self.scaler.transform(inputs)
            inputs = unfold_batch(inputs, dim)
        else:
            inputs = self.scaler.transform(inputs)
        # predict
        ensemble_mean, ensemble_var = [], []
        batch_dim = 0 if len(inputs.shape) == 2 else 1
        for i in range(0, inputs.shape[batch_dim], batch_size):
            if keep_ensemble:
                # inputs: [E, B, D]
                input = inputs[:, i:i + batch_size]
            else:
                # input:  [B, D]
                input = unsqueeze_repeat(inputs[i:i + batch_size], self.ensemble_size)
            b_mean, b_var = self.ensemble_model(input, ret_log_var=False)
            ensemble_mean.append(b_mean)
            ensemble_var.append(b_var)
        ensemble_mean = torch.cat(ensemble_mean, 1)
        ensemble_var = torch.cat(ensemble_var, 1)
        if keep_ensemble:
            ensemble_mean[:, :, 1:] += obs
        else:
            ensemble_mean[:, :, 1:] += obs.unsqueeze(0)
        ensemble_std = ensemble_var.sqrt()
        # sample from the predicted distribution
        if self.deterministic_rollout:
            ensemble_sample = ensemble_mean
        else:
            ensemble_sample = ensemble_mean + torch.randn_like(ensemble_mean).to(ensemble_mean) * ensemble_std
        if keep_ensemble:
            # [E, B, D]
            rewards, next_obs = ensemble_sample[:, :, 0], ensemble_sample[:, :, 1:]
            next_obs_flatten, dim = fold_batch(next_obs)
            done = unfold_batch(self.env.termination_fn(next_obs_flatten), dim)
            return rewards, next_obs, done
        # sample from ensemble
        model_idxes = torch.from_numpy(np.random.choice(self.elite_model_idxes, size=len(obs))).to(inputs.device)
        batch_idxes = torch.arange(len(obs)).to(inputs.device)
        sample = ensemble_sample[model_idxes, batch_idxes]
        rewards, next_obs = sample[:, 0], sample[:, 1:]

        return rewards, next_obs, self.env.termination_fn(next_obs)

    def eval(self, env_buffer, envstep, train_iter):
        data = env_buffer.sample(self.eval_freq, train_iter)
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)

        # build eval samples
        inputs = torch.cat([obs, action], dim=1)
        labels = torch.cat([reward, next_obs - obs], dim=1)
        if self._cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # normalize
        inputs = self.scaler.transform(inputs)

        # repeat for ensemble
        inputs = unsqueeze_repeat(inputs, self.ensemble_size)
        labels = unsqueeze_repeat(labels, self.ensemble_size)

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

    def train(self, env_buffer, envstep, train_iter):
        data = env_buffer.sample(env_buffer.count(), train_iter)
        data = default_collate(data)
        data['done'] = data['done'].float()
        data['weight'] = data.get('weight', None)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
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
        holdout_inputs = unsqueeze_repeat(holdout_inputs, self.ensemble_size)
        holdout_labels = unsqueeze_repeat(holdout_labels, self.ensemble_size)

        self._epochs_since_update = 0
        self._snapshots = {i: (-1, 1e10) for i in range(self.ensemble_size)}
        self._save_states()
        for epoch in itertools.count():

            train_idx = torch.stack([torch.randperm(train_inputs.shape[0])
                                     for _ in range(self.ensemble_size)]).to(train_inputs.device)
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
            self.middle_holdout_mse_loss = sorted_loss[self.ensemble_size // 2]
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
        return self._epochs_since_update > self.max_epochs_since_update
