import itertools
import copy

import numpy as np
import torch
from torch import nn

from .mbpo import EnsembleModel, StandardScaler
from ding.utils import MODEL_REGISTRY
from ding.utils.data import default_collate


class EnsembleGradientModel(EnsembleModel):

    def train(self, loss, loss_reg, reg):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss += reg * loss_reg
        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()

        self.optimizer.step()



@MODEL_REGISTRY.register('DDPPO')
class EnsembleDoubleModel(nn.Module):
    """rollout model + gradient model"""

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
        tb_logger=None,

        # parameters for DDPPO
        n_near=3,
        reg=1,
    ):
        super(EnsembleDoubleModel, self).__init__()
        self._cuda = cuda
        self.tb_logger = tb_logger

        self.network_size = network_size
        self.elite_size = elite_size
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        self.rollout_model = EnsembleModel(
            state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay
        )
        self.gradient_model = EnsembleModel(
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

        self._n_near = n_near
        self._reg = reg


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
        inputs = inputs[None, :, :].repeat(self.network_size, 1, 1)
        labels = labels[None, :, :].repeat(self.network_size, 1, 1)

        # eval
        with torch.no_grad():
            mean, logvar = self.rollout_model(inputs, ret_log_var=True)
            loss, mse_loss = self.rollout_model.loss(mean, logvar, labels)
            ensemble_mse_loss = torch.pow(mean.mean(0) - labels[0], 2)
            model_variance = mean.var(0)
            self.tb_logger.add_scalar('env_model_step/eval_mse_loss', mse_loss.mean().item(), envstep)
            self.tb_logger.add_scalar('env_model_step/eval_ensemble_mse_loss', ensemble_mse_loss.mean().item(), envstep)
            self.tb_logger.add_scalar('env_model_step/eval_model_variances', model_variance.mean().item(), envstep)

        self.last_eval_step = envstep


    def train(self, buffer, train_iter, envstep):

        def train_sample(data) -> tuple['inputs', 'labels']:
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
            return inputs, labels

        data = buffer.sample(buffer.count(), train_iter)
        inputs, labels = train_sample(data)

        # Sample from the end of the buffer to get clustered data points for gradient loss regulation.
        # https://github.com/paperddppo/ddppo/blob/main/ddppo/model_regular_on_jacobian.py#L513
        data_reg = None # TODO (jrn)
        inputs_reg, labels_reg = train_sample(data_reg)

        # train
        logvar = dict()
        logvar.update(self._train_rollout_model(inputs, labels))
        logvar.update(self._train_gradient_model(inputs, labels, inputs_reg, labels_reg))
        self.last_train_step = envstep
        # log
        if self.tb_logger is not None:
            for k, v in logvar.items():
                self.tb_logger.add_scalar('env_model_step/' + k, v, envstep)


    def _train_rollout_model(self, inputs, labels):
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
                mean, logvar = self.rollout_model(train_input, ret_log_var=True)
                loss, mse_loss = self.rollout_model.loss(mean, logvar, train_label)
                self.rollout_model.train(loss)
                self.mse_loss.append(mse_loss.mean().item())
            self.mse_loss = sum(self.mse_loss) / len(self.mse_loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.rollout_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_loss = self.rollout_model.loss(holdout_mean, holdout_logvar, holdout_labels)
                self.curr_holdout_mse_loss = holdout_mse_loss.mean().item()
                break_train = self._save_best(epoch, holdout_mse_loss)
                if break_train:
                    break

        self._load_states()
        with torch.no_grad():
            holdout_mean, holdout_logvar = self.rollout_model(holdout_inputs, ret_log_var=True)
            _, holdout_mse_loss = self.rollout_model.loss(holdout_mean, holdout_logvar, holdout_labels)
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

    
    def _train_gradient_model(self, inputs, labels, inputs_reg, labels_reg):
        #split
        num_holdout = int(inputs.shape[0] * self.holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        #normalize
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        #no split and normalization on regulation data 
        train_inputs_reg, train_labels_reg = inputs_reg, labels_reg

        # TODO (jrn): KDTree 

        #repeat for ensemble
        holdout_inputs = holdout_inputs[None, :, :].repeat(self.network_size, 1, 1)
        holdout_labels = holdout_labels[None, :, :].repeat(self.network_size, 1, 1)

        self._epochs_since_update = 0
        self._snapshots = {i: (-1, 1e10) for i in range(self.network_size)}
        self._save_states()
        for epoch in itertools.count():

            train_idx = torch.stack([torch.randperm(train_inputs.shape[0])
                                     for _ in range(self.network_size)]).to(train_inputs.device)

            train_idx_reg = torch.stack([torch.randperm(train_inputs_reg.shape[0])
                                     for _ in range(self.network_size)]).to(train_inputs_reg.device)
            
            self.mse_loss = []
            for start_pos in range(0, train_inputs.shape[0], self.batch_size):
                idx = train_idx[:, start_pos:start_pos + self.batch_size]
                train_input = train_inputs[idx]
                train_label = train_labels[idx]
                mean, logvar = self.gradient_model(train_input, ret_log_var=True)
                loss, mse_loss = self.gradient_model.loss(mean, logvar, train_label)

                # TODO (jrn): regulation loss 
                loss_reg = None

                self.gradient_model.train(loss, loss_reg, self._reg)
                self.mse_loss.append(mse_loss.mean().item())
            self.mse_loss = sum(self.mse_loss) / len(self.mse_loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.gradient_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_loss = self.gradient_model.loss(holdout_mean, holdout_logvar, holdout_labels)
                self.curr_holdout_mse_loss = holdout_mse_loss.mean().item()
                break_train = self._save_best(epoch, holdout_mse_loss)
                if break_train:
                    break

        self._load_states()
        with torch.no_grad():
            holdout_mean, holdout_logvar = self.gradient_model(holdout_inputs, ret_log_var=True)
            _, holdout_mse_loss = self.gradient_model.loss(holdout_mean, holdout_logvar, holdout_labels)
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
            # TODO (jrn): log regulation loss
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
        class Predict(torch.autograd.Function):
            # use different model for forward and backward
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return self.rollout_model(x, ret_log_var=False)[0]

            @staticmethod
            def backward(ctx, grad_out):
                x, = ctx.saved_tensors
                with torch.enable_grad():
                    x = x.detach()
                    x.requires_grad_(True)
                    y = self.gradient_model(x, ret_log_var=False)[0]
                    return torch.autograd.grad(y, x, grad_outputs=grad_out, create_graph=True)

        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        inputs = self.scaler.transform(torch.cat([obs, action], dim=-1)).unsqueeze(0).repeat(self.network_size, 1, 1)
        # predict
        outputs = Predict.apply(inputs)
        outputs = outputs.mean(0)
        return outputs[:, 0], outputs[:, 1:] + obs


    def predict(self, obs, act, batch_size=8192, deterministic=True):
        # to predict the whole buffer and return cpu tensor
        # form inputs
        if len(act.shape) == 1:
            act = act.unsqueeze(1)
        if self._cuda:
            obs = obs.cuda()
            act = act.cuda()
        inputs = torch.cat([obs, act], dim=1)
        inputs = self.scaler.transform(inputs)
        # predict
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = inputs[i:i + batch_size].unsqueeze(0).repeat(self.network_size, 1, 1)
            b_mean, b_var = self.rollout_model(input, ret_log_var=False)
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
