import itertools
import copy
import multiprocessing

import numpy as np
import torch
from torch import nn
from scipy.spatial import KDTree

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



@MODEL_REGISTRY.register('ddppo')
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
        deterministic_rollout=False,
        cuda=True,
        tb_logger=None,

        # parameters for DDPPO
        use_gradient_model=True,
        k=3,
        reg=1,
        neighbor_pool_size=10000,
        train_freq_gradient_model=250
    ):
        super(EnsembleDoubleModel, self).__init__()
        self.deterministic_rollout = deterministic_rollout
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

        self.scaler = StandardScaler(state_size + action_size)

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

        # parameters for DDPPO
        self.use_gradient_model = use_gradient_model
        self.k = k
        self.reg = reg
        self.neighbor_pool_size = neighbor_pool_size
        self.train_freq_gradient_model = train_freq_gradient_model

        self.gradient_model = EnsembleGradientModel(
            state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay
        )
        self.elite_model_idxes_gradient_model = []
                
        self.last_train_step_gradient_model = 0

        if self._cuda:
            self.cuda()


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

        def train_sample(data) -> tuple:
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


        logvar = dict()

        data = buffer.sample(buffer.count(), train_iter)
        inputs, labels = train_sample(data)
        logvar.update(self._train_rollout_model(inputs, labels))

        if self.use_gradient_model:
            # update neighbor pool
            if (envstep - self.last_train_step_gradient_model) >= self.train_freq_gradient_model:
                n = min(buffer.count(), self.neighbor_pool_size)
                self.neighbor_pool = buffer.sample(n, train_iter, sample_range=slice(-n, None))
                inputs_reg, labels_reg = train_sample(self.neighbor_pool)
                logvar.update(self._train_gradient_model(inputs, labels, inputs_reg, labels_reg))
                self.last_train_step_gradient_model = envstep


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
            'rollout_model/mse_loss': self.mse_loss,
            'rollout_model/curr_holdout_mse_loss': self.curr_holdout_mse_loss,
            'rollout_model/best_holdout_mse_loss': self.best_holdout_mse_loss,
            'rollout_model/top_holdout_mse_loss': self.top_holdout_mse_loss,
            'rollout_model/middle_holdout_mse_loss': self.middle_holdout_mse_loss,
            'rollout_model/bottom_holdout_mse_loss': self.bottom_holdout_mse_loss,
        }

    
    def _get_jacobian(self, model, train_input_reg):
        """
            train_input_reg: [network_size, B, state_size+action_size]

            ret: [network_size, B, state_size+reward_size, state_size+action_size]
        """
        def func(x):
            x = x.view(self.network_size, -1, self.state_size+self.action_size)
            state = x[:, :, :self.state_size]
            x = self.scaler.transform(x)
            y, _ = model(x)
            # y[:, :, self.reward_size:] += state, inplace operation leads to error
            null = torch.zeros_like(y)
            null[:, :, self.reward_size:] += state
            y = y + null

            return y.view(-1, self.state_size+self.reward_size, self.state_size+self.reward_size)

        # reshape input
        train_input_reg = train_input_reg.view(-1, self.state_size+self.action_size)
        jacobian = get_batch_jacobian(func, train_input_reg, self.state_size+self.reward_size)

        # reshape jacobian
        return jacobian.view(self.network_size, -1, self.state_size+self.reward_size, self.state_size+self.action_size)

    
    def _train_gradient_model(self, inputs, labels, inputs_reg, labels_reg):
        #split
        num_holdout = int(inputs.shape[0] * self.holdout_ratio)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        #normalize
        # self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        #repeat for ensemble
        holdout_inputs = holdout_inputs[None, :, :].repeat(self.network_size, 1, 1)
        holdout_labels = holdout_labels[None, :, :].repeat(self.network_size, 1, 1)

        #no split and normalization on regulation data 
        train_inputs_reg, train_labels_reg = inputs_reg, labels_reg

        neighbor_index = get_neighbor_index(train_inputs_reg, self.k)
        neighbor_inputs = train_inputs_reg[neighbor_index]  # [N, k, state_size+action_size]
        neighbor_labels = train_labels_reg[neighbor_index]  # [N, k, state_size+reward_size]
        neighbor_inputs_distance = (neighbor_inputs - train_inputs_reg.unsqueeze(1))  # [N, k, state_size+action_size]
        neighbor_labels_distance = (neighbor_labels - train_labels_reg.unsqueeze(1))  # [N, k, state_size+reward_size]

        self._epochs_since_update = 0
        self._snapshots = {i: (-1, 1e10) for i in range(self.network_size)}
        self._save_states()
        for epoch in itertools.count():

            train_idx = torch.stack([torch.randperm(train_inputs.shape[0])
                                     for _ in range(self.network_size)]).to(train_inputs.device)

            train_idx_reg = torch.stack([torch.randperm(train_inputs_reg.shape[0])
                                     for _ in range(self.network_size)]).to(train_inputs_reg.device)
            
            self.mse_loss = []
            self.grad_loss = []
            for start_pos in range(0, train_inputs.shape[0], self.batch_size):
                idx = train_idx[:, start_pos:start_pos + self.batch_size]
                train_input = train_inputs[idx]
                train_label = train_labels[idx]
                mean, logvar = self.gradient_model(train_input, ret_log_var=True)
                loss, mse_loss = self.gradient_model.loss(mean, logvar, train_label)

                # regulation loss
                if start_pos % train_inputs_reg.shape[0] < (start_pos + self.batch_size) % train_inputs_reg.shape[0]:
                    idx_reg = train_idx_reg[:, start_pos % train_inputs_reg.shape[0]: (start_pos + self.batch_size) % train_inputs_reg.shape[0]]
                else:
                    idx_reg = train_idx_reg[:, 0: (start_pos + self.batch_size) % train_inputs_reg.shape[0]]

                train_input_reg = train_inputs_reg[idx_reg]
                neighbor_input_distance = neighbor_inputs_distance[idx_reg]  # [network_size, B, k, state_size+action_size]
                neighbor_label_distance = neighbor_labels_distance[idx_reg]  # [network_size, B, k, state_size+reward_size]

                jacobian = self._get_jacobian(self.gradient_model, train_input_reg).unsqueeze(2).repeat_interleave(self.k, dim=2)  # [network_size, B, k(repeat), state_size+reward_size, state_size+action_size]

                directional_derivative = (jacobian @ neighbor_input_distance.unsqueeze(-1)).squeeze(-1)  # [network_size, B, k, state_size+reward_size]

                loss_reg = torch.pow((neighbor_label_distance - directional_derivative), 2).sum(0).mean()  # sumed over network

                self.gradient_model.train(loss, loss_reg, self.reg)
                self.mse_loss.append(mse_loss.mean().item())
                self.grad_loss.append(loss_reg.item())

            self.mse_loss = sum(self.mse_loss) / len(self.mse_loss)
            self.grad_loss = sum(self.grad_loss) / len(self.grad_loss)

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
            self.elite_model_idxes_gradient_model = sorted_loss_idx[:self.elite_size]
            self.top_holdout_mse_loss = sorted_loss[0]
            self.middle_holdout_mse_loss = sorted_loss[self.network_size // 2]
            self.bottom_holdout_mse_loss = sorted_loss[-1]
            self.best_holdout_mse_loss = holdout_mse_loss.mean().item()
        return {
            'gradient_model/mse_loss': self.mse_loss,
            'gradient_model/grad_loss': self.grad_loss,
            'gradient_model/curr_holdout_mse_loss': self.curr_holdout_mse_loss,
            'gradient_model/best_holdout_mse_loss': self.best_holdout_mse_loss,
            'gradient_model/top_holdout_mse_loss': self.top_holdout_mse_loss,
            'gradient_model/middle_holdout_mse_loss': self.middle_holdout_mse_loss,
            'gradient_model/bottom_holdout_mse_loss': self.bottom_holdout_mse_loss,
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

    # def batch_predict(self, obs, action):
    #     # to predict a batch
    #     # norm and repeat for ensemble
    #     class Predict(torch.autograd.Function):
    #         # use different model for forward and backward
    #         @staticmethod
    #         def forward(ctx, x):
    #             ctx.save_for_backward(x)
    #             return self.rollout_model(x, ret_log_var=False)[0]

    #         @staticmethod
    #         def backward(ctx, grad_out):
    #             x, = ctx.saved_tensors
    #             with torch.enable_grad():
    #                 x = x.detach()
    #                 x.requires_grad_(True)
    #                 y = self.gradient_model(x, ret_log_var=False)[0]
    #                 return torch.autograd.grad(y, x, grad_outputs=grad_out, create_graph=True)

    #     if len(action.shape) == 1:
    #         action = action.unsqueeze(1)
    #     inputs = self.scaler.transform(torch.cat([obs, action], dim=-1)).unsqueeze(0).repeat(self.network_size, 1, 1)
    #     # predict
    #     if self.use_gradient_model:
    #         outputs = Predict.apply(inputs)
    #     else: 
    #         outputs, _ = self.rollout_model(inputs, ret_log_var=False)
    #     outputs = outputs.mean(0)
    #     return outputs[:, 0], outputs[:, 1:] + obs

    def batch_predict(self, obs, action):

        def forward(inputs, mode='rollout'):
            # model = self.rollout_model
            if mode == 'rollout':
                model = self.rollout_model
                elite_model_indxes = self.elite_model_idxes
            else:
                model = self.gradient_model
                elite_model_indxes = self.elite_model_idxes_gradient_model
            ensemble_mean, ensemble_var = model(inputs, ret_log_var=False)
            ensemble_std = ensemble_var.sqrt()
            if self.deterministic_rollout:
                ensemble_sample = ensemble_mean
            else:
                ensemble_sample = ensemble_mean + torch.randn(*ensemble_mean.shape).to(ensemble_mean) * ensemble_std
            model_idxes = torch.from_numpy(np.random.choice(elite_model_indxes, size=len(obs))).to(inputs.device)
            batch_idxes = torch.arange(len(obs)).to(inputs.device)
            sample = ensemble_sample[model_idxes, batch_idxes]
            return sample

        class Predict(torch.autograd.Function):
            # use different model for forward and backward
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return forward(x)

            @staticmethod
            def backward(ctx, grad_out):
                x, = ctx.saved_tensors
                with torch.enable_grad():
                    x = x.detach()
                    x.requires_grad_(True)
                    y = forward(x, mode='gradient')
                    return torch.autograd.grad(y, x, grad_outputs=grad_out, create_graph=True)

        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        inputs = self.scaler.transform(torch.cat([obs, action], dim=-1)).unsqueeze(0).repeat(self.network_size, 1, 1)
        if self.use_gradient_model:
            sample = Predict.apply(inputs)
        else: 
            sample = forward(inputs)
        rewards, next_obs = sample[:, 0], sample[:, 1:] + obs

        return rewards, next_obs


    def predict(self, obs, act, batch_size=8192):
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
        if self.deterministic_rollout:
            ensemble_sample = ensemble_mean
        else:
            ensemble_sample = ensemble_mean + torch.randn(*ensemble_mean.shape).to(ensemble_mean) * ensemble_std
        # sample from ensemble
        model_idxes = torch.from_numpy(np.random.choice(self.elite_model_idxes, size=len(obs))).to(inputs.device)
        batch_idxes = torch.arange(len(obs)).to(inputs.device)
        sample = ensemble_sample[model_idxes, batch_idxes]
        rewards, next_obs = sample[:, :1], sample[:, 1:]

        return rewards.detach().cpu(), next_obs.detach().cpu()


#======================= Helper functions =======================
def get_neighbor_index(data, k):
    """
        data: [B, N]
        k: int

        ret: [B, k]
    """
    data = data.cpu().numpy()
    tree = KDTree(data)
    global tree_query
    # tree_query = lambda datapoint: tree.query(datapoint, k=k+1)[1][1:]
    def tree_query(datapoint):
        return tree.query(datapoint, k=k+1)[1][1:]
    # TODO: multiprocessing
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    nn_index = torch.from_numpy(
        np.array(list(pool.map(tree_query, data)), dtype=np.int32)
    ).to(torch.long)
    pool.close()
    return nn_index


def get_batch_jacobian(net, x, noutputs): # x: b, in dim, noutpouts: out dim
    x = x.unsqueeze(1) # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
    x.requires_grad_(True)
    y = net(x)
    upstream_gradient = torch.eye(noutputs
        ).reshape(1, noutputs, noutputs).repeat(n, 1, 1).to(x.device)
    re = torch.autograd.grad(y, x, upstream_gradient, create_graph=True)[0]

    return re
