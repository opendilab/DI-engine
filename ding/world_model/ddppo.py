import itertools
import numpy as np
import multiprocessing
import copy
import torch
from torch import nn

from scipy.spatial import KDTree
from functools import partial
from ding.utils import WORLD_MODEL_REGISTRY
from ding.utils.data import default_collate
from ding.torch_utils import unsqueeze_repeat
from ding.world_model.base_world_model import HybridWorldModel
from ding.world_model.model.ensemble import EnsembleModel, StandardScaler


#======================= Helper functions =======================
# tree_query = lambda datapoint: tree.query(datapoint, k=k+1)[1][1:]
def tree_query(datapoint, tree, k):
    return tree.query(datapoint, k=k + 1)[1][1:]


def get_neighbor_index(data, k, serial=False):
    """
        data: [B, N]
        k: int

        ret: [B, k]
    """
    data = data.cpu().numpy()
    tree = KDTree(data)

    if serial:
        nn_index = [torch.from_numpy(np.array(tree_query(d, tree, k))) for d in data]
        nn_index = torch.stack(nn_index).long()
    else:
        # TODO: speed up multiprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        fn = partial(tree_query, tree=tree, k=k)
        nn_index = torch.from_numpy(np.array(list(pool.map(fn, data)), dtype=np.int32)).to(torch.long)
        pool.close()
    return nn_index


def get_batch_jacobian(net, x, noutputs):  # x: b, in dim, noutpouts: out dim
    x = x.unsqueeze(1)  # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1)  # b, out_dim, in_dim
    x.requires_grad_(True)
    y = net(x)
    upstream_gradient = torch.eye(noutputs).reshape(1, noutputs, noutputs).repeat(n, 1, 1).to(x.device)
    re = torch.autograd.grad(y, x, upstream_gradient, create_graph=True)[0]

    return re


class EnsembleGradientModel(EnsembleModel):

    def train(self, loss, loss_reg, reg):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss += reg * loss_reg
        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()

        self.optimizer.step()


# TODO: derive from MBPO instead of implementing from scratch
@WORLD_MODEL_REGISTRY.register('ddppo')
class DDPPOWorldMode(HybridWorldModel, nn.Module):
    """rollout model + gradient model"""
    config = dict(
        model=dict(
            ensemble_size=7,
            elite_size=5,
            state_size=None,  # has to be specified
            action_size=None,  # has to be specified
            reward_size=1,
            hidden_size=200,
            use_decay=False,
            batch_size=256,
            holdout_ratio=0.2,
            max_epochs_since_update=5,
            deterministic_rollout=True,
            # parameters for DDPPO
            gradient_model=True,
            k=3,
            reg=1,
            neighbor_pool_size=10000,
            train_freq_gradient_model=250
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
        # parameters for DDPPO
        self.gradient_model = cfg.gradient_model
        self.k = cfg.k
        self.reg = cfg.reg
        self.neighbor_pool_size = cfg.neighbor_pool_size
        self.train_freq_gradient_model = cfg.train_freq_gradient_model

        self.rollout_model = EnsembleModel(
            self.state_size,
            self.action_size,
            self.reward_size,
            self.ensemble_size,
            self.hidden_size,
            use_decay=self.use_decay
        )
        self.scaler = StandardScaler(self.state_size + self.action_size)

        self.ensemble_mse_losses = []
        self.model_variances = []
        self.elite_model_idxes = []

        if self.gradient_model:
            self.gradient_model = EnsembleGradientModel(
                self.state_size,
                self.action_size,
                self.reward_size,
                self.ensemble_size,
                self.hidden_size,
                use_decay=self.use_decay
            )
        self.elite_model_idxes_gradient_model = []

        self.last_train_step_gradient_model = 0
        self.serial_calc_nn = False

        if self._cuda:
            self.cuda()

    def step(self, obs, act, batch_size=8192):

        class Predict(torch.autograd.Function):
            # TODO: align rollout_model elites with gradient_model elites
            # use different model for forward and backward
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                mean, var = self.rollout_model(x, ret_log_var=False)
                return torch.cat([mean, var], dim=-1)

            @staticmethod
            def backward(ctx, grad_out):
                x, = ctx.saved_tensors
                with torch.enable_grad():
                    x = x.detach()
                    x.requires_grad_(True)
                    mean, var = self.gradient_model(x, ret_log_var=False)
                    y = torch.cat([mean, var], dim=-1)
                    return torch.autograd.grad(y, x, grad_outputs=grad_out, create_graph=True)

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
            input = unsqueeze_repeat(inputs[i:i + batch_size], self.ensemble_size)
            if not torch.is_grad_enabled() or not self.gradient_model:
                b_mean, b_var = self.rollout_model(input, ret_log_var=False)
            else:
                # use gradient model to compute gradients during backward pass
                output = Predict.apply(input)
                b_mean, b_var = output.chunk(2, dim=2)
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
            ensemble_sample = ensemble_mean + torch.randn_like(ensemble_mean).to(ensemble_mean) * ensemble_std
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
            mean, logvar = self.rollout_model(inputs, ret_log_var=True)
            loss, mse_loss = self.rollout_model.loss(mean, logvar, labels)
            ensemble_mse_loss = torch.pow(mean.mean(0) - labels[0], 2)
            model_variance = mean.var(0)
            self.tb_logger.add_scalar('env_model_step/eval_mse_loss', mse_loss.mean().item(), envstep)
            self.tb_logger.add_scalar('env_model_step/eval_ensemble_mse_loss', ensemble_mse_loss.mean().item(), envstep)
            self.tb_logger.add_scalar('env_model_step/eval_model_variances', model_variance.mean().item(), envstep)

        self.last_eval_step = envstep

    def train(self, env_buffer, envstep, train_iter):

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

        data = env_buffer.sample(env_buffer.count(), train_iter)
        inputs, labels = train_sample(data)
        logvar.update(self._train_rollout_model(inputs, labels))

        if self.gradient_model:
            # update neighbor pool
            if (envstep - self.last_train_step_gradient_model) >= self.train_freq_gradient_model:
                n = min(env_buffer.count(), self.neighbor_pool_size)
                self.neighbor_pool = env_buffer.sample(n, train_iter, sample_range=slice(-n, None))
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
            self.middle_holdout_mse_loss = sorted_loss[self.ensemble_size // 2]
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
            train_input_reg: [ensemble_size, B, state_size+action_size]

            ret: [ensemble_size, B, state_size+reward_size, state_size+action_size]
        """

        def func(x):
            x = x.view(self.ensemble_size, -1, self.state_size + self.action_size)
            state = x[:, :, :self.state_size]
            x = self.scaler.transform(x)
            y, _ = model(x)
            # y[:, :, self.reward_size:] += state, inplace operation leads to error
            null = torch.zeros_like(y)
            null[:, :, self.reward_size:] += state
            y = y + null

            return y.view(-1, self.state_size + self.reward_size, self.state_size + self.reward_size)

        # reshape input
        train_input_reg = train_input_reg.view(-1, self.state_size + self.action_size)
        jacobian = get_batch_jacobian(func, train_input_reg, self.state_size + self.reward_size)

        # reshape jacobian
        return jacobian.view(
            self.ensemble_size, -1, self.state_size + self.reward_size, self.state_size + self.action_size
        )

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
        holdout_inputs = unsqueeze_repeat(holdout_inputs, self.ensemble_size)
        holdout_labels = unsqueeze_repeat(holdout_labels, self.ensemble_size)

        #no split and normalization on regulation data
        train_inputs_reg, train_labels_reg = inputs_reg, labels_reg

        neighbor_index = get_neighbor_index(train_inputs_reg, self.k, serial=self.serial_calc_nn)
        neighbor_inputs = train_inputs_reg[neighbor_index]  # [N, k, state_size+action_size]
        neighbor_labels = train_labels_reg[neighbor_index]  # [N, k, state_size+reward_size]
        neighbor_inputs_distance = (neighbor_inputs - train_inputs_reg.unsqueeze(1))  # [N, k, state_size+action_size]
        neighbor_labels_distance = (neighbor_labels - train_labels_reg.unsqueeze(1))  # [N, k, state_size+reward_size]

        self._epochs_since_update = 0
        self._snapshots = {i: (-1, 1e10) for i in range(self.ensemble_size)}
        self._save_states()
        for epoch in itertools.count():

            train_idx = torch.stack([torch.randperm(train_inputs.shape[0])
                                     for _ in range(self.ensemble_size)]).to(train_inputs.device)

            train_idx_reg = torch.stack([torch.randperm(train_inputs_reg.shape[0])
                                         for _ in range(self.ensemble_size)]).to(train_inputs_reg.device)

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
                    idx_reg = train_idx_reg[:, start_pos % train_inputs_reg.shape[0]:(start_pos + self.batch_size) %
                                            train_inputs_reg.shape[0]]
                else:
                    idx_reg = train_idx_reg[:, 0:(start_pos + self.batch_size) % train_inputs_reg.shape[0]]

                train_input_reg = train_inputs_reg[idx_reg]
                neighbor_input_distance = neighbor_inputs_distance[idx_reg
                                                                   ]  # [ensemble_size, B, k, state_size+action_size]
                neighbor_label_distance = neighbor_labels_distance[idx_reg
                                                                   ]  # [ensemble_size, B, k, state_size+reward_size]

                jacobian = self._get_jacobian(self.gradient_model, train_input_reg).unsqueeze(2).repeat_interleave(
                    self.k, dim=2
                )  # [ensemble_size, B, k(repeat), state_size+reward_size, state_size+action_size]

                directional_derivative = (jacobian @ neighbor_input_distance.unsqueeze(-1)).squeeze(
                    -1
                )  # [ensemble_size, B, k, state_size+reward_size]

                loss_reg = torch.pow((neighbor_label_distance - directional_derivative),
                                     2).sum(0).mean()  # sumed over network

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
            self.middle_holdout_mse_loss = sorted_loss[self.ensemble_size // 2]
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
        return self._epochs_since_update > self.max_epochs_since_update
