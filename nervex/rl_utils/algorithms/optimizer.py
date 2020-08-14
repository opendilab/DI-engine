"""Library for RL optimization"""
import math
import numpy as np
import torch
import torch.Tensor as Tensor  # TODO(nyz) update specific tensor API


class BaseOptimizer:
    """optimizer

    args:
        cfg.rl.gamma: discount factor
        cfg.rl.lamda: GAE discount factor
        cfg.rl.advantage_norm: whether use advantage normalization
        cfg.rl.EPS: a small number for numerical stability
        cfg.rl.num_train_epoch: averaged training epoch
        cfg.rl.mini_batch_size: training batch size
        cfg.rl.ppo_clip: episolon in PPO2
        cfg.rl.loss_coeff_value: coefficient for value loss
        cfg.rl.loss_coeff_entropy: coefficient for entropy loss
    """
    def __init__(self, optim, actor, critic, cfg):
        self.cfg = cfg
        self.optim = optim
        self.actor = actor
        self.critic = critic
        self.old_critic = critic
        self.old_critic.eval()
        self.old_actor = actor
        self.old_actor.eval()

    def update_weight(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

    def update_ppo(self, a_batch_of_data):
        batch_size = a_batch_of_data.size(0)

        rewards = a_batch_of_data['rewards']
        masks = a_batch_of_data['masks']
        values = a_batch_of_data['values']
        states = a_batch_of_data['states']
        actions = a_batch_of_data['actions']
        oldlogproba = a_batch_of_data['oldlogproba']
        returns = Tensor(batch_size, 1)
        deltas = Tensor(batch_size, 1)
        advantages = Tensor(batch_size, 1)
        prev_return = 0.
        prev_value = 0.
        prev_advantage = 0.
        for i in reversed(range(batch_size)):
            '''GAE'''
            returns[i] = rewards[i] + self.cfg.rl.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.cfg.rl.gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + self.cfg.rl.gamma * self.cfg.rl.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        if self.cfg.rl.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.cfg.rl.EPS)

        losses, surr_losses, value_losses, ent_losses = [], [], [], []

        for i_epoch in range(int(self.cfg.rl.num_train_epoch * batch_size / self.cfg.rl.mini_batch_size)):
            minibatch_ind = np.random.choice(batch_size, self.cfg.rl.mini_batch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = self.actor.get_logproba(minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = self.critic(minibatch_states).flatten()

            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - self.cfg.rl.ppo_clip, 1 + self.cfg.rl.ppo_clip) * minibatch_advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))
            loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))
            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)
            total_loss = (
                loss_surr + self.cfg.rl.loss_coeff_value * loss_value + self.cfg.rl.loss_coeff_entropy * loss_entropy
            )

            losses.append(total_loss.detach().cpu().numpy())
            surr_losses.append(loss_surr.detach().cpu().numpy())
            value_losses.append(loss_value.detach().cpu().numpy())
            ent_losses.append(loss_entropy.detach().cpu().numpy())

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        self.sync_weight()
        return {
            'loss': losses,
            'loss_surr': surr_losses,
            'loss_value': value_losses,
            'loss_entropy': ent_losses,
        }

    def update_a2c(self, a_batch_of_data):
        batch_size = a_batch_of_data.size(0)

        rewards = a_batch_of_data['rewards']
        masks = a_batch_of_data['masks']
        values = a_batch_of_data['values']
        states = a_batch_of_data['states']
        actions = a_batch_of_data['actions']
        oldlogproba = a_batch_of_data['oldlogproba']

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        returns = Tensor(batch_size, 1)
        deltas = Tensor(batch_size, 1)
        advantages = Tensor(batch_size, 1)
        prev_return = 0.
        prev_value = 0.
        prev_advantage = 0.
        for i in reversed(range(batch_size)):
            '''GAE'''
            returns[i] = rewards[i] + self.cfg.rl.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.cfg.rl.gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + self.cfg.rl.gamma * self.cfg.rl.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        if self.cfg.rl.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.cfg.rl.EPS)

        action_mean, action_std_curr, value_curr = self.actor(Tensor(states))
        logproba_curr = self.actor.get_logproba(Tensor(states), Tensor(actions))

        loss_policy = torch.mean(-logproba_curr * advantages)
        loss_value = torch.mean((value_curr - returns).pow(2))
        loss_entropy = torch.mean(-(torch.log(2 * math.pi * action_std_curr.pow(2)) + 1) / 2)
        loss = loss_policy + self.cfg.rl.loss_coeff_value * loss_value + self.cfg.rl.loss_coeff_entropy * loss_entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            'loss': loss,
            'loss_policy': loss_policy,
            'loss_value': loss_value,
            'loss_entropy': loss_entropy,
        }

    def process_data(a_batch_of_data):
        raise NotImplementedError


class UPGOOptimizer(BaseOptimizer):
    def update(a_batch_of_data):
        raise NotImplementedError

    def process_data(a_batch_of_data):
        raise NotImplementedError


class VTraceOptimizer(BaseOptimizer):
    def update(a_batch_of_data):
        raise NotImplementedError

    def process_datas(a_batch_of_data):
        raise NotImplementedError
