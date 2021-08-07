import os

import numpy as np
import torch.distributions
import torch.nn.functional as F

from ding.config.config import read_config_yaml
from ding.model import create_model
from ding.utils import POLICY_REGISTRY

cfg_path = os.path.join(os.getcwd(), 'alphazero_config.yaml')
default_az_cfg = read_config_yaml(cfg_path)


@POLICY_REGISTRY.register('alphazero')
class AlphaZeroPolicy:
    def __init__(self, cfg, model=None):
        self._cfg = cfg
        self._model = model if model else create_model(cfg.model)
        self._cuda = cfg.learner.cuda and torch.cuda.is_available()
        self._device = torch.cuda.current_device() if self._cuda else 'cpu'

        self._grad_norm = cfg.learner.get('grad_norm', 1)
        self._lr_multiplier = 1.0
        self._learning_rate = self.cfg.policy.learning_rate
        self._weight_decay = self._cfg.learner.get('weight_decay', 0)
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           weight_decay=self._weight_decay,
                                           lr=self._learning_rate)
        self._kl_targ = self._cfg.policy.get('kl_targ', 0.02)

    def compute_logit_value(self, state_batch):
        logits, values = self._model(state_batch)
        return logits, values

    def compute_prob_value(self, state_batch):
        logits, values = self._model(state_batch)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs()
        return probs, values

    def policy_value_fn(self, env):
        """
        input: env
        output: a list of (action, probability) tuples for each available
        action and the score of the env state
        """
        legal_positions = env.legal_actions
        current_state = env.current_state().reshape(-1, 4, self.env_width, self.env_height)
        current_state = torch.from_numpy(current_state).to(device=self._device, dtype=torch.float)
        act_probs, value = self.compute_logit_value(current_state)
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, inputs):
        state_batch = inputs['state']
        mcts_probs = inputs['mcts_prob']
        winner_batch = inputs['winner']
        lr = inputs['lr']

        state_batch = state_batch.to(device=self._device, dtype=torch.float)
        mcts_probs = mcts_probs.to(device=self._device, dtype=torch.float)
        winner_batch = winner_batch.to(device=self._device, dtype=torch.float)

        log_act_probs, values = self.compute_prob_value(state_batch)
        # ============
        # policy loss
        # ============
        policy_loss = - torch.mean(torch.sum(mcts_probs * log_act_probs))

        # ============
        # value loss
        # ============
        value_loss = F.mse_loss(values.view(-1), winner_batch)

        total_loss = value_loss + policy_loss
        self._optimizer.zero_grad()
        self.set_learning_rate(self._optimizer, lr)

        total_loss.backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(
        #     list(self._model.parameters()),
        #     max_norm=self._grad_norm,
        # )
        self._optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )

        return total_loss.item(), entropy.item()

    def update_policy(self, inputs, update_per_collect):
        state_batch = inputs['state']
        mcts_probs = inputs['mcts_prob']
        winner_batch = inputs['winner']

        old_logits, old_values = self._model.compute_logit_value(state_batch)
        old_log_probs = F.log_softmax(old_logits, dim=-1)
        old_probs = torch.exp(old_log_probs)
        for _ in range(update_per_collect):
            loss, entropy = self.train_step(state_batch, mcts_probs, winner_batch,
                                            self._learning_rate * self._lr_multiplier)
            new_logits, new_v = self.compute_logit_value(state_batch)
            new_log_probs = F.log_softmax(new_logits, dim=-1)
            kl = torch.mean(torch.sum(old_probs * (old_log_probs - new_log_probs), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self._kl_tart * 2 and self._lr_multiplier > 0.1:
            self._lr_multiplier /= 1.5
        elif kl < self._kl_tart / 2 and self._lr_multiplier < 10:
            self._lr_multiplier *= 1.5

        return loss.item(), entropy.item()

    @staticmethod
    def set_learning_rate(optimizer, lr):
        """Sets the learning rate to the given value"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    policy = AlphaZeroPolicy(default_az_cfg)
