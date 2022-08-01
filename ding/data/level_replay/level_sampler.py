from typing import Optional, Union, Any, List
from easydict import EasyDict
from ding.utils import deep_merge_dicts, SequenceType
from collections import namedtuple
import numpy as np
import torch


class LevelSampler():
    """
    Overview:
        Policy class of Prioritized Level Replay algorithm.
        https://arxiv.org/pdf/2010.03934.pdf

        PLR is a method for improving generalization and sample-efficiency of \
            deep RL agents on procedurally-generated environments by adaptively updating \
            a sampling distribution over the training levels based on a score of the learning \
            potential of replaying each level.
    """
    config = dict(
        strategy='policy_entropy',
        replay_schedule='fixed',
        score_transform='rank',
        temperature=1.0,
        eps=0.05,
        rho=0.2,
        nu=0.5,
        alpha=1.0,
        staleness_coef=0,
        staleness_transform='power',
        staleness_temperature=1.0,
    )

    def __init__(
        self,
        seeds: Optional[List[int]],
        obs_space: Union[int, SequenceType],
        action_space: int,
        num_actors: int,
        cfg: EasyDict,
    ):
        self.cfg = EasyDict(deep_merge_dicts(self.config, cfg))
        self.cfg.update(cfg)
        self.obs_space = obs_space
        self.action_space = action_space
        self.strategy = self.cfg.strategy
        self.replay_schedule = self.cfg.replay_schedule
        self.score_transform = self.cfg.score_transform
        self.temperature = self.cfg.temperature
        # Eps means the level replay epsilon for eps-greedy sampling
        self.eps = self.cfg.eps
        # Rho means the minimum size of replay set relative to total number of levels before sampling replays
        self.rho = self.cfg.rho
        # Nu means the probability of sampling a new level instead of a replay level
        self.nu = self.cfg.nu
        # Alpha means the level score EWA smoothing factor
        self.alpha = self.cfg.alpha
        self.staleness_coef = self.cfg.staleness_coef
        self.staleness_transform = self.cfg.staleness_transform
        self.staleness_temperature = self.cfg.staleness_temperature

        # Track seeds and scores as in np arrays backed by shared memory
        self.seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

        self.unseen_seed_weights = np.ones(len(seeds))
        self.seed_scores = np.zeros(len(seeds))
        self.partial_seed_scores = np.zeros((num_actors, len(seeds)), dtype=np.float)
        self.partial_seed_steps = np.zeros((num_actors, len(seeds)), dtype=np.int64)
        self.seed_staleness = np.zeros(len(seeds))

        self.next_seed_index = 0  # Only used for sequential strategy

    def update_with_rollouts(self, train_data: dict, num_actors: int):
        total_steps = train_data['reward'].shape[0]
        if self.strategy == 'random':
            return

        if self.strategy == 'policy_entropy':
            score_function = self._entropy
        elif self.strategy == 'least_confidence':
            score_function = self._least_confidence
        elif self.strategy == 'min_margin':
            score_function = self._min_margin
        elif self.strategy == 'gae':
            score_function = self._gae
        elif self.strategy == 'value_l1':
            score_function = self._value_l1
        elif self.strategy == 'one_step_td_error':
            score_function = self._one_step_td_error
        else:
            raise ValueError('Not supported strategy: {}'.format(self.strategy))

        self._update_with_rollouts(train_data, num_actors, total_steps, score_function)

        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_score(actor_index, seed_idx, 0, 0)
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

    def update_seed_score(self, actor_index: int, seed_idx: int, score: float, num_steps: int):
        score = self._partial_update_seed_score(actor_index, seed_idx, score, num_steps, done=True)

        self.unseen_seed_weights[seed_idx] = 0.  # No longer unseen

        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha) * old_score + self.alpha * score

    def _partial_update_seed_score(
        self, actor_index: int, seed_idx: int, score: float, num_steps: int, done: bool = False
    ):
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps / float(running_num_steps)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0.  # zero partial score, partial num_steps
            self.partial_seed_steps[actor_index][seed_idx] = 0
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

        return merged_score

    def _entropy(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        num_actions = self.action_space
        max_entropy = -(1. / num_actions) * np.log(1. / num_actions) * num_actions

        return (-torch.exp(episode_logits) * episode_logits).sum(-1).mean().item() / max_entropy

    def _least_confidence(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        return (1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])).mean().item()

    def _min_margin(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        return 1 - (top2_confidence[:, 0] - top2_confidence[:, 1]).mean().item()

    def _gae(self, **kwargs):

        advantages = kwargs['adv']

        return advantages.mean().item()

    def _value_l1(self, **kwargs):
        advantages = kwargs['adv']
        # If the absolute value of ADV is large, it means that the level can significantly change
        # the policy and can be used to learn more

        return advantages.abs().mean().item()

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs['rewards']
        value = kwargs['value']

        max_t = len(rewards)
        td_errors = (rewards[:-1] + value[:max_t - 1] - value[1:max_t]).abs()

        return td_errors.abs().mean().item()

    def _update_with_rollouts(self, train_data: dict, num_actors: int, all_total_steps: int, score_function):
        level_seeds = train_data['seed'].reshape(num_actors, int(all_total_steps / num_actors)).transpose(0, 1)
        policy_logits = train_data['logit'].reshape(num_actors, int(all_total_steps / num_actors), -1).transpose(0, 1)
        done = train_data['done'].reshape(num_actors, int(all_total_steps / num_actors)).transpose(0, 1)
        total_steps, num_actors = policy_logits.shape[:2]
        num_decisions = len(policy_logits)

        for actor_index in range(num_actors):
            done_steps = done[:, actor_index].nonzero()[:total_steps, 0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps:
                    break

                if t == 0:  # if t is 0, then this done step caused a full update of previous seed last cycle
                    continue

                seed_t = level_seeds[start_t, actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:t, actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)

                if self.strategy in ['gae', 'value_l1', 'one_step_td_error']:
                    rewards = train_data['reward'].reshape(num_actors,
                                                           int(all_total_steps / num_actors)).transpose(0, 1)
                    adv = train_data['adv'].reshape(num_actors, int(all_total_steps / num_actors)).transpose(0, 1)
                    value = train_data['value'].reshape(num_actors, int(all_total_steps / num_actors)).transpose(0, 1)
                    score_function_kwargs['adv'] = adv[start_t:t, actor_index]
                    score_function_kwargs['rewards'] = rewards[start_t:t, actor_index]
                    score_function_kwargs['value'] = value[start_t:t, actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = len(episode_logits)
                self.update_seed_score(actor_index, seed_idx_t, score, num_steps)

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t, actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:, actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)

                if self.strategy in ['gae', 'value_l1', 'one_step_td_error']:
                    rewards = train_data['reward'].reshape(num_actors,
                                                           int(all_total_steps / num_actors)).transpose(0, 1)
                    adv = train_data['adv'].reshape(num_actors, int(all_total_steps / num_actors)).transpose(0, 1)
                    value = train_data['value'].reshape(num_actors, int(all_total_steps / num_actors)).transpose(0, 1)
                    score_function_kwargs['adv'] = adv[start_t:, actor_index]
                    score_function_kwargs['rewards'] = rewards[start_t:, actor_index]
                    score_function_kwargs['value'] = value[start_t:, actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = len(episode_logits)
                self._partial_update_seed_score(actor_index, seed_idx_t, score, num_steps)

    def _update_staleness(self, selected_idx: int):
        if self.staleness_coef > 0:
            self.seed_staleness += 1
            self.seed_staleness[selected_idx] = 0

    def _sample_replay_level(self):
        sample_weights = self._sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float) / len(sample_weights)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def _sample_unseen_level(self):
        sample_weights = self.unseen_seed_weights / self.unseen_seed_weights.sum()
        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def sample(self, strategy: Optional[str] = None):
        if not strategy:
            strategy = self.strategy

        if strategy == 'random':
            seed_idx = np.random.choice(range(len(self.seeds)))
            seed = self.seeds[seed_idx]
            return int(seed)

        elif strategy == 'sequential':
            seed_idx = self.next_seed_index
            self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
            seed = self.seeds[seed_idx]
            return int(seed)

        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen) / len(self.seeds)

        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho:
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else:  # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()

    def _sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1 - self.unseen_seed_weights)  # zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(
                self.staleness_transform, self.staleness_temperature, self.seed_staleness
            )
            staleness_weights = staleness_weights * (1 - self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0:
                staleness_weights /= z

            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights

        return weights

    def _score_transform(self, transform: Optional[str], temperature: float, scores: Optional[List[float]]):
        if transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1. / temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1. / temperature)

        return weights
