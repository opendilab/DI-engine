import time
import argparse
import torch
import os
import sys

from typing import Optional

from nervex.data import PrioritizedBuffer, default_collate
from nervex.torch_utils import to_device
from nervex.rl_utils import epsilon_greedy
from nervex.utils import read_config
from nervex.worker.learner import LearnerHook


class ActorProducerHook(LearnerHook):
    def __init__(self, runner, position, priority, freq, episode_num):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq
        self._episode_num = episode_num

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            for _ in range(self._episode_num):
                self._runner.collect_data()


class ActorUpdateHook(LearnerHook):
    def __init__(self, runner, position, priority, freq):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            self._runner.actor_agent.load_state_dict(engine.agent.state_dict())


class EvaluateHook(LearnerHook):
    def __init__(self, runner, priority, freq):
        super().__init__(name='evaluate', position='after_iter', priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            self._runner.evaluate_agent.load_state_dict(engine.agent.state_dict())
            self._runner.evaluate()


class SingleMachineRunner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_cuda = self.cfg.learner.use_cuda
        self.batch_size = self.cfg.learner.batch_size
        self._setup_env()
        self.buffer = PrioritizedBuffer(cfg.learner.data.buffer_length, max_reuse=cfg.learner.data.max_reuse)
        self.bandit = epsilon_greedy(0.95, 0.05, 100000)

        self._setup_learner()
        self._setup_actor_agent()
        self._setup_evaluate_agent()
        self._setup_data_source()
        self.train_step = cfg.learner.train_step
        self.learner.register_hook(ActorUpdateHook(self, 'before_run', 40, self.train_step))
        self.learner.register_hook(
            ActorProducerHook(self, 'before_run', 100, self.train_step, self.cfg.actor.episode_num)
        )
        self.learner.register_hook(ActorUpdateHook(self, 'after_iter', 40, self.train_step))
        self.learner.register_hook(
            ActorProducerHook(self, 'after_iter', 100, self.train_step, self.cfg.actor.episode_num)
        )
        self.learner.register_hook(EvaluateHook(self, 100, cfg.actor.eval_step))
        self.actor_step_count = 0
        self.learner_step_count = 0

    def _setup_learner(self):
        raise NotImplementedError

    def _setup_actor_agent(self):
        raise NotImplementedError

    def _setup_evaluate_agent(self):
        raise NotImplementedError

    def _setup_env(self):
        raise NotImplementedError

    def _setup_data_source(self):
        def data_iterator():
            while True:
                while True:
                    data = self.buffer.sample(self.batch_size)
                    if data is not None:
                        break
                    time.sleep(5)
                yield default_collate(data)

        self.learner._data_source = data_iterator()

    def collect_data(self):
        obs = self.env.reset()
        obs = default_collate(obs)
        alive_env = [True for _ in range(self.env.env_num)]
        while True:
            eps_threshold = self.bandit(self.actor_step_count)
            if self.use_cuda:
                obs = to_device(obs, 'cuda')
            actions, _ = self.actor_agent.forward(obs, eps=eps_threshold)
            if self.use_cuda:
                actions = to_device(actions, 'cpu')
            timestep = self.env.step(actions)
            dones = timestep.done
            for i, d in enumerate(dones):
                if not alive_env[i]:
                    continue
                # only append not done data
                step = {
                    'obs': obs[i],
                    'action': actions[i],
                    'next_obs': timestep.obs[i],
                    'reward': timestep.reward[i],
                    'done': timestep.done[i],
                }
                self.buffer.append(step)
                obs[i] = timestep.obs[i]
                if d:
                    alive_env[i] = False
            self.actor_step_count += 1

            if all(dones):
                break
            if self.actor_step_count % 200 == 0:
                self.learner.info(
                    'actor run step {} with replay buffer size {} with eps {:.4f}'.format(
                        self.actor_step_count, self.buffer.validlen, eps_threshold
                    )
                )
        self.learner_step_count += self.train_step

    def evaluate(self):
        obs = self.env.reset()
        obs = torch.stack(obs, dim=0)
        cum_rewards = [0 for _ in range(self.env.env_num)]
        alive_env = [True for _ in range(self.env.env_num)]
        while True:
            if self.use_cuda:
                obs = to_device(obs, 'cuda')
            actions, _ = self.evaluate_agent.forward(obs)
            if self.use_cuda:
                actions = to_device(actions, 'cpu')
            timestep = self.env.step(actions)
            dones = timestep.done
            for i, d in enumerate(dones):
                if not alive_env[i]:
                    continue
                obs[i] = timestep.obs[i]
                if d:
                    alive_env[i] = False
                    cum_rewards[i] = self.learner.computation_graph.get_weighted_reward(timestep.info[i]['cum_reward']).item()

            if all(dones):
                avg_reward = sum(cum_rewards) / len(cum_rewards)
                self.learner.info('evaluate average reward: {:.3f}\t{}'.format(avg_reward, cum_rewards))
                if avg_reward >= self.cfg.env.stop_val:
                    sys.exit(0)
                break

    def run(self):
        self.learner.run()
