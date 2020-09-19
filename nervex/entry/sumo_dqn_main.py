import numpy as np
import time
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import threading
from multiprocessing import Process
import multiprocessing

from typing import Optional

from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.worker import SubprocessEnvManager
from nervex.data.structure.buffer import PrioritizedBuffer
from nervex.data.collate_fn import sumo_dqn_collate_fn
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent
from nervex.torch_utils import to_device
from nervex.rl_utils import epsilon_greedy
from nervex.utils import read_config
from nervex.worker.learner import LearnerHook


class ActorProducerHook(LearnerHook):
    def __init__(self, runner, position, priority, freq=100):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            self._runner.actor()


class ActorUpdateHook(LearnerHook):
    def __init__(self, runner, position, priority, freq=100):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            self._runner.actor_agent.load_state_dict(engine.agent.state_dict())


def setup_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'sumo_dqn_default_config.yaml')
    return read_config(path)


class SumoDqnRun():
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_cuda = self.cfg.learner.use_cuda
        self.batch_size = self.cfg.learner.batch_size
        env_num = cfg.env.env_num
        self.env = SubprocessEnvManager(SumoWJ3Env, env_cfg=[cfg.env for _ in range(env_num)], env_num=env_num)
        self.buffer = PrioritizedBuffer(cfg.learner.data.buffer_length, max_reuse=cfg.learner.data.max_reuse)
        self.bandit = epsilon_greedy(0.95, 0.03, 10000)
        self.learner = SumoDqnLearner(self.cfg)
        self.actor_agent = SumoDqnActorAgent(self.learner.agent.model)
        self._setup_data_source()
        self.learner.register_hook(ActorUpdateHook(self, 'before_run', 40))
        self.learner.register_hook(ActorProducerHook(self, 'before_run', 100))
        self.learner.register_hook(ActorUpdateHook(self, 'after_iter', 40))
        self.learner.register_hook(ActorProducerHook(self, 'after_iter', 100))
        self.total_step_count = 0

    def _setup_data_source(self):
        def data_iterator():
            while True:
                while True:
                    data = self.buffer.sample(self.batch_size)
                    if data is not None:
                        break
                    time.sleep(5)
                yield sumo_dqn_collate_fn(data)

        self.learner._data_source = data_iterator()

    def actor(self):
        obs = self.env.reset()
        dones = [False for _ in range(self.env.env_num)]
        while True:
            eps_threshold = self.bandit(self.total_step_count)
            if self.use_cuda:
                obs = to_device(obs, 'cuda')
            actions, _ = self.actor_agent.forward(obs, eps=eps_threshold)
            if self.use_cuda:
                actions = to_device(actions, 'cpu')
            timestep = self.env.step(actions)
            for i, d in enumerate(dones):
                if not d:
                    step = {
                        'obs': obs[i],
                        'action': actions[i],
                        'next_obs': timestep.obs[i],
                        'reward': timestep.reward[i],
                        'done': timestep.done[i],
                    }
                    self.buffer.append(step)
                    obs[i] = timestep.obs[i]
            dones = timestep.done
            self.total_step_count += 1

            if all(dones):
                break
            if self.total_step_count % 50 == 0:
                print(
                    'actor run step {} with replay buffer size {}'.format(self.total_step_count, self.buffer.validlen)
                )

    def run(self):
        self.learner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    args = parser.parse_known_args()[0]
    sumo_dqn_run = SumoDqnRun(setup_config(args.config_path))
    sumo_dqn_run.run()
