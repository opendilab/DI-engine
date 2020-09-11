import numpy as np
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import threading

from typing import Optional

from nervex.worker.actor.env_manager.sumowj3_env_manager import SumoWJ3EnvManager
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.data.structure.buffer import PrioritizedBufferWrapper
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent
from nervex.data.collate_fn import sumo_dqn_collate_fn
from nervex.torch_utils import to_device
from nervex.utils import read_config
from nervex.rl_utils import epsilon_greedy


def setup_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'sumo_dqn_default_config.yaml')
    return read_config(path)


class SumoDqnRun():
    def __init__(self, cfg):
        self.cfg = cfg
        self.use_cuda = self.cfg.learner.use_cuda
        self.batch_size = self.cfg.learner.batch_size
        self.env = SumoWJ3EnvManager(cfg.env)
        self.total_frame_num = cfg.learner.dqn.total_frame_num
        self.max_epoch_frame = cfg.learner.dqn.max_epoch_frame
        self.buffer = PrioritizedBufferWrapper(cfg.learner.dqn.buffer_length)
        self.bandit = epsilon_greedy(0.95, 0.03, 10000)
        self.learner = SumoDqnLearner(self.cfg)
        self.actor_agent = SumoDqnActorAgent(self.learner.computation_graph.agent.model)
        self.actor_agent.load_state_dict(self.learner.computation_graph.agent.state_dict())
        self._setup_data_source()

    def _setup_data_source(self):
        def data_iterator():
            while True:
                yield sumo_dqn_collate_fn(next(self.buffer.iterable_sample(self.batch_size)))
        self.learner._data_source = data_iterator()

    def actor(self):
        total_frame_count = 0
        while True:
            obs = self.env.reset()
            dones = [False for _ in range(self.env.env_num)]
            while True:
                eps_threshold = self.bandit(self.learner.last_iter.val)
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
                total_frame_count += 1

                if all(dones):
                    break

    def run(self):
        threads = []
        threads.append(threading.Thread(target=self.learner.run))
        threads.append(threading.Thread(target=self.actor))
        for t in threads:
            t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    args = parser.parse_known_args()[0]
    sumo_dqn_run = SumoDqnRun(setup_config(args.config_path))
    sumo_dqn_run.run()
