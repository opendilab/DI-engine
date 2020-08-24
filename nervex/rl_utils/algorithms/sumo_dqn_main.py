import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import yaml
from easydict import EasyDict
import threading

from typing import Optional

from nervex.worker.actor.env_manager.sumowj3_env_manager import SumoWJ3EnvManager
from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.data.structure.buffer import PrioritizedBufferWrapper
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent


def epsilon_greedy(start, end, decay):
    return lambda x: (start - end) * math.exp(-1 * x / decay) + end


def setup_config():
    with open(os.path.join(os.path.dirname(__file__), 'sumo_dqn_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


class SumoDqnRun():
    def __init__(self, cfg):
        self.cfg = cfg
        sumo_env = SumoWJ3Env(cfg.env)
        self.action_dim = [v for k, v in sumo_env.info().act_space.shape.items()]
        self.batch_size = self.cfg.train.batch_size
        self.env = SumoWJ3EnvManager(cfg.env)
        self.total_frame_num = cfg.train.dqn.total_frame_num
        self.max_epoch_frame = cfg.train.dqn.max_epoch_frame
        self.buffer = PrioritizedBufferWrapper(cfg.train.dqn.buffer_length)
        self.bandit = epsilon_greedy(0.95, 0.03, 10000)
        self.learner = SumoDqnLearner(self.cfg, self.buffer.iterable_sample(self.batch_size))
        self.actor_agent = SumoDqnActorAgent(self.learner.agent.model)
        self.actor_agent.load_state_dict(self.learner.agent.state_dict())

    def actor(self):
        total_frame_count = 0
        while True:
            obs = self.env.reset()
            dones = [False for _ in range(self.env.env_num)]
            while True:
                eps_threshold = self.bandit(self.learner.last_iter.val)
                actions, _ = self.actor_agent.forward(obs, eps=eps_threshold)
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


sumo_dqn_run = SumoDqnRun(setup_config())
sumo_dqn_run.run()
