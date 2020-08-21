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

from nervex.envs.sumo.vec_sumo_env import SumoEnvManager
from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.data.structure.buffer import PrioritizedBufferWrapper
from nervex.worker.agent.sumo_dqn_agent import SumoDqnAgent


def epsilon_greedy(start, end, decay):
    return lambda x: (start - end) * math.exp(-1 * x / decay) + end


def setup_config():
    with open(os.path.join(os.path.dirname(__file__), 'sumo_dqn_default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg


class SumoDqnRun():
    def __init__(self):
        sumo_env = SumoWJ3Env({})
        self.action_dim = [v for k, v in sumo_env.info().act_space.shape.items()]
        self.cfg = setup_config()
        self.batch_size = self.cfg.train.batch_size
        self.env = SumoEnvManager(EasyDict({'env_num': 4}))
        self.total_frame_num = 1000000
        self.max_epoch_frame = 2000
        self.buffer = PrioritizedBufferWrapper(10000)
        self.bandit = epsilon_greedy(0.95, 0.03, 10000)
        self.learner = SumoDqnLearner(self.cfg, self.buffer.iterable_sample(self.batch_size))
        self.agent = self.learner.agent

    def select_sumo_actions(self, states, curstep):
        actions = []
        for state in states:
            sample = random.random()
            if curstep is not None:
                eps_threshold = self.bandit(curstep)
            else:
                eps_threshold = 0.3
            if state is None:
                actions.append([torch.tensor([random.randint(0, dim - 1)]) for dim in self.action_dim])
                continue
            if sample > eps_threshold:
                with torch.no_grad():
                    action = []
                    for q in self.agent.model.forward(state):
                        action.append(q.argmax(dim=0))
                    actions.append(action)
            else:
                actions.append([torch.tensor([random.randint(0, dim - 1)]) for dim in self.action_dim])
        return actions

    def train(self):
        epoch_num = 0
        duration = 0
        for i_frame in range(self.total_frame_num):
            duration += 1
            states = self.env.reset()
            # TODO FIX env.reset()
            # print("states after reset = ", states)
            next_states = states
            cur_epoch_frame = 0
            dones = [False] * len(states)
            while True:
                # actions = self.select_actions(states, i_frame)
                actions = self.select_sumo_actions(states, i_frame)
                rets = self.env.step(actions)
                for i in range(len(rets)):
                    next_states[i], reward, dones[i], _ = rets[i]
                    step = {}
                    # step['"obs", "acts", "nextobs", "rewards", "termianls"']
                    step['obs'] = states[i]
                    step['acts'] = actions[i]
                    step['next_obs'] = next_states[i]
                    step['rewards'] = reward
                    if dones[i]:
                        isdone = torch.ones(1)
                    else:
                        isdone = torch.zeros(1)
                    step['terminals'] = isdone

                    self.buffer.append(step)

                    states[i] = next_states[i]

                if all(dones) or (i_frame - cur_epoch_frame) % self.max_epoch_frame:
                    cur_epoch_frame = i_frame
                    epoch_num += 1
                    # death = 0
                    duration = 0
                    states = self.env.reset()
                    break

    def run(self):
        threads = []
        threads.append(threading.Thread(target=self.learner.run))
        threads.append(threading.Thread(target=self.train))
        for t in threads:
            print(t)
            t.start()


sumo_dqn_run = SumoDqnRun()
sumo_dqn_run.run()
