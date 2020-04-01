"""
Test script for actor worker on SLURM
"""
import random
import time
import os

import yaml
import torch
from absl import app
from absl import flags
from easydict import EasyDict

from sc2learner.data.tests.fake_dataset import FakeReplayDataset
from sc2learner.worker.actor.alphastar_actor_worker import AlphaStarActorWorker

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', '', 'Path to the config yaml file for test')
flags.DEFINE_bool('fake_dataset', True, 'Whether to use fake dataset')


class FakeEnv:
    def __init__(self, num_agents, *args, **kwargs):
        self.dataset = FakeReplayDataset(dict(trajectory_len=1))
        self.num_agents = num_agents

    def _get_obs(self):
        return [random.choice(self.dataset)[0] for _ in range(self.num_agents)]

    def reset(self):
        return self._get_obs()

    def step(self, *args, **kwargs):
        step = 16
        due = [True] * self.num_agents
        obs = self._get_obs()
        reward = 0.0
        done = False
        info = {}
        return step, due, obs, reward, done, info


class TestActor(AlphaStarActorWorker):
    def __init__(self, cfg):
        super(TestActor, self).__init__(cfg)

    def _make_env(self, players):
        if FLAGS.fake_dataset:
            return FakeEnv(len(players))
        else:
            return super()._make_env(players)

    def _module_init(self):
        super()._module_init()
        self.last_time = None

    def action_modifier(self, act, step):
        if self.cfg.env.use_cuda:
            print('Max CUDA memory:{}'.format(torch.cuda.max_memory_allocated()))
        t = time.time()
        if self.last_time is not None:
            print('Time between action:{}'.format(t - self.last_time))
        self.last_time = t
        for n in range(len(act)):
            if act[n] and act[n]['delay'] == 0:
                act[n]['delay'] = random.randint(0, 10)
            print('Act {}:{}'.format(n, str(act[n])))
        return act


def main(unused_argv):
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    cfg["log_path"] = os.path.dirname(FLAGS.config_path)
    ta = TestActor(cfg)
    ta.run()


if __name__ == '__main__':
    app.run(main)
