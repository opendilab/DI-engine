"""
Test script for actor
Usage:
      srun -p cpu -w SH-IDC1-10-198-6-160 \
      python3 -u -m sc2learner.worker.actor.alphastar_actor_test\
      --config_path PATH_TO_TEST_YAML
"""
import random

import yaml
from absl import app
from absl import flags
from easydict import EasyDict

from pysc2.env.sc2_env import Difficulty
from sc2learner.worker.actor.alphastar_actor import AlphaStarActor

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', '', 'Path to the config yaml file for test')


class TestActor(AlphaStarActor):
    def _module_init(self):
        self.job_getter = DummyJobGetter(self.cfg)
        self.model_requester = DummyModelRequester(self.cfg)
        self.data_pusher = DummyDataPusher(self.cfg)

    def action_modifier(self, act, agent):
        if act['delay'] == 0:
            act['delay'] = random.randint(0, 10)
        print('Act {}:{}'.format(agent, str(act)))
        return act


class DummyJobGetter:
    def __init__(self, cfg):
        self.connection = None
        self.job_request_id = 0
        pass

    def get_job(self, actor_id):
        print('received job req from:{}'.format(actor_id))
        # job1 = {
        #     'game_type': 'self_play',
        #     'model_id': 'test',
        #     'map_name': 'AbyssalReef',
        #     'random_seed': 10,
        #     'home_race': 'zerg',
        #     'away_race': 'zerg',
        #     'data_push_length': 64,
        # }
        job2 = {
            'game_type': 'game_vs_bot',
            'model_id': 'test',
            'map_name': 'AbyssalReef',
            'random_seed': 10,
            'home_race': 'zerg',
            'away_race': 'zerg',
            'difficulty': Difficulty.very_easy,
            'build': None,
            'data_push_length': 64,
        }
        return job2


class DummyModelRequester:
    def __init__(self, cfg):
        pass

    def request_model(self, job, agent_no):
        print('received request, job:{}, agent_no:{}'.format(str(job), agent_no))
        return 'not a model'


class DummyDataPusher:
    def __init__(self, cfg):
        pass

    def push(self, job, agent_no, data_buffer):
        print('pushed agent no:{} len:{}'.format(agent_no, len(data_buffer)))


def main(unused_argv):
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    ta = TestActor(cfg)
    ta.run()


if __name__ == '__main__':
    app.run(main)
