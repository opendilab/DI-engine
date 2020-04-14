"""
Test script for actor, this is intended to be run as a independent test, not a pytest unittest
as this need starting SC2 games and is resource heavy
Example Usage:

    srun -p x_cerebra --gres=gpu:1 -w SH-IDC1-10-198-6-64 \
    python3 -u -m sc2learner.tests.alphastar_actor_env_test \
    --config_path test_alphastar_actor.yaml --nofake_dataset

If you want to test this script in your local computer, try to run:

    python alphastar_actor_env_test.py --fake_dataset --config_path test_alphastar_actor.yaml
or this if you want to test real SC2 environment
    python alphastar_actor_env_test.py --nofake_dataset --config_path test_alphastar_actor.yaml --check_data_structure

"""
import random
import time

import yaml
import torch
from absl import app
from absl import flags
from easydict import EasyDict

from sc2learner.worker.actor.alphastar_actor import AlphaStarActor
from sc2learner.data.fake_dataset import fake_stat_processed, get_single_step_data
from sc2learner.tests.fake_env import FakeEnv
from sc2learner.envs.alphastar_env import AlphaStarEnv

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', 'test_alphastar_actor.yaml', 'Path to the config yaml file for test')
flags.DEFINE_bool('fake_dataset', True, 'Whether to use fake dataset')
flags.DEFINE_bool('check_data_structure', False, 'Compare the output and FakeEnv')
flags.DEFINE_bool('single_agent', False, 'Test game_vs_bot mode')

IGNORE_LIST = ['actions']


def recu_check_keys(ref, under_test, trace='ROOT'):
    import warnings
    # only testing shape and type
    for item in IGNORE_LIST:
        if item in trace:
            print('Skipped {}'.format(trace))
            return
    # print('Checking {}'.format(trace))
    if under_test is None and ref is not None\
       or ref is None and under_test is not None:
        warnings.warn('Only one is None. REF={} DUT={} {}'.format(ref, under_test, trace))
    elif isinstance(under_test, torch.Tensor) or isinstance(ref, torch.Tensor):
        assert(isinstance(under_test, torch.Tensor) and isinstance(ref, torch.Tensor)),\
            'one is tensor and the other is not tensor or None {}'.format(trace)
        if under_test.size() != ref.size():
            warnings.warn('Mismatch size: REF={} DUT={} {}'.format(ref.size(), under_test.size(), trace))
    elif isinstance(under_test, list) or isinstance(under_test, tuple):
        if len(under_test) != len(ref):
            warnings.warn('Mismatch length: REF={} DUT={} {}'.format(len(ref), len(under_test), trace))
        for n in range(min(len(ref), len(under_test))):
            recu_check_keys(ref[n], under_test[n], trace=trace + ':' + str(n))
    elif isinstance(under_test, dict):
        assert isinstance(ref, dict)
        for k, v in ref.items():
            if k in under_test and k not in IGNORE_LIST:
                recu_check_keys(v, under_test[k], trace=trace + ':' + str(k))
            else:
                warnings.warn('Missing key: {}'.format(trace + ':' + str(k)))


class TestEnv(AlphaStarEnv):
    def reset(self):
        obs = super().reset()
        if FLAGS.check_data_structure:
            recu_check_keys(get_single_step_data(), obs[0])
        return obs

    def step(self, actions):
        out = super().step(actions)
        return out


class TestActor(AlphaStarActor):
    def __init__(self, cfg):
        super(TestActor, self).__init__(cfg)
        self._module_init()

    def _make_env(self, players):
        if FLAGS.fake_dataset:
            return FakeEnv(len(players))
        else:
            # return super()._make_env(players)
            return TestEnv(self.cfg, players)

    def _module_init(self):
        self.job_getter = DummyJobGetter(self.cfg)
        self.model_loader = DummyModelLoader(self.cfg)
        self.data_pusher = DummyDataPusher(self.cfg)
        self.stat_requester = DummyStatLoader(self.cfg)
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
                print('clipping delay == 0')
                act[n]['delay'] = torch.LongTensor([random.randint(0, 10)])
            print('Act {}:{}'.format(n, str(act[n])))
        return act


class DummyJobGetter:
    def __init__(self, cfg):
        self.connection = None
        self.job_request_id = 0
        pass

    def get_job(self, actor_uid):
        print('received job req from:{}'.format(actor_uid))
        if FLAGS.single_agent:
            job = {
                'job_id': 'test0',
                'game_type': 'game_vs_bot',
                'obs_compressor': 'lz4',
                'model_id': 'test',
                'teacher_model_id': 'test',
                'stat_id': '',
                'map_name': 'AbyssalReef',
                'random_seed': 10,
                'home_race': 'zerg',
                'away_race': 'zerg',
                'difficulty': 'very_easy',
                'build': None,
                'data_push_length': 64,
            }
        else:
            job = {
                'job_id': 'test0',
                'game_type': 'self_play',
                'obs_compressor': 'lz4',
                'model_id': 'test',
                'teacher_model_id': 'test',
                'stat_id': '',
                'map_name': 'AbyssalReef',
                'random_seed': 10,
                'home_race': 'zerg',
                'away_race': 'zerg',
                'data_push_length': 64,
            }
        return job


class DummyModelLoader:
    def __init__(self, cfg):
        pass

    def load_model(self, job, agent_no, model):
        print('received request, job:{}, agent_no:{}'.format(str(job), agent_no))

    def load_teacher_model(self, job, model):
        pass


class DummyDataPusher:
    def __init__(self, cfg):
        pass

    def push(self, metadata, data_buffer):
        print('pushed agent no:{} len:{}'.format(metadata['agent_no'], len(data_buffer)))

    def finish_job(self, job_id):
        pass


class DummyStatLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def request_stat(self, job, agent_no):
        print('received stat req')
        return fake_stat_processed()


def main(unused_argv):
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    ta = TestActor(cfg)
    ta.run()


if __name__ == '__main__':
    app.run(main)
