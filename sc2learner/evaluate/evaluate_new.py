import random
import time
import torch

import yaml
from absl import app
from absl import flags
from easydict import EasyDict

from sc2learner.worker.actor.alphastar_actor import AlphaStarActor
from sc2learner.torch_utils import build_checkpoint_helper

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', '', 'Path to the config yaml file')


class EvalActor(AlphaStarActor):
    def _make_env(self, players):
        return super()._make_env(players)

    def _module_init(self):
        self.job_getter = EvalJobGetter(self.cfg)
        self.model_loader = LocalModelLoader(self.cfg)
        self.stat_requester = LocalStatLoader(self.cfg)
        self.data_pusher = EvalTrajProcessor(self.cfg)
        self.last_time = None

    def action_modifier(self, act):
        t = time.time()
        if self.last_time is not None:
            print('Time between action:{}'.format(t - self.last_time))
        self.last_time = t
        for n in range(len(act)):
            # if act[n]['delay'] == 0:
            #     act[n]['delay'] = 1
            print('Act {}:{}'.format(n, self.env.action_to_string(act[n])))
        return act


class EvalJobGetter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.job_req_id = 0

    def get_job(self, actor_id):
        print('received job req from:{}'.format(actor_id))
        if self.cfg.evaluate.seed:
            random_seed = self.job_req_id
        print('seed:{}'.format(random_seed))
        if self.cfg.evaluate.game_type == 'game_vs_bot':
            job = {
                'game_type': 'game_vs_bot',
                'model_id': ['agent0'],
                'stat_id': ['agent0'],
                'map_name': self.cfg.evaluate.map_name,
                'random_seed': random_seed,
                'home_race': self.cfg.evaluate.home_race,
                'away_race': self.cfg.evaluate.away_race,
                'difficulty': self.cfg.evaluate.bot_difficulty,
                'build': self.cfg.evaluate.bot_build,
                'data_push_length': 64,
            }
        else:
            job = {
                'game_type': 'self_play',
                'model_id': ['agent0', 'agent1'],
                'stat_id': ['agent0', 'agent1'],
                'map_name': self.cfg.evaluate.map_name,
                'random_seed': random_seed,
                'home_race': self.cfg.evaluate.home_race,
                'away_race': self.cfg.evaluate.away_race,
                'data_push_length': 64,
            }
        self.job_req_id += 1
        return job


class LocalModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_model(self, job, agent_no, model):
        print('received request, job:{}, agent_no:{}'.format(str(job), agent_no))
        t = time.time()
        model_path = self.cfg.evaluate.model_path[job['model_id'][agent_no]]
        helper = build_checkpoint_helper('')
        helper.load(model_path, model, prefix_op='remove', prefix='module.')
        print('loaded, time:{}'.format(time.time() - t))


class LocalStatLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def request_stat(self, job, agent_no):
        stat = torch.load(self.cfg.evaluate.stat_path[job['stat_id'][agent_no]], map_location='cpu')
        return stat


class EvalTrajProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.return_sum = []

    def push(self, job, agent_no, data_buffer):
        rewards_list = [d['rewards'] for d in data_buffer]
        traj_return = sum(rewards_list)
        print('agent no:{} ret:{}'.format(agent_no, traj_return))
        if agent_no + 1 > len(self.return_sum):
            self.return_sum.extend([0] * (agent_no - len(self.return_sum) + 1))
        self.return_sum[agent_no] += traj_return


def main(unused_argv):
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    ea = EvalActor(cfg)
    ea.run_episode()
    print(ea.data_pusher.return_sum)
    if cfg.evaluate.replay_path:
        ea.save_replay(cfg.evaluate.replay_path)


if __name__ == '__main__':
    app.run(main)
