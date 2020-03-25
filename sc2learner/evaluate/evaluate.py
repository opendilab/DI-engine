import random
import time
from multiprocessing import Pool
import copy

import torch
import numpy as np
import yaml
from absl import app
from absl import flags
from easydict import EasyDict

from sc2learner.worker.actor.alphastar_actor import AlphaStarActor
from sc2learner.torch_utils import build_checkpoint_helper
from sc2learner.utils import build_logger
from pysc2.lib.action_dict import ACTION_INFO_MASK

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', '', 'Path to the config yaml file')


class EvalActor(AlphaStarActor):
    def _make_env(self, players):
        self.action_counts = [[0] * (max(ACTION_INFO_MASK.keys()) + 1)] * self.agent_num
        return super()._make_env(players)

    def _module_init(self):
        self.job_getter = EvalJobGetter(self.cfg)
        self.model_loader = LocalModelLoader(self.cfg)
        self.stat_requester = LocalStatLoader(self.cfg)
        self.data_pusher = EvalTrajProcessor(self.cfg)
        print(self.cfg)
        self.last_time = None

    def action_modifier(self, act, step):
        if self.cfg.evaluate.get(show_system_stat, False) and self.cfg.env.use_cuda:
            print('Max CUDA memory:{}'.format(torch.cuda.max_memory_allocated()))

        # Here we implement statistics and optional clipping on actions
        for n in range(len(act)):
            if act[n] is not None:
                if act[n]['delay'] == 0:
                    act[n]['delay'] = 1
                self.action_counts[n][self.env.get_action_type(act[n])] += 1
            print('Act {}:{}:{:5}:{}'.format(self.cfg.evaluate.job_id, n, step, self.env.action_to_string(act[n])))
        return act


class EvalJobGetter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.job_req_id = 0

    def get_job(self, actor_id):
        print('received job req from:{}'.format(actor_id))
        if self.cfg.evaluate.game_type == 'game_vs_bot':
            job = {
                'game_type': 'game_vs_bot',
                'model_id': ['agent0'],
                'teacher_model_id': None,
                'stat_id': ['agent0'],
                'map_name': self.cfg.evaluate.map_name,
                'random_seed': self.cfg.evaluate.seed,
                'home_race': self.cfg.evaluate.home_race,
                'away_race': self.cfg.evaluate.away_race,
                'difficulty': self.cfg.evaluate.bot_difficulty,
                'build': self.cfg.evaluate.bot_build,
                'data_push_length': 64,
            }
        elif self.cfg.evaluate.game_type == 'self_play':
            job = {
                'game_type': 'self_play',
                'model_id': ['agent0', 'agent1'],
                'teacher_model_id': None,
                'stat_id': ['agent0', 'agent1'],
                'map_name': self.cfg.evaluate.map_name,
                'random_seed': self.cfg.evaluate.seed,
                'home_race': self.cfg.evaluate.home_race,
                'away_race': self.cfg.evaluate.away_race,
                'data_push_length': 64,
            }
        else:
            raise NotImplementedError('Unknown game_type')
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

    def load_teacher_model(self, job, model):
        raise NotImplementedError('Why we need teacher model for eval?')


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
    use_multiprocessing = cfg.evaluate.get("use_multiprocessing", False)
    if use_multiprocessing:
        pool = Pool(min(cfg.evaluate.num_episodes, cfg.evaluate.num_instance_per_node))
        var_list = []
        for n in range(cfg.evaluate.num_episodes):
            new_cfg = copy.deepcopy(cfg)
            if not cfg.evaluate.get('fix_seed', False):
                new_cfg.evaluate.seed = seed_gen(n)
            new_cfg.evaluate.job_id = n
            var_list.append(new_cfg)
        return_list = pool.map(run_episode, var_list)
        pool.close()
    else:
        cfg.evaluate.job_id = 0
        return_list = [run_episode(cfg)]
    agent_nums, return_sums, action_counts = zip(*return_list)
    agent_num = agent_nums[0]  # assumming all jobs have the same number of agents
    return_sum = np.mean(return_sums, axis=0)
    action_counts = np.mean(action_counts, axis=0)  # axis 0:games, 1:agents, 2:actions
    print('Returns: {}'.format(str(return_sum)))
    for n in range(agent_num):
        print('Action Statistics of Agent {}'.format(n))
        sorted_action_counts = sorted(enumerate(action_counts[n]), key=lambda x: x[1], reverse=True)
        for action_count in sorted_action_counts:
            if action_count[1]:
                print('ID: {:3d}  Times: {:5}'.format(action_count[0], action_count[1]))


def seed_gen(seq):
    return seq


def run_episode(cfg):
    ea = EvalActor(cfg)
    ea.run_episode()
    if cfg.evaluate.get('save_replay', True) and cfg.evaluate.replay_path:
        ea.save_replay(cfg.evaluate.replay_path)
    return ea.agent_num, ea.data_pusher.return_sum, ea.action_counts


if __name__ == '__main__':
    app.run(main)
