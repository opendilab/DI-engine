import time
import os
import random
from multiprocessing import Pool
import copy
import pickle
import logging
import sys

import torch
import numpy as np
import yaml
from absl import app
from absl import flags
from easydict import EasyDict

from sc2learner.worker.actor.alphastar_actor import AlphaStarActor
from sc2learner.torch_utils import build_checkpoint_helper
from sc2learner.utils import build_logger, read_file_ceph
from pysc2.lib.action_dict import ACTION_INFO_MASK

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', '../../DATA/Config/eval.yaml', 'Path to the config yaml file')
FLAGS(sys.argv)
Z_LIST = {
    's3://replay_decode_493_0515/Zerg_Zerg_5722_7ccb1bd08dca7715db2282abf9bb55dec4f34874fa629a80f48b8521da71dc13': '12d_base',  # noqa
    's3://replay_decode_493_0515/Zerg_Zerg_6469_b9cd6cecb271a9fb61a36e50c04e082b5e27876ed587b4f91e2ee63365f0e8a6': 'normal_hydra_lurker',  # noqa
    's3://replay_decode_493_0515/Zerg_Zerg_5678_25cc1c8fbf7fff4ab12d13e4ae027e3e5d9fe19372bf9ea80eefdf8d6a45addf': 'pool_base_muta',  # noqa
    's3://replay_decode_493_0515/Zerg_Zerg_5689_5550a6274f14b6f5a6bfbdb93249aa05baf4f8bee1e68cf0295c4c5d95d2c9a2': 'base_pool_normal',  # noqa
    's3://replay_decode_493_0515/Zerg_Zerg_5670_33042f61da2d477a6f13a6cb6ded3ebeed109cd2aac358c3e1ffb5e16eaf1027': 'base_pool_dog_rush',  # noqa
}

DIFFICULTY = [
    'very_easy', 'easy', 'medium', 'medium_hard', 'hard', 'harder', 'very_hard', 'cheat_vision', 'cheat_money',
    'cheat_insane'
]


class EvalActor(AlphaStarActor):
    def __init__(self, cfg):
        super(EvalActor, self).__init__(cfg)
        self.enable_push_data = False
        self._module_init()
        self.bot_multi_test = self.cfg.evaluate.get('bot_multi_test', False)
        self.last_print = 0

    def _module_init(self):
        self.job_getter = EvalJobGetter(self.cfg)
        self.model_loader = LocalModelLoader(self.cfg)
        self.stat_requester = LocalStatLoader(self.cfg)
        self.data_pusher = EvalTrajProcessor(self.cfg)
        print(self.cfg)
        self.last_time = None

    def action_modifier(self, act, step):
        if self.cfg.evaluate.get('show_system_stat', False) and self.cfg.env.use_cuda:
            print('Max CUDA memory:{}'.format(torch.cuda.max_memory_allocated()))

        # Here we implement statistics and optional clipping on actions
        for n in range(len(act)):
            if act[n] is not None:
                if act[n]['action']['target_location'] is not None:
                    act[n]['action']['target_location'] = act[n]['action']['target_location'].float()
                if False:
                    locations = [
                        [15.5, 13.5], [13.5, 36.5], [11.5, 66.5], [10.5, 95.5], [12.5, 126.5], [48.5, 21.5],
                        [39.5, 59.5], [42.5, 125.5], [71.5, 118.5], [80.5, 80.5], [77.5, 14.5], [109.5, 44.5],
                        [108.5, 73.5], [106.5, 103.5], [104.5, 126.5]
                    ]
                    action = act[n]
                    for i in range(self.agent_num):
                        if action['action']['action_type'].item() == 32:
                            x = action['action']['target_location'][1]
                            y = action['action']['target_location'][0]
                            threshold = 100.0
                            distance = []
                            for location in locations:
                                distance.append((x - location[0])**2 + (y - location[1])**2)
                            idx = distance.index(min(distance))
                            if distance[idx] <= threshold:
                                action['action']['target_location'][1] = locations[idx][0]
                                action['action']['target_location'][0] = locations[idx][1]
                            else:
                                action['action']['action_type'] *= 0

                        # if action['action_type'].item() == 48:
                        #     action['target_location'][1] = 97
                        #     action['target_location'][0] = 120.5
                        # if action['action_type'].item() == 53:
                        #     action['target_location'][1] = 100
                        #     action['target_location'][0] = 130.5

                if act[n]['action']['delay'] == 0:
                    print('clipping delay == 0 to 1')
                    act[n]['action']['delay'] = torch.LongTensor([1])
        return act

    def _set_agent_mode(self):
        for agent in self.agents:
            agent.eval()
            # agent.train()

    def _print_action_info(self, step):
        def action_to_dict(action):
            tmp = {}
            for f in action._fields:
                tmp[f] = getattr(action, f)
            return tmp

        def action_to_string(action):
            return '[Action: type({}) delay({}) queued({}) selected_units({}) target_units({}) target_location({})]'.format(  # noqa
                action['action_type'], action['delay'], action['queued'], action['selected_units'],
                action['target_units'], action['target_location']
            )

        if not hasattr(self, 'action_counts'):
            self.action_counts = [[0 for _ in range(max(ACTION_INFO_MASK.keys()) + 1)] for _ in range(self.agent_num)]
        action = []
        for i, a in enumerate(self.env.action):
            if a is not None:
                action.append(action_to_dict(a))
            else:
                action.append(None)
        for n in range(self.agent_num):
            if action[n] is None:
                print('Act {}:{}:{}: None'.format(self.cfg.evaluate.job_id, n, step))
                continue
            self.action_counts[n][action[n]['action_type']] += 1
            if self.bot_multi_test:
                if step - self.last_print > 1000:
                    self.last_print = step
                    logging.info(
                        '{:15}{:22} steps: {}/{}'.format(
                            self.cfg.evaluate.player1.difficulty, Z_LIST[self.cfg.evaluate.player0.stat], step,
                            self.cfg.env.game_steps_per_episode
                        )
                    )
            else:
                print('Act {}:{}:{}:{}'.format(self.cfg.evaluate.job_id, n, step, action_to_string(action[n])))


class EvalJobGetter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.job_req_id = 0

    def get_job(self, actor_uid):
        print('received job req from:{}'.format(actor_uid))
        game_type = self.cfg.evaluate.game_type
        assert game_type in ['game_vs_bot', 'game_vs_agent',
                             'agent_vs_agent'], 'Unknown game_type: {}'.format(game_type)
        job = {
            'job_id': '{}_{}'.format(game_type, actor_uid),
            'game_type': game_type,
            'step_data_compressor': 'simple',
            'map_name': self.cfg.evaluate.map_name,
            'random_seed': self.cfg.evaluate.seed,
            'player0': self.cfg.evaluate.player0,
            'player1': self.cfg.evaluate.player1,
            'data_push_length': 100000  # necessary for compatibility
        }
        # TODO(nyz) config(yaml) deal with `None`
        if job['player0']['teacher_model'] == 'None':
            job['player0']['teacher_model'] = None
        if job['player1']['teacher_model'] == 'None':
            job['player1']['teacher_model'] = None
        self.job_req_id += 1
        return job


class LocalModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_model(self, job, agent_no, model):
        print('received request, job:{}, agent_no:{}'.format(str(job), agent_no))
        t = time.time()
        model_path = job['player{}'.format(agent_no)]['model']
        helper = build_checkpoint_helper('')
        helper.load(model_path, model, prefix_op='remove', prefix='module.', strict=True)
        print('loaded, time:{}'.format(time.time() - t))

    def load_teacher_model(self, job, model):
        raise NotImplementedError('Why we need teacher model for eval?')


class LocalStatLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def request_stat(self, job, agent_no):
        path = [
            '../../DATA/Z/normal_hydra_lurker.pkl',
            '../../DATA/Z/12d_base.pkl',
            '../../DATA/Z/base_pool_dog_rush.pkl',
            '../../DATA/Z/base_pool_normal.pkl',
            '../../DATA/Z/pool_base_muta.pkl',
        ]
        if self.cfg.evaluate.get('local', False):
            import random
            p = path[random.randint(0, 4)]
            print('[INFO] choosed Z:', p)
        else:
            dirs = os.path.dirname(__file__).split('/')
            dirs = dirs[:dirs.index('sc2learner')]
            p = '/' + os.path.join(*dirs) + '/DATA/Z/' + Z_LIST[self.cfg.evaluate.stat_path[job['stat_id'][agent_no]]
                                                                ] + '.pkl'  # noqa
        f = open(p, 'rb')
        stat = pickle.load(f)
        f.close()
        return stat


class EvalTrajProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.return_sum = []

    def push(self, metadata, data_buffer):
        agent_no = metadata['agent_no']
        traj_return = metadata.get('final_reward', 0)
        print('agent no:{} ret:{}'.format(agent_no, traj_return))
        if agent_no + 1 > len(self.return_sum):
            # extending return store
            self.return_sum.extend([0] * (agent_no - len(self.return_sum) + 1))
        self.return_sum[agent_no] += traj_return

    def finish_job(self, job_id, result):
        pass


def main(unused_argv):
    with open(FLAGS.config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    # cfg.common.save_path = os.path.dirname(FLAGS.config_path)
    use_multiprocessing = cfg.evaluate.get("use_multiprocessing", False)
    if use_multiprocessing:
        pool = Pool(min(cfg.evaluate.num_episodes, cfg.evaluate.num_instance_per_node))
        var_list = []
        if cfg.evaluate.get("bot_multi_test", False):
            pool = Pool(cfg.evaluate.num_instance_per_node)
            difficulty = DIFFICULTY[DIFFICULTY.index(cfg.evaluate.
                                                     bot_difficulty):DIFFICULTY.index(cfg.evaluate.max_bot_difficulty) +
                                    1]
            z_keys = list(Z_LIST.keys())
            random.shuffle(z_keys)
            z_list = {k: Z_LIST[k] for k in z_keys[:cfg.evaluate.z_number]}
            print('total games:', cfg.evaluate.num_episodes * len(difficulty) * len(z_list))
            print('selected difficulty:', difficulty)
            print('selected Z:', list(z_list.values()))
            for i in range(len(difficulty)):
                for j in range(len(z_list)):
                    for n in range(cfg.evaluate.num_episodes):
                        new_cfg = copy.deepcopy(cfg)
                        if not cfg.evaluate.get('fix_seed', False):
                            new_cfg.evaluate.seed = seed_gen(n)
                        new_cfg.evaluate.bot_difficulty = difficulty[i]
                        new_cfg.evaluate.stat_path.agent0 = list(z_list.keys())[j]
                        new_cfg.evaluate.job_id = n
                        var_list.append(new_cfg)
        else:
            pool = Pool(min(cfg.evaluate.num_episodes, cfg.evaluate.num_instance_per_node))
            for n in range(cfg.evaluate.num_episodes):
                new_cfg = copy.deepcopy(cfg)
                if not cfg.evaluate.get('fix_seed', False):
                    new_cfg.evaluate.seed = seed_gen(n)
                new_cfg.evaluate.job_id = n
                var_list.append(new_cfg)

        return_list = pool.map(run_episode, var_list)
        pool.close()
        pool.join()
    else:
        cfg.evaluate.job_id = 0
        return_list = [run_episode(cfg)]
    agent_nums, return_sums, action_counts, cfgs = zip(*return_list)
    agent_num = agent_nums[0]  # assuming all jobs have the same number of agents
    return_sum = np.mean(return_sums, axis=0)
    action_counts = np.mean(action_counts, axis=0)  # axis 0:games, 1:agents, 2:actions
    print('Returns: {}'.format(str(return_sum)))
    for n in range(agent_num):
        print('Action Statistics of Agent {}'.format(n))
        sorted_action_counts = sorted(enumerate(action_counts[n]), key=lambda x: x[1], reverse=True)
        for action_count in sorted_action_counts:
            if action_count[1]:
                print('ID: {:3d}  Times: {:5}'.format(action_count[0], action_count[1]))
    if cfg.evaluate.get("bot_multi_test", False):
        result_dict = {d: {z: {'win': 0, 'lose': 0, 'tie': 0} for z in z_list.keys()} for d in difficulty}
        pretty = {1: 'win', -1: 'lose', 0: 'tie'}
        for idx, item in enumerate(return_sums):
            result_dict[cfgs[idx].evaluate.bot_difficulty][cfgs[idx].evaluate.stat_path.agent0][pretty[item[0]]] += 1
        for d in difficulty:
            print(d + ':')
            total_wins = 0
            for z in z_list.keys():
                winrate = result_dict[d][z]['win'] / cfg.evaluate.num_episodes
                total_wins += result_dict[d][z]['win']
                print('  {:20} {}, winrate:{:.2f}'.format(z_list[z] + ':', result_dict[d][z], winrate))
            print('  total_winrate: {:.2f}'.format(total_wins / (cfg.evaluate.num_episodes * len(z_list))))


def seed_gen(seq):
    return seq


def run_episode(cfg):
    ea = EvalActor(cfg)
    ea.run_episode()
    if cfg.evaluate.get('save_replay', True) and cfg.evaluate.replay_path:
        result = {0: 'tie', 1: 'win', -1: 'lose'}
        if cfg.evaluate.bot_multi_test:
            prefix = cfg.evaluate.bot_difficulty + '--' + result[ea.data_pusher.return_sum[0]
                                                                 ] + '--' + Z_LIST[cfg.evaluate.stat_path.agent0]
            logging.info('saving replay:' + prefix)
        else:
            prefix = cfg.evaluate.bot_difficulty + '--' + result[ea.data_pusher.return_sum[0]]
        ea.save_replay(cfg.evaluate.replay_path, prefix)
    return ea.agent_num, ea.data_pusher.return_sum, ea.action_counts, cfg


if __name__ == '__main__':
    app.run(main)
