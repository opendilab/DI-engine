import os
import sys
import time
import json
import threading
import requests
import numpy as np
from itertools import count
import logging
import argparse
import yaml
import traceback
import uuid
import random
from easydict import EasyDict
from queue import Queue
import torch

from sc2learner.data.online import ReplayBuffer
from sc2learner.utils import read_file_ceph, save_file_ceph, LockContext
from sc2learner.league import LeagueManager
from sc2learner.envs import StatManager


class Coordinator(object):
    def __init__(self, cfg):
        super(Coordinator, self).__init__()
        self.cfg = cfg

        self.ceph_path = cfg['system']['ceph_model_path']
        self.use_fake_data = cfg['coordinator']['use_fake_data']
        if self.use_fake_data:
            self.fake_model_path = cfg['coordinator']['fake_model_path']
            self.fake_stat_path = cfg['coordinator']['fake_stat_path']
        # self.learner_port = cfg['system']['learner_port']
        self.league_manager_port = cfg['system']['league_manager_port']

        self.resume_dir = cfg.system.resume_dir
        self.resume_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

        # {manager_uid: {actor_uid: [job_id]}}
        self.manager_record = {}
        # {job_id: {content: info, state: running/finish}}
        self.job_record = {}
        # {learner_uid: {"learner_ip_port_list": [[learner_ip, learner_port]],
        #                "job_ids": [job_id],
        #                "checkpoint_path": checkpoint_path,
        #                "replay_buffer": replay_buffer,
        #                "ret_metadatas": {data_index: metadata},
        #                "last_beats_time": last_beats_time,
        #                "state": choices in ['alive', 'dead']}
        self.learner_record = {}

        self.url_prefix_format = 'http://{}:{}/'
        if not self.use_fake_data:
            self.stat_manager = StatManager(cfg.system.stat_path_list)
        self.map_name_list = cfg.env.map_name_list
        assert len(self.map_name_list) > 0

        self.lock = LockContext(lock_type='process')
        self.save_ret_metadata_num = 5
        self._set_logger()

        # for league
        self.player_to_learner = {}
        self.learner_to_player = {}
        self.job_queue = Queue()
        self.league_manager_flag = False
        # thread to launch league manager
        launch_league_thread = threading.Thread(target=self._launch_league_manager)
        launch_league_thread.data_index = True
        launch_league_thread.start()
        self.logger.info("[UP] launch league manager thread ")

        self.check_dead_learner_freq = cfg.system.coordinator_check_dead_learner_freq

        # thread to check actor if dead
        check_learner_dead_thread = threading.Thread(target=self.check_learner_dead)
        check_learner_dead_thread.daemon = True
        check_learner_dead_thread.start()
        self.logger.info("[UP] check learner dead thread ")

        # resume
        self._load_resume()
        self.save_resume_freq = 60 * 1

        # thread to save resume
        check_resume_thread = threading.Thread(target=self.check_resume)
        check_resume_thread.daemon = True
        check_resume_thread.start()
        self.logger.info("[UP] check resume thread ")

    def close(self):
        for k, v in self.learner_record.items():
            self.learner_record[k]['replay_buffer'].close()

    def _get_random_map_name(self):
        return np.random.choice(self.map_name_list)

    def _set_logger(self, level=1):
        self.logger = logging.getLogger("coordinator.log")

    def _load_resume(self):
        if self.cfg.system.coordinator_resume_path and os.path.isfile(self.cfg.system.coordinator_resume_path):
            data = torch.load(self.cfg.system.coordinator_resume_path)
            self.manager_record, self.job_record, self.learner_record = data
            for k, v in self.learner_record.items():
                self.learner_record[k]['replay_buffer'] = ReplayBuffer(EasyDict(self.cfg['replay_buffer']))
                self.learner_record[k]['replay_buffer'].run()
                self.learner_record[k]['ret_metadatas'] = {}
                self.learner_record[k]['last_beats_time'] = time.time()

    def _save_resume(self):
        tmp = {}
        for k, v in self.learner_record.items():
            tmp[k] = {}
            tmp[k]['learner_ip_port_list'] = v['learner_ip_port_list']
            tmp[k]['job_ids'] = v['job_ids']
            tmp[k]['checkpoint_path'] = v['checkpoint_path']
            tmp[k]['state'] = v['state']
        data = [self.manager_record, self.job_record, tmp]
        torch.save(data, os.path.join(self.resume_dir, 'coordinator.resume.' + self.resume_label))

    def _get_job(self):
        '''
            Overview: return job info for actor
            Returns:
                - (:obj`dict`): job info
        '''
        job_id = str(uuid.uuid1())
        ret = {}

        if self.use_fake_data:
            if not self.learner_record:
                self.learner_record['test1'] = {
                    "learner_ip_port_list": [['0.0.0.0'], [11111]],
                    "job_ids": [],
                    "checkpoint_path": '',
                    "replay_buffer": ReplayBuffer(EasyDict(self.cfg['replay_buffer'])),
                    'ret_metadatas': {},
                    "last_beats_time": int(time.time()),
                    "state": 'alive'
                }
                self.learner_record['test2'] = {
                    "learner_ip_port_list": [['0.0.0.1'], [11112]],
                    "job_ids": [],
                    "checkpoint_path": '',
                    "replay_buffer": ReplayBuffer(EasyDict(self.cfg['replay_buffer'])),
                    'ret_metadatas': {},
                    "last_beats_time": int(time.time()),
                    "state": 'alive'
                }
                self.learner_record['test1']['replay_buffer'].run()
                self.learner_record['test2']['replay_buffer'].run()
            learner_uid1 = random.choice(list(self.learner_record.keys()))
            learner_uid2 = random.choice(list(self.learner_record.keys()))
            model_name1 = self.fake_model_path
            model_name2 = self.fake_model_path
            ret = {
                'job_id': job_id,
                'learner_uid': [learner_uid1, learner_uid2],
                'stat_id': [self.fake_stat_path, self.fake_stat_path],
                'game_type': 'league',
                'step_data_compressor': 'lz4',
                'model_id': [model_name1, model_name2],
                'teacher_model_id': model_name1,
                'map_name': 'AbyssalReef',
                'random_seed': 0,
                'home_race': 'zerg',
                'away_race': 'zerg',
                'difficulty': 'easy',
                'build': 'random',
                'data_push_length': 8
            }
        else:
            ret = self.job_queue.get()
        return ret

    # deprecated
    def deal_with_register_model(self, learner_uid, model_name):
        '''
            Overview: deal with register from learner to register model
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - model_name (:obj:`str`): model's name saved in ceph
        '''
        return True

    def deal_with_register_manager(self, manager_uid):
        '''
            Overview: deal with register from manager
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
        '''
        if manager_uid not in self.manager_record:
            self.manager_record[manager_uid] = {}
        return True

    def _launch_league_manager(self):
        while True:
            if self.league_manager_flag and len(self.player_ids) == len(self.player_to_learner):
                try:
                    url_prefix = self.url_prefix_format.format(self.league_manager_ip, self.league_manager_port)
                    response = requests.get(url_prefix + "league/run_league").json()
                    if response['code'] == 0:
                        self.logger.info('league_manager run with table {}. '.format(self.player_to_learner))
                        break
                except Exception as e:
                    self.logger.info('launch_league_thread error {}'.format(e))
            time.sleep(10)

    def deal_with_register_learner(self, learner_uid, learner_ip, learner_port, learner_re_register=False):
        '''
            Overview: deal with register from learner, make learner and player pairs
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
        '''
        if hasattr(self, 'player_ids'):
            with self.lock:
                self.logger.info('now learner_record: {}'.format(self.learner_record))
                if learner_uid not in self.learner_record:
                    if len(self.player_to_learner) < len(self.player_ids):
                        self.learner_record[learner_uid] = {
                            "learner_ip_port_list": [[learner_ip, learner_port]],
                            "job_ids": [],
                            "checkpoint_path": '',
                            "replay_buffer": ReplayBuffer(EasyDict(self.cfg['replay_buffer'])),
                            "ret_metadatas": {},
                            "last_beats_time": int(time.time()),
                            "state": 'alive'
                        }
                        self.learner_record[learner_uid]['replay_buffer'].run()
                        for index, player_id in enumerate(self.player_ids):
                            if player_id not in self.player_to_learner:
                                self.player_to_learner[player_id] = learner_uid
                                self.learner_to_player[learner_uid] = player_id
                                self.learner_record[learner_uid]['checkpoint_path'] = self.player_ckpts[index]
                                self.logger.info('learner ({}) set to player ({})'.format(learner_uid, player_id))
                                break
                        self.logger.info(
                            '{}/{} learners have been registered'.format(len(self.player_to_learner), len(self.player_ids))
                        )
                    else:
                        self.logger.info(
                            'learner {} try to register, but enough learners have been registered.'.format(learner_uid)
                        )
                        return False
                else:
                    if learner_re_register:
                        self.learner_record[learner_uid]['learner_ip_port_list'] = []
                        self.learner_record[learner_uid]['state'] = 'alive'
                    if [learner_ip, learner_port] not in self.learner_record[learner_uid]['learner_ip_port_list']:
                        self.learner_record[learner_uid]['learner_ip_port_list'].append([learner_ip, learner_port])

                self.logger.info('learner ({}) register, ip {}, port {}'.format(learner_uid, learner_ip, learner_port))
                return self.learner_record[learner_uid]['checkpoint_path']
        else:
            if not hasattr(self, 'player_ids'):
                self.logger.info('learner can not register now, because league manager is not set up')
            if hasattr(self, 'player_ids') and len(self.player_to_learner) == len(self.player_ids):
                self.logger.info('enough learners have been registered.')
            return False

    def deal_with_ask_for_job(self, manager_uid, actor_uid):
        '''
            Overview: deal with job request from manager
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
                - actor_uid (:obj:`str`): actor's uid
        '''
        job = self._get_job()
        job_id = job['job_id']
        if manager_uid not in self.manager_record:
            self.deal_with_register_manager(manager_uid)
        if actor_uid not in self.manager_record[manager_uid]:
            self.manager_record[manager_uid][actor_uid] = []
        self.manager_record[manager_uid][actor_uid].append(job_id)
        self.job_record[job_id] = {'content': job, 'state': 'running'}
        return job

    def deal_with_get_metadata(self, manager_uid, actor_uid, job_id, metadata):
        '''
            Overview: when receiving manager's request of sending metadata, return True/False
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
                - metadata (:obj:`dict`): actor's metadata
            Returns:
                - (:obj`bool`): state
        '''
        # assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        learner_uid = metadata['learner_uid']
        self.learner_record[learner_uid]['replay_buffer'].push_data(metadata)
        return True

    def deal_with_finish_job(self, manager_uid, actor_uid, job_id, result):
        '''
            Overview: when receiving actor's request of finishing job, ,return True/False
            Arguments:
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
            Returns:
                - (:obj`bool`): state
        '''
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        self.job_record[job_id]['state'] = 'finish'
        if not self.use_fake_data:
            home_learner_uid = self.job_record[job_id]['content']['learner_uid'][0]
            away_learner_uid = self.job_record[job_id]['content']['learner_uid'][1]
            home_id = home_learner_uid if home_learner_uid.endswith('_sl') else self.learner_to_player[home_learner_uid]
            away_id = away_learner_uid if away_learner_uid.endswith('_sl') else self.learner_to_player[away_learner_uid]
            match_info = {'home_id': home_id, 'away_id': away_id, 'result': result}
            url_prefix = self.url_prefix_format.format(self.league_manager_ip, self.league_manager_port)
            d = {'match_info': match_info}
            while True:
                try:
                    response = requests.post(url_prefix + "league/finish_match", json=d).json()
                    if response['code'] == 0:
                        return True
                except Exception as e:
                    print("something wrong with league_manager, {}".format(e))
                time.sleep(10)
            return False
        return True

    def deal_with_ask_for_metadata(self, learner_uid, batch_size, data_index):
        '''
            Overview: when receiving learner's request of asking for metadata, return metadatas
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - batch_size (:obj:`int`): batch size
                - data_index (:obj:`int`): data index, return same data if same
            Returns:
                - (:obj`list`): metadata list
        '''
        assert learner_uid in self.learner_record, 'learner_uid ({}) not in learner_record'.format(learner_uid)
        self.learner_record[learner_uid]['last_beats_time'] = int(time.time())
        with self.lock:
            if data_index not in self.learner_record[learner_uid]['ret_metadatas']:
                metadatas = self.learner_record[learner_uid]['replay_buffer'].sample(batch_size)
                self.learner_record[learner_uid]['ret_metadatas'][data_index] = metadatas
                self.logger.info('[ask_for_metadata] [first] learner ({}) data_index ({})'.format(learner_uid, data_index))
            else:
                metadatas = self.learner_record[learner_uid]['ret_metadatas'][data_index]
                self.logger.info('[ask_for_metadata] [second] learner ({}) data_index ({})'.format(learner_uid, data_index))
        # clean saved metadata in learner_record
        for i in range(data_index - self.save_ret_metadata_num):
            if i in self.learner_record[learner_uid]['ret_metadatas']:
                del self.learner_record[learner_uid]['ret_metadatas'][i]
        return metadatas

    def deal_with_update_replay_buffer(self, learner_uid, update_info):
        '''
            Overview: when receiving learner's request of updating replay buffer, return True/False
            Arguments:
                - update_info (:obj:`dict`): info dict
            Returns:
                - (:obj`bool`): True
        '''
        self.learner_record[learner_uid]['last_beats_time'] = int(time.time())
        self.learner_record[learner_uid]['replay_buffer'].update(update_info)
        return True

    def deal_with_get_learner_train_step(self, learner_uid, train_step):
        self.learner_record[learner_uid]['last_beats_time'] = int(time.time())
        player_id = self.learner_to_player.get(learner_uid)
        player_info = {'player_id': player_id, 'train_step': train_step}
        self.league_manager.update_active_player(player_info)
        return True

    def deal_with_register_league_manager(self, league_manager_ip, player_ids, player_ckpts):
        self.league_manager_ip = league_manager_ip
        self.player_ids = player_ids
        self.player_ckpts = player_ckpts
        self.logger.info('register league_manager from {}'.format(self.league_manager_ip))
        self.league_manager_flag = True
        return True

    def deal_with_ask_learner_to_reset(self, player_id, checkpoint_path):
        learner_uid = self.player_to_learner[player_id]
        d = {'checkpoint_path': checkpoint_path}
        for learner_ip, learner_port in self.learner_record[learner_uid]['learner_ip_port_list']:
            url_prefix = self.url_prefix_format.format(learner_ip, learner_port)
            while True:
                try:
                    response = requests.post(url_prefix + "learner/reset", json=d).json()
                    if response['code'] == 0:
                        break
                except Exception as e:
                    print("something wrong with learner {}, {}".format(learner_uid, e))
                time.sleep(0.5)
        return True

    def deal_with_add_launch_info(self, launch_info):
        home_id = launch_info['home_id']
        away_id = launch_info['away_id']
        home_checkpoint_path = launch_info['home_checkpoint_path']
        away_checkpoint_path = launch_info['away_checkpoint_path']
        home_teacher_checkpoint_path = launch_info['home_teacher_checkpoint_path']
        away_teacher_checkpoint_path = launch_info['away_teacher_checkpoint_path']
        home_learner_uid = home_id if home_id.endswith('_sl') else self.player_to_learner[home_id]
        away_learner_uid = away_id if away_id.endswith('_sl') else self.player_to_learner[away_id]
        map_name = self._get_random_map_name()
        random_seed = np.random.randint(0, 314) + int(1e7)
        # stats(len=2, stat path)
        kwargs = {
            'home_race': launch_info['home_race'],
            'away_race': launch_info['away_race'],
            'map_name': map_name,
            'player_id': 'ava'
        }
        stats = self.stat_manager.get_ava_stats(**kwargs)
        job = {
            'job_id': str(uuid.uuid1()),
            'learner_uid': [home_learner_uid, away_learner_uid],
            'stat_id': stats,
            'game_type': 'league',
            'step_data_compressor': 'lz4',
            'model_id': [home_checkpoint_path, away_checkpoint_path],
            'teacher_model_id': home_teacher_checkpoint_path,  # away_teacher_checkpoint_path
            'map_name': map_name,
            'random_seed': random_seed,
            'home_race': launch_info['home_race'],
            'away_race': launch_info['away_race'],
            'data_push_length': self.cfg.train.trajectory_len,
        }
        self.job_queue.put(job)
        return True

    def deal_with_get_heartbeats(self, learner_uid):
        '''
            Overview: when receiving learner's heartbeats, update last_beats_time.
            Arguments:
                - actor_uid (:obj:`str`): learner's uid
            Returns:
                - (:obj`bool`): state
        '''
        assert learner_uid in self.learner_record
        self.learner_record[learner_uid]['last_beats_time'] = int(time.time())
        return True

    ###################################################################################
    #                                     threads                                     #
    ###################################################################################

    def time_format(self, time_item):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_item))

    def deal_with_dead_learner(self, learner_uid, reuse=True):
        self.learner_record[learner_uid]['state'] = 'dead'
        os.system('scancel ' + learner_uid)
        self.logger.info('[kill-dead] dead learner {} was killed'.format(learner_uid))

    def check_learner_dead(self):
        while True:
            nowtime = int(time.time())
            for learner_uid, learner_info in self.learner_record.items():
                if learner_info['state'] == 'alive' and\
                   nowtime - learner_info['last_beats_time'] > self.check_dead_learner_freq:
                    # dead learner
                    self.logger.info(
                        "[coordinator][check_learner_dead] {} is dead, last_beats_time = {}".format(
                            learner_uid, self.time_format(learner_info['last_beats_time'])
                        )
                    )
                    self.deal_with_dead_learner(learner_uid)
            time.sleep(self.check_dead_learner_freq)

    def check_resume(self):
        self.lasttime = int(time.time())
        while True:
            nowtime = int(time.time())
            if nowtime - self.lasttime > self.save_resume_freq:
                self._save_resume()
                p = os.path.join(self.resume_dir, 'coordinator.resume.' + self.resume_label)
                self.logger.info('[resume] save to {}'.format(p))
                self.lasttime = nowtime
            time.sleep(self.save_resume_freq)

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

    def deal_with_get_all_manager(self):
        return self.manager_record

    def deal_with_get_all_learner(self):
        return self.learner_record

    def deal_with_get_all_job(self):
        return self.job_record

    def deal_with_get_replay_buffer(self, learner_uid):
        return self.learner_record[learner_uid]['replay_buffer']

    def deal_with_get_job_queue(self):
        pass

    def deal_with_push_data_to_replay_buffer(self, learner_uid):
        job_id = '8d2e8eda-83d9-11ea-8bb0-1be4f1872daf'
        # learner_uid = '3458436'
        trajectory_path = 'model_main_player_zerg_0_ckpt'\
            '.pth_job_0098e642-841e-11ea-9918-6f27a4855242_agent_0_step_1159_0707b170-8423-11ea-99b0-db6573da5763.traj'
        self.learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        self.learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        self.learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        self.learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        return True
