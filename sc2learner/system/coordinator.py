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
from multiprocessing import Lock

from sc2learner.data.online import ReplayBuffer
from sc2learner.utils import read_file_ceph, save_file_ceph
from sc2learner.league import LeagueManager


class Coordinator(object):
    def __init__(self, cfg):
        super(Coordinator, self).__init__()
        self.cfg = cfg

        self.ceph_path = cfg['system']['ceph_model_path']
        self.use_fake_data = cfg['coordinator']['use_fake_data']
        if self.use_fake_data:
            self.fake_model_path = cfg['coordinator']['fake_model_path']
            self.fake_stat_path = cfg['coordinator']['fake_stat_path']
        self.learner_port = cfg['system']['learner_port']
        self.league_manager_port = cfg['system']['league_manager_port']

        # {manager_uid: {actor_uid: [job_id]}}
        self.manager_record = {}
        # {job_id: {content: info, state: running/finish}}
        self.job_record = {}
        # {learner_uid: {"learner_ip": learner_ip,
        #                "job_ids": [job_id],
        #                "checkpoint_path": checkpoint_path,
        #                "replay_buffer": replay_buffer,
        #                "ret_metadata": {data_index: metadata}}
        self.learner_record = {}

        self.url_prefix_format = 'http://{}:{}/'

        # TODO(nyz) each learner has its own replay_buffer
        self.replay_buffer = ReplayBuffer(EasyDict(self.cfg['replay_buffer']))
        self.replay_buffer.run()

        self.lock = Lock()
        self.save_ret_metadata_num = 5
        self._set_logger()

        # for league
        self.player_to_learner = {}
        self.learner_to_player = {}
        self.job_queue = Queue()
        self.league_flag = False

    def close(self):
        self.replay_buffer.close()

    def _set_logger(self, level=1):
        self.logger = logging.getLogger("coordinator.log")

    def _acquire_lock(self):
        self.lock.acquire()

    def _release_lock(self):
        self.lock.release()

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
                self.learner_record['test1'] = {"learner_ip": '0.0.0.0', "job_ids": [], "checkpoint_path": '', 'ret_metadatas': {}}
                self.learner_record['test2'] = {"learner_ip": '0.0.0.0', "job_ids": [], "checkpoint_path": '', 'ret_metadatas': {}}
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

    def _tell_league_manager_to_run(self):
        url_prefix = self.url_prefix_format.format(self.league_manager_ip, self.league_manager_port)
        while True:
            try:
                response = requests.get(url_prefix + "league/run_league").json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                print("something wrong with league_manager, {}".format(e))
            time.sleep(10)
        return False

    def deal_with_register_learner(self, learner_uid, learner_ip):
        '''
            Overview: deal with register from learner, make learner and player pairs
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
        '''
        if hasattr(self, 'player_ids'):
            if learner_uid in self.learner_record:
                self.logger.info('learner ({}) exists, ip {}'.format(learner_uid, learner_ip))
                return True
            else:
                self.learner_record[learner_uid] = {"learner_ip": learner_ip, "job_ids": [], "checkpoint_path": '', 'ret_metadatas': {}}
                self.logger.info('learner ({}) register, ip {}'.format(learner_uid, learner_ip))
            
                if len(self.player_to_learner) == len(self.player_ids):
                    self.logger.info('enough learners have been registered.')
                    return False
                
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
            # TODO(nyz) learner load init model
            if len(self.player_ids) == len(self.player_to_learner) and not self.league_flag:
                self.league_flag = True
                self._tell_league_manager_to_run()
                self.logger.info('league_manager run with table {}. '.format(self.player_to_learner))
            return self.learner_record[learner_uid]['checkpoint_path']
        else:
            self.logger.info('learner can not register now, because league manager is not set up')
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
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        self.replay_buffer.push_data(metadata)
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
        self._acquire_lock()
        if data_index not in self.learner_record[learner_uid]['ret_metadatas']:
            metadatas = self.replay_buffer.sample(batch_size)
            self.learner_record[learner_uid]['ret_metadatas'][data_index] = metadatas
            self.logger.info('[ask_for_metadata] [first] learner ({}) data_index ({})'.format(learner_uid, data_index))
        else:
            metadatas = self.learner_record[learner_uid]['ret_metadatas'][data_index]
            self.logger.info('[ask_for_metadata] [second] learner ({}) data_index ({})'.format(learner_uid, data_index))
        self._release_lock()
        # clean saved metadata in learner_record
        for i in range(data_index - self.save_ret_metadata_num):
            if i in self.learner_record[learner_uid]['ret_metadatas']:
                del self.learner_record[learner_uid]['ret_metadatas'][i]
        return metadatas

    def deal_with_update_replay_buffer(self, update_info):
        '''
            Overview: when receiving learner's request of updating replay buffer, return True/False
            Arguments:
                - update_info (:obj:`dict`): info dict
            Returns:
                - (:obj`bool`): True
        '''
        self.replay_buffer.update(update_info)
        return True

    def get_url_prefix(self, learner_uid):
        learner_ip = self.learner_record[learner_uid]['learner_ip']
        url_prefix = self.url_prefix_format.format(learner_ip, self.learner_port)
        return url_prefix

    def deal_with_get_learner_train_step(self, learner_uid, train_step):
        player_id = self.learner_to_player.get(learner_uid)
        player_info = {'player_id': player_id, 'train_step': train_step}
        self.league_manager.update_active_player(player_info)
        return True

    def deal_with_register_league_manager(self, league_manager_ip, player_ids, player_ckpts):
        self.league_manager_ip = league_manager_ip
        self.player_ids = player_ids
        self.player_ckpts = player_ckpts
        self.logger.info('register league_manager from {}'.format(self.league_manager_ip))
        return True

    def deal_with_ask_learner_to_reset(self, player_id, checkpoint_path):
        learner_uid = self.player_to_learner(player_id)
        d = {'checkpoint_path': checkpoint_path}
        url_prefix = self.get_url_prefix(learner_uid)
        while True:
            try:
                response = requests.post(url_prefix + "learner/reset", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                print("something wrong with learner {}, {}".format(learner_uid, e))
            time.sleep(10)
        return False

    def deal_with_add_launch_info(self, launch_info):
        home_id = launch_info['home_id']
        away_id = launch_info['away_id']
        home_checkpoint_path = launch_info['home_checkpoint_path']
        away_checkpoint_path = launch_info['away_checkpoint_path']
        home_teacher_checkpoint_path = launch_info['home_teacher_checkpoint_path']
        away_teacher_checkpoint_path = launch_info['away_teacher_checkpoint_path']
        home_learner_uid = home_id if home_id.endswith('_sl') else self.player_to_learner[home_id]
        away_learner_uid = away_id if away_id.endswith('_sl') else self.player_to_learner[away_id]
        stat = 'Zerg_Zerg_6280_d35c22b2d7e462f1481621cbf765709961e3f9a2a99f8f6c6fa814ccffc831d6.stat_processed'
        job = {
            'job_id': str(uuid.uuid1()),
            'learner_uid': [home_learner_uid, away_learner_uid],
            # TODO(nyz) adaptive z
            'stat_id': [stat, stat],
            'game_type': 'league',
            'step_data_compressor': 'lz4',
            'model_id': [home_checkpoint_path, away_checkpoint_path],
            'teacher_model_id': home_teacher_checkpoint_path,  # away_teacher_checkpoint_path
            # TODO(nyz) random map and seed
            'map_name': 'KairosJunction',
            'random_seed': 0,
            'home_race': launch_info['home_race'],
            'away_race': launch_info['away_race'],
            'data_push_length': self.cfg.train.trajectory_len,
        }
        self.job_queue.put(job)
        return True

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

    def deal_with_get_all_manager(self):
        return self.manager_record

    def deal_with_get_all_learner(self):
        return self.learner_record

    def deal_with_get_all_job(self):
        return self.job_record

    def deal_with_get_replay_buffer(self):
        return self.replay_buffer

    def deal_with_get_job_queue(self):
        pass

    def deal_with_push_data_to_replay_buffer(self):
        self.replay_buffer.push_data({
                'job_id': 'job_id',
                'trajectory_path': 'trajectory_path',
                'learner_uid': 'learner_uid',
                'data': [[1, 2, 3], [4, 5, 6]]
            })
        self.replay_buffer.push_data({
                'job_id': 'job_id',
                'trajectory_path': 'trajectory_path',
                'learner_uid': 'learner_uid',
                'data': [[1, 2, 3], [4, 5, 6]]
            })
        return True

