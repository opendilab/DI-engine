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

from sc2learner.data.online import ReplayBuffer
from sc2learner.utils import read_file_ceph, save_file_ceph
from sc2learner.league import LeagueManager


class Coordinator(object):
    def __init__(self, cfg):
        super(Coordinator, self).__init__()
        self.cfg = cfg

        self.ceph_path = cfg['coordinator']['ceph_path']
        self.use_fake_data = cfg['coordinator']['use_fake_data']
        if self.use_fake_data:
            self.fake_model_path = cfg['coordinator']['fake_model_path']
            self.fake_stat_path = cfg['coordinator']['fake_stat_path']

        # {manager_uid: {actor_uid: [job_id]}}
        self.manager_record = {}
        # {job_id: {content: info, state: running/finish}}
        self.job_record = {}
        # {learner_uid: {"learner_ip": learner_ip,
        #                "job_ids": [job_id], "models": [model_name]}}
        self.learner_record = {}

        self.url_prefix_format = 'http://{}:{}/'

        self.replay_buffer = ReplayBuffer(EasyDict(self.cfg['replay_buffer']))
        self.replay_buffer.run()

        self._set_logger()

        # for league
        self.player_to_learner = {}
        self.learner_to_player = {}
        self.job_queue = Queue()
        self.league_flag = False
        self._init_league_manager()
        print('initialize league')
        self.logger.info('[logger] initialize league')

    def close(self):
        self.replay_buffer.close()

    def _set_logger(self, level=1):
        self.logger = logging.getLogger("coordinator.log")

    def _init_league_manager(self):
        def save_checkpoint_fn(src_checkpoint, dst_checkpoint, read_type='pickle'):
            '''
                Overview: copy src_checkpoint as dst_checkpoint
                Arguments:
                    - src_checkpoint (:obj:`str`): source checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
                    - dst_checkpoint (:obj:`str`): dst checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
            '''
            src_checkpoint = self.ceph_path + src_checkpoint
            dst_checkpoint = self.ceph_path + dst_checkpoint
            checkpoint = read_file_ceph(src_checkpoint, read_type=read_type)
            ceph_path, file_name = dst_checkpoint.strip().rsplit('/', 1)
            save_file_ceph(ceph_path, file_name, checkpoint)

        def load_checkpoint_fn(player_id, checkpoint_path):
            d = {'checkpoint_path': checkpoint_path}
            # need to be refine
            learner_uid = self.player_to_learner.get(player_id, random.choice(list(self.learner_record.keys())))
            while True:
                try:
                    response = requests.post(self.get_url_prefix(learner_uid) + "learner/reset", json=d).json()
                    if response['code'] == 0:
                        return True
                except Exception as e:
                    print("something wrong with coordinator, {}".format(e))
                time.sleep(10)
            return False

        def launch_match_fn(launch_info):
            home_id = launch_info['home_id']
            away_id = launch_info['away_id']
            home_checkpoint_path = launch_info['home_checkpoint_path']
            away_checkpoint_path = launch_info['away_checkpoint_path']
            home_learner_uid = home_id if home_id.endswith('_sl') else self.player_to_learner[home_id]
            away_learner_uid = away_id if away_id.endswith('_sl') else self.player_to_learner[away_id]
            job = {
                'job_id': str(uuid.uuid1()),
                'learner_uid': [home_learner_uid, away_learner_uid],
                'stat_id': ['fake_stat_path', 'fake_stat_path'],
                'game_type': 'league',
                'obs_compressor': 'lz4',
                'model_id': [home_checkpoint_path, away_checkpoint_path],
                'teacher_model_id': home_checkpoint_path,
                'map_name': 'AbyssalReef',
                'random_seed': 0,
                'home_race': 'zerg',
                'away_race': 'zerg',
                'difficulty': 'easy',
                'build': 'random',
                'data_push_length': 8
            }
            self.job_queue.put(job)

        self.league_manager = LeagueManager(self.cfg, save_checkpoint_fn, load_checkpoint_fn, launch_match_fn)
        self.player_ids = self.league_manager.active_players_ids
        print('{} learners should be registered totally. '.format(len(self.player_ids)))

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
                self.learner_record['test1'] = {"learner_ip": '0.0.0.0', "job_ids": [], "models": []}
                self.learner_record['test2'] = {"learner_ip": '0.0.0.0', "job_ids": [], "models": []}
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
            use_learner_uid_list = []
            for learner_uid in list(self.learner_record.keys()):
                if len(self.learner_record[learner_uid]['models']) > 0:
                    use_learner_uid_list.append(learner_uid)
            if use_learner_uid_list:
                learner_uid1 = random.choice(use_learner_uid_list)
                learner_uid2 = random.choice(use_learner_uid_list)
                model_name1 = self.learner_record[learner_uid1]['models'][-1]
                model_name2 = self.learner_record[learner_uid2]['models'][-1]
                ret = {
                    'job_id': job_id,
                    'learner_uid': [learner_uid1, learner_uid2],
                    'stat_id': [self.fake_stat_path, self.fake_stat_path],
                    'game_type': 'league',
                    'step_data_compressor': 'lz4',
                    'model_id': [model_name1, model_name2],
                    'teacher_model_id': model_name1,
                    'map_name': '',
                    'random_seed': 0,
                    'home_race': 'Zerg',
                    'away_race': 'Zerg',
                    'difficulty': 1,
                    'build': 0,
                    'data_push_length': 8
                }
            else:
                ret = {}
        return ret

    def deal_with_register_model(self, learner_uid, model_name):
        '''
            Overview: deal with register from learner to register model
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - model_name (:obj:`str`): model's name saved in ceph
        '''
        self.learner_record[learner_uid]['models'].append(model_name)
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

    def deal_with_register_learner(self, learner_uid, learner_ip):
        '''
            Overview: deal with register from learner
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
        '''
        if learner_uid not in self.learner_record:
            self.learner_record[learner_uid] = {"learner_ip": learner_ip, "job_ids": [], "models": []}
        if len(self.player_to_learner) == len(self.player_ids):
            self.logger.info('enough learners have been registered.')
            return False
        else:
            for player_id in self.player_ids:
                if player_id not in self.player_to_learner:
                    self.player_to_learner[player_id] = learner_uid
                    self.learner_to_player[learner_uid] = player_id
                    self.logger.info('leaner ({}) set to player ({})'.format(learner_uid, player_id))
                    break
        self.logger.info(
            '{}/{} learners have been registered'.format(len(self.player_to_learner), len(self.player_ids))
        )
        if len(self.player_ids) == len(self.player_to_learner) and not self.league_flag:
            self.league_flag = True
            self.league_manager.run()
            self.logger.info('league_manager run. ')
        return True

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
        home_learner_uid = self.job_record[job_id]['content']['learner_uid'][0]
        away_learner_uid = self.job_record[job_id]['content']['learner_uid'][1]
        home_id = home_learner_uid if home_learner_uid.endswith('_sl') else self.learner_to_player[home_learner_uid]
        away_id = away_learner_uid if away_learner_uid.endswith('_sl') else self.learner_to_player[away_learner_uid]
        match_info = {'home_id': home_id, 'away_id': away_id, 'result': result}
        self.league_manager.finish_match(match_info)
        return True

    def deal_with_ask_for_metadata(self, learner_uid, batch_size):
        '''
            Overview: when receiving learner's request of asking for metadata, return metadatas
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - batch_size (:obj:`int`): batch size
            Returns:
                - (:obj`list`): metadata list
        '''
        assert learner_uid in self.learner_record, 'learner_uid ({}) not in learner_record'.format(learner_uid)
        metadatas = self.replay_buffer.sample(batch_size)
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
        player_id = self.player_to_learner.get(learner_uid, random.choice(list(self.player_to_learner.values())))
        player_info = {'player_id': player_id, 'train_step': train_step}
        self.league_manager.update_active_player(player_info)
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
