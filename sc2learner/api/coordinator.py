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

from sc2learner.data.online import ReplayBuffer


class Coordinator(object):
    def __init__(self, cfg):
        super(Coordinator, self).__init__()
        self.cfg = cfg

        self.learner_port = cfg['api']['learner_port']
        self.manager_ip = cfg['api']['manager_ip']
        self.manager_port = cfg['api']['manager_port']

        self.use_fake_data = cfg['api']['coordinator']['use_fake_data']
        if self.use_fake_data:
            self.fake_model_path = cfg['api']['coordinator']['fake_model_path']
            self.fake_stat_path = cfg['api']['coordinator']['fake_stat_path']

        # {manager_uid: {actor_uid: [job_id]}}
        self.manager_record = {}
        # {job_id: {content: info, state: running/finish}}
        self.job_record = {}
        # {learner_uid: {"learner_ip": learner_ip,
        #                "job_ids": [job_id], "models": [model_name]}}
        self.learner_record = {}
        self.replay_buffer = ReplayBuffer(EasyDict(self.cfg['replay_buffer']))
        self.replay_buffer.run()

        self._set_logger()

    def close(self):
        self.replay_buffer.close()

    def _set_logger(self, level=1):
        self.logger = logging.getLogger("coordinator.log")

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

    def deal_with_finish_job(self, manager_uid, actor_uid, job_id):
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
                - update_info (:obj:`str`): learner's uid
            Returns:
                - (:obj`list`): metadata list
        '''
        self.replay_buffer.update(update_info)
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
