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

from sc2learner.data.online import ReplayBuffer


class Coordinator(object):
    def __init__(self, cfg):
        super(Coordinator, self).__init__()
        self.cfg = cfg

        self.learner_port = cfg['api']['learner_port']
        self.manager_ip = cfg['api']['manager_ip']
        self.manager_port = cfg['api']['manager_port']

        # {manager_uid: {actor_uid: [job_id]}}
        self.manager_record = {}  
        # {job_id: {content: info, state: run/finish}}
        self.job_record = {}  
        # {learner_uid: {"learner_ip": learner_ip,
        #                "job_ids": [job_id], "models": [model_name]}}
        self.learner_record = {}  
        self.replay_buffer = ReplayBuffer(self.cfg)
        self.replay_buffer.run()

        self._set_logger()

    def _set_logger(self, level=1):
        self.logger = logging.getLogger("coordinator.log")

    def _get_job(self):
        '''
            Overview: return job info for actor
            Returns:
                - (:obj`dict`): job info
        '''
        job_id = str(uuid.uuid1())
        learner_uid1 = random.choice(list(self.learner_record.keys()))
        learner_uid2 = random.choice(list(self.learner_record.keys()))
        return {'job_id': job_id, 'learner_uid1': learner_uid1, 'learner_uid2': learner_uid2}

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
            self.learner_record[learner_uid] = {
                    "learner_ip": learner_ip, 
                    "job_ids": [], 
                    "models": []
                }
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
        learner_uid = job['learner_uid']
        if manager_uid not in self.manager_record:
            self.deal_with_register_manager(manager_uid)
        if actor_uid not in self.manager_record[manager_uid]:
            self.manager_record[manager_uid][actor_uid] = []
        self.manager_record[manager_uid][actor_uid].append(job_id)
        self.job_record[job_id] = {
                'content': job, 
                'state': 'running'
            }
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
