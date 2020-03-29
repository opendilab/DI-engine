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


class Coordinator(object):
    def __init__(self, cfg):
        super(Coordinator, self).__init__()
        self.cfg = cfg

        self.learner_port = cfg['api']['learner_port']
        self.manager_ip = cfg['api']['manager_ip']
        self.manager_port = cfg['api']['manager_port']

        self.manager_record = {}  # {manager_uid: {actor_uid: [job_id]}}
        self.job_record = {}  # {job_id: {content: info, metadatas: [metadata], state: run/finish}}
        self.learner_record = {
        }  # {learner_uid: {"learner_ip": learner_ip, "job_ids": [job_id], "models": [model_name]}}

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
        self.job_record[job_id] = {'content': job, 'metadatas': [], 'state': 'running'}
        return job

    def deal_with_get_metadata(self, manager_uid, actor_uid, job_id, metadata):
        '''
            Overview: when receiving manager's request of sending metadata, ,return True/False
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
                - metadata (:obj:`dict`): actor's metadata
            Returns:
                - (:obj`bool`): state
        '''
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        self.job_record[job_id]['metadatas'].append(metadata)

        learner_uid_list = [
            self.job_record[job_id]['content']['learner_uid1'], self.job_record[job_id]['content']['learner_uid2']
        ]

        for learner_uid in learner_uid_list:
            learner_ip = self.learner_record[learner_uid]['learner_ip']
            url_prefix = 'http://{}:{}/'.format(learner_ip, self.learner_port)
            d = {"metadata": metadata}
            while True:
                try:
                    response = requests.post(url_prefix + 'learner/get_metadata', json=d).json()
                    if response['code'] == 0:
                        job = response['info']
                        return True
                except Exception as e:
                    self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
                    self.logger.info("[error] {}".format(sys.exc_info()))
                time.sleep(1)

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

    def deal_with_get_all_manager(self):
        return self.manager_record

    def deal_with_get_all_learner(self):
        return self.learner_record

    def deal_with_get_all_job(self):
        return self.job_record
