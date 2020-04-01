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
import subprocess

from utils.log_helper import TextLogger


class Manager(object):
    def __init__(self, cfg):
        super(Manager, self).__init__()

        self.cfg = cfg

        self.coordinator_ip = cfg['api']['coordinator_ip']
        self.coordinator_port = cfg['api']['coordinator_port']
        self.manager_ip = cfg['api']['manager_ip']
        self.manager_uid = self.manager_ip

        # to attach coordinator
        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)
        self.check_dead_actor_freq = 120

        # auto run actor
        self.auto_run_actor = cfg['api']['manager']['auto_run_actor']
        if self.auto_run_actor:
            self.use_partitions = cfg['api']['manager']['use_partitions']
            self.actor_num = cfg['api']['manager']['actor_num']

        self.actor_record = {
        }  # {actor_uid: {"job_ids": [job_id], "last_beats_time": last_beats_time, "state": 'alive'/'dead'}}}
        self.job_record = {}  # {job_id: {content: info, metadatas: [metadata], state: running/finish/dead}}
        self.reuse_job_list = []

        self._set_logger()
        self.register_manager_in_coordinator()

        # thread to check actor if dead
        check_actor_dead_thread = threading.Thread(target=self.check_actor_dead)
        check_actor_dead_thread.start()
        self.logger.info("[UP] check actor dead thread ")

        # launch actor 
        check_run_actor_thread = threading.Thread(target=self.check_run_actor)
        check_run_actor_thread.start()
        self.logger.info("[UP] check run actor thread ")

    def _set_logger(self):
        self.logger = logging.getLogger("manager.log")

    def register_manager_in_coordinator(self):
        '''
            Overview: register manager to coordinator.
        '''
        while True:
            try:
                d = {'manager_uid': self.manager_uid}
                response = requests.post(self.url_prefix + "coordinator/register_manager", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def deal_with_register_actor(self, actor_uid):
        '''
            Overview: when receiving actor's request of register, use this function to deal with this request.
        '''
        assert actor_uid not in self.actor_record
        last_beats_time = int(time.time())
        self.actor_record[actor_uid] = {'job_ids': [], 'last_beats_time': last_beats_time, 'state': 'alive'}
        return True

    def _add_job_to_record(self, actor_uid, job):
        job_id = job['job_id']
        self.job_record[job_id] = {'content': job, 'metadatas': [], 'state': 'running'}
        self.actor_record[actor_uid]['job_ids'].append(job_id)

    def deal_with_ask_for_job(self, actor_uid):
        '''
            Overview: when receiving actor's request of asking for job, ,return job
            Arguments:
                - actor_uid (:obj:`str`): actor's uid
            Returns:
                - (:obj`dict`): job info
        '''
        if actor_uid not in self.actor_record:
            self.deal_with_register_actor(actor_uid)

        # refresh actor's last_beats_time
        self.actor_record[actor_uid]['last_beats_time'] = time.time()

        # reuse job
        if len(self.reuse_job_list) > 0:
            job_id = self.reuse_job_list[0]
            job = self.job_record[job_id]['content']
            self.job_record[job_id]['state'] = 'running'
            del self.reuse_job_list[0]
            self.actor_record[actor_uid]['job_ids'].append(job_id)
            return job
        else:
            d = {"manager_uid": self.manager_uid, "actor_uid": actor_uid}
            while True:
                try:
                    response = requests.post(self.url_prefix + 'coordinator/ask_for_job', json=d).json()
                    if response['code'] == 0:
                        job = response['info']
                        self._add_job_to_record(actor_uid, job)
                        return job
                except Exception as e:
                    self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
                    self.logger.info("[error] {}".format(sys.exc_info()))
                time.sleep(5)

    def deal_with_get_metadata(self, actor_uid, job_id, metadata):
        '''
            Overview: when receiving actor's request of sending metadata, ,return True/False
            Arguments:
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
                - metadata (:obj:`dict`): actor's metadata
            Returns:
                - (:obj`bool`): state
        '''
        assert actor_uid in self.actor_record, 'actor_uid ({}) not in actor_record'.format(actor_uid)
        # refresh actor's last_beats_time
        self.actor_record[actor_uid]['last_beats_time'] = time.time()
        self.job_record[job_id]['metadatas'].append(metadata)
        d = {"manager_uid": self.manager_uid, "actor_uid": actor_uid, "job_id": job_id, "metadata": metadata}
        while True:
            try:
                response = requests.post(self.url_prefix + 'coordinator/get_metadata', json=d).json()
                if response['code'] == 0:
                    job = response['info']
                    return True
                else:
                    self.logger.info("[manager - deal_with_get_metadata] response = ".format(response))
            except Exception as e:
                self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
                self.logger.info("[error] {}".format(sys.exc_info()))
            time.sleep(1)

    def deal_with_finish_job(self, actor_uid, job_id):
        '''
            Overview: when receiving actor's request of finishing job, ,return True/False
            Arguments:
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
            Returns:
                - (:obj`bool`): state
        '''
        assert actor_uid in self.actor_record, 'actor_uid ({}) not in actor_record'.format(actor_uid)
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        self.job_record[job_id]['state'] = 'finish'
        d = {"manager_uid": self.manager_uid, "actor_uid": actor_uid, "job_id": job_id}
        while True:
            try:
                response = requests.post(self.url_prefix + 'coordinator/finish_job', json=d).json()
                if response['code'] == 0:
                    job = response['info']
                    return True
            except Exception as e:
                self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
                self.logger.info("[error] {}".format(sys.exc_info()))
            time.sleep(1)

    def deal_with_get_heartbeats(self, actor_uid):
        '''
            Overview: when receiving worker's heartbeats, update last_beats_time.
            Arguments:
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
                - metadata (:obj:`dict`): actor's metadata
            Returns:
                - (:obj`bool`): state
        '''
        assert actor_uid in self.actor_record
        self.actor_record[actor_uid]['last_beats_time'] = int(time.time())
        return True

    ###################################################################################
    #                                     threads                                     #
    ###################################################################################

    def time_format(self, time_item):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_item))

    def deal_with_dead_actor(self, actor_uid, reuse=True):
        self.actor_record[actor_uid]['state'] = 'dead'
        os.system('scancel ' + actor_uid)
        self.logger.info('[kill-dead] dead actor {} was killed'.format(actor_uid))
        job_ids = self.actor_record[actor_uid]['job_ids']
        if len(job_ids) > 0:
            to_process_job_id = job_ids[-1]
            if self.job_record[to_process_job_id]['state'] != 'finish':
                self.job_record[to_process_job_id]['state'] = 'dead'
                self.logger.info("[manager][check_actor_dead] modify {} to dead".format(to_process_job_id))
                if reuse:
                    self.reuse_job_list.append(to_process_job_id)
                    self.logger.info("[manager][check_actor_dead] add {} to reuse_job_list".format(to_process_job_id))

    def check_actor_dead(self):
        while True:
            nowtime = int(time.time())
            for actor_uid, actor_info in self.actor_record.items():
                if actor_info['state'
                              ] == 'alive' and nowtime - actor_info['last_beats_time'] > self.check_dead_actor_freq:
                    # dead actor
                    self.logger.info(
                        "[manager][check_actor_dead] {} is dead, last_beats_time = {}".format(
                            actor_uid, self.time_format(actor_info['last_beats_time'])
                        )
                    )
                    self.deal_with_dead_actor(actor_uid)
            time.sleep(self.check_dead_actor_freq)

    def launch_actor(self, partition):
        self.logger.info('[launch_actor] in {}'.format(partition))
        subprocess.Popen(['bash', 'run_actor.sh', partition, '&'])

    def check_run_actor(self):
        time.sleep(10)
        if self.auto_run_actor:
            for partition, num in zip(self.use_partitions, self.actor_num):
                for i in range(num):
                    self.launch_actor(partition)
                    self.logger.info('launch actor {} in {}'.format(i, partition))

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

    def deal_with_get_all_actor_from_debug(self):
        return self.actor_record

    def deal_with_get_all_job_from_debug(self):
        return self.job_record

    def deal_with_get_reuse_job_list_from_debug(self):
        return self.reuse_job_list

    def deal_with_clear_reuse_job_list_from_debug(self):
        self.reuse_job_list = []
        return True

    def deal_with_launch_actor(self, use_partitions, actor_num):
        for partition, num in zip(use_partitions, actor_num):
            for i in range(num):
                self.launch_actor(partition)
                self.logger.info('launch actor {} in {}'.format(i, partition))
        return True
