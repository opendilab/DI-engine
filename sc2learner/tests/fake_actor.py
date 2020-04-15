import torch
import sys
import os
import time
import uuid
import requests
import threading
import numpy as np
import traceback
from itertools import count
import logging
import argparse
import yaml

from sc2learner.utils.log_helper import TextLogger
from sc2learner.utils import read_file_ceph, save_file_ceph

parser = argparse.ArgumentParser(description='implementation of NAS worker')
parser.add_argument('--config', type=str, help='training config yaml file')


class FakeActor(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.manager_ip = cfg['system']['manager_ip']
        self.manager_port = cfg['system']['manager_port']
        self.actor_uid = os.environ.get('SLURM_JOB_ID', str(uuid.uuid1()))
        self.ceph_path = self.cfg['ceph_path']
        self.heartbeats_thread = cfg['actor']['heartbeats_thread']

        self.log_save_dir = os.path.join(self.cfg['log_save_dir'], 'actor')
        self.url_prefix = 'http://{}:{}/'.format(self.manager_ip, self.manager_port)
        self.heartbeats_freq = 60 * 2
        self.stop_flag = False

        self._set_logger()
        self.register_actor()

        # start sending heartbeats thread
        if self.heartbeats_thread:
            check_send_actor_heartbeats_thread = threading.Thread(target=self.send_actor_heartbeats)
            check_send_actor_heartbeats_thread.start()
            self.logger.info("[UP] send actor heartbeats thread ")

    def _set_logger(self):
        self.log_name = self.actor_uid + ".log"
        self.logger = TextLogger(self.log_save_dir, name=self.log_name)

    def send_actor_heartbeats(self):
        '''
            Overview: actor sends heartbeats to manager
        '''
        while not self.stop_flag:
            d = {'actor_uid': self.actor_uid}
            try:
                response = requests.post(self.url_prefix + 'manager/get_heartbeats', json=d).json()
                for _ in range(self.heartbeats_freq):
                    if not self.stop_flag:
                        time.sleep(1)
                    else:
                        break
                if self.stop_flag:
                    break
            except Exception as e:
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(1)
        self.logger.info('check_send_actor_heartbeats_thread stop as job finished.')

    def register_actor(self):
        '''
            Overview: register actor to manager.
        '''
        while True:
            try:
                d = {'actor_uid': self.actor_uid}
                response = requests.post(self.url_prefix + 'manager/register', json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def ask_for_job(self):
        '''
            Overview: ask manager for a job
        '''
        try:
            while True:
                d = {'actor_uid': self.actor_uid}
                response = requests.post(self.url_prefix + 'manager/ask_for_job', json=d).json()
                if response['code'] == 0 and response['info'] != '':
                    self.logger.info('[ask_for_job] {}'.format(response['info']))
                    self.job = response['info']
                    if self.job:
                        break
                time.sleep(2)
        except Exception as e:
            self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
            self.logger.info('[error][ask_for_job] {}'.format(sys.exc_info()[0]))
        else:
            self.logger.info('[actor {}] get job {} success'.format(self.actor_uid, self.job['job_id']))
            self.job_id = self.job['job_id']

    def _get_model(self, model_name):
        model = torch.load(read_file_ceph(self.ceph_path + model_name))
        return model

    def _get_feedback(self):
        trajectory = {
            'step': 1,
            'agent_no': 1,
            'job_id': self.job['job_id'],
            'job': self.job,
            'prev_obs': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'lstm_state_before': True,
            'lstm_state_after': False,
            'logits': 'logits_value',
            'action': 'action_value',
            'next_obs': 1,
            'done': 1,
            'rewards': 1,
            'info': 1,
            'have_teacher': 1,
            'teacher_lstm_state_before': 1,
            'teacher_lstm_state_after': 1,
            'teacher_logits': 1
        }
        trajectory = [trajectory] * 1
        ceph_name = "job_{}_{}.traj".format(self.job_id, str(uuid.uuid1()))
        t1 = time.time()
        save_file_ceph(self.ceph_path, ceph_name, trajectory)
        self.logger.info("save to ceph cost {} seconds. ".format(time.time() - t1))
        metadata = {
            'job_id': self.job_id,
            'trajectory_path': ceph_name,
            'learner_uid': self.job['learner_uid'],
            'data': torch.tensor([[1, 2, 3], [4, 5, 6]]).tolist(),
            'step_data_compressor': 'none'
        }
        return trajectory, metadata

    def simulate(self):
        try:
            # only use one model
            model = self._get_model(self.job['model_id'][0])
            stat = self._get_model(self.job['stat_id'][0])
            self.logger.info(
                'actor {} simulate job {}, using model ({}) & stat ({})'.format(
                    self.actor_uid, self.job_id, self.job['model_id'][0], self.job['stat_id'][0]
                )
            )
            trajectory, metadata = self._get_feedback()
        except Exception as e:
            self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
            self.logger.info("[error] {}".format(sys.exc_info()))
            return False, {}, {}
        return True, trajectory, metadata

    def send_metadata(self, metadata):
        d = {'actor_uid': self.actor_uid, 'job_id': self.job_id, 'metadata': metadata}
        response = requests.post(self.url_prefix + 'manager/get_metadata', json=d).json()
        if response['code'] == 0:
            self.logger.info("succeed sending result: {}".format(self.job_id))
        else:
            self.logger.info("failed to send result: {}".format(self.job_id))

    def finish_job(self):
        d = {'actor_uid': self.actor_uid, 'job_id': self.job_id}
        response = requests.post(self.url_prefix + 'manager/finish_job', json=d).json()
        if response['code'] == 0:
            self.logger.info("succeed finishing job: {}".format(self.job_id))
        else:
            self.logger.info("failed to finish job: {}".format(self.job_id))

    def stop(self):
        self.stop_flag = True


def main():
    args = parser.parse_args()
    # experiments/xxx/api-log/learner-api.log
    log_path = os.path.dirname(args.config)
    api_dir_name = 'api-log'
    log_path = os.path.join(log_path, api_dir_name)

    cfg = yaml.load(open(args.config, 'r'))
    cfg['log_save_dir'] = log_path
    api_info = cfg['system']
    learner_port = api_info['learner_port']
    coordinator_ip = api_info['coordinator_ip']
    coordinator_port = api_info['coordinator_port']
    manager_ip = api_info['manager_ip']
    manager_port = api_info['manager_port']

    actor = FakeActor(cfg)

    actor.ask_for_job()
    state, trajectory, metadata = actor.simulate()
    if state:
        actor.send_metadata(metadata)
    actor.finish_job()
    actor.stop()
    # exit()


if __name__ == '__main__':
    main()
