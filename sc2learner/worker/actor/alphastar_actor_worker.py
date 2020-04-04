import torch
import sys
import os
import time
import uuid
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import numpy as np
import traceback
from itertools import count
import logging
import argparse
import yaml

from sc2learner.utils.log_helper import TextLogger
from sc2learner.torch_utils import build_checkpoint_helper
from sc2learner.utils import read_file_ceph, save_file_ceph, get_manager_node_ip, merge_two_dicts
from .alphastar_actor import AlphaStarActor

parser = argparse.ArgumentParser(description='AlphaStar Actor (for training on slurm cluster)')
parser.add_argument('--config', type=str, help='training config yaml file')


class AlphaStarActorWorker(AlphaStarActor):
    def __init__(self, cfg):
        super(AlphaStarActorWorker, self).__init__(cfg)
        self.stop_flag = False
        self._module_init()
        self.job_getter.register_actor()
        if self.cfg.actor.get('heartbeats_thread', True):
            self.heartbeat_worker.start_heartbeats_thread()

    def _module_init(self):
        self.context = ActorContext(self.cfg, self.actor_uid)  # actor_uid is from super
        self.job_getter = JobGetter(self.context)
        self.model_loader = ModelLoader(self.context)
        self.stat_requester = StatRequester(self.context)
        self.data_pusher = DataPusher(self.context)
        self.heartbeat_worker = HeartBeatWorker(self.context)


class ActorContext:
    def __init__(self, cfg, actor_uid):
        self.cfg = cfg
        if cfg['actor']['manager_ip'].lower() == 'auto':
            self.manager_ip = get_manager_node_ip()
        else:
            self.manager_ip = cfg['actor']['manager_ip']
        self.actor_uid = actor_uid
        self.manager_port = cfg['actor']['manager_port']
        self.url_prefix = 'http://{}:{}/'.format(self.manager_ip, self.manager_port)
        self._set_logger()
        self._set_req_session()
    
    def _set_logger(self):
        self.log_name = 'actor' + self.actor_uid + ".log"
        self.logger = logging.getLogger(self.log_name)

    def _set_req_session(self):
        self.requests_session = requests.session()
        retries = Retry(total=self.cfg.actor.get('max_connection_retries', 20), backoff_factor=1)
        h = HTTPAdapter(max_retries=retries)
        self.requests_session.mount('http://', h)


class HeartBeatWorker:
    def __init__(self, context):
        self.context = context
        self.stop_flag = False

    def start_heartbeats_thread(self):
        # start sending heartbeats thread
        check_send_actor_heartbeats_thread = threading.Thread(target=self.send_actor_heartbeats)
        check_send_actor_heartbeats_thread.daemon = True
        check_send_actor_heartbeats_thread.start()
        self.context.logger.info("[UP] send actor heartbeats thread")

    def start_heartbeat(self):
        self.stop_flag = False

    def stop_heatbeat(self):
        self.stop_flag = True

    def send_actor_heartbeats(self):
        '''
            Overview: actor sends heartbeats to manager
        '''
        while not self.stop_flag:
            d = {'actor_uid': self.context.actor_uid}
            try:
                response = self.context.requests_session.post(
                           self.context.url_prefix + 'manager/get_heartbeats', json=d).json()
                if response['code'] != 0:
                    self.context.logger.warn('send heartbeat failed, response={}'.format(response))
                else:
                    self.context.logger.info('check_send_actor_heartbeats_thread stop as job finished.')
                for _ in range(self.context.cfg["actor"]["heartbeats_freq"]):
                    if not self.stop_flag:
                        time.sleep(1)
                    else:
                        break
            except Exception as e:
                self.context.logger.error(''.join(traceback.format_tb(e.__traceback__)))
                self.context.logger.error("[error] {}".format(sys.exc_info()))


class JobGetter:
    def __init__(self, context):
        self.context = context
        self.job_request_id = 0

    def register_actor(self):
        '''
            Overview: register actor to manager.
        '''
        while True:
            try:
                d = {'actor_uid': self.context.actor_uid}
                response = self.context.requests_session.post(
                           self.context.url_prefix + 'manager/register', json=d).json()
                if response['code'] == 0:
                    self.context.logger.info('actor register succeed')
                    return
            except Exception as e:
                self.context.logger.error(''.join(traceback.format_tb(e.__traceback__)))
                self.context.logger.error("[error] {}".format(sys.exc_info()))

    def get_job(self, actor_uid):
        """
        Overview: asking for a job from the coordinator
        Input:
            - actor_uid
        Output:
            - job: a dict with description of how the game should be
        """
        try:
            while True:
                d = {'request_id': self.job_request_id, 'actor_uid': actor_uid}
                response = self.context.requests_session.post(
                           self.context.url_prefix + 'manager/ask_for_job', json=d).json()
                if response['code'] == 0 and response['info'] != '':
                    self.context.logger.info('[ask_for_job] {}'.format(response['info']))
                    self.job_request_id += 1
                    return response['info']
                time.sleep(2)
        except Exception as e:
            self.context.logger.info(''.join(traceback.format_tb(e.__traceback__)))
            self.context.logger.info('[error][ask_for_job] {}'.format(sys.exc_info()[0]))


class DataPusher:
    def __init__(self, context):
        self.context = context
        self.ceph_path = self.context.cfg['actor']['ceph_traj_path']

    def push(self, metadata, data_buffer):
        job = metadata['job']
        job_id = job['job_id']
        agent_no = metadata['agent_no']
        model_id = job['model_id'][agent_no]
        # saving to ceph
        ceph_name = "model_{}_job_{}_agent_{}_step_{}_{}.traj".format(
            model_id, job_id, agent_no, metadata['game_step'], str(uuid.uuid1())
        )
        save_file_ceph(self.ceph_path, ceph_name, data_buffer)

        # sending report to manager
        metadata_supp = {
            'ceph_path': self.ceph_path,
            'ceph_name': ceph_name,
            'this_traj_learner_uid': [job.get('learner_uid1'), job.get('learner_uid2')][agent_no],
            'learner_uid1': job.get('learner_uid1'),
            'learner_uid2': job.get('learner_uid2')
        }
        metadata = merge_two_dicts(metadata, metadata_supp)
        d = {'actor_uid': self.context.actor_uid, 'job_id': job_id, 'metadata': metadata}
        try:
            response = self.context.requests_session.post(
                       self.context.url_prefix + 'manager/get_metadata', json=d).json()
            if response['code'] == 0:
                self.context.logger.info("succeed sending result: {}".format(job_id))
            else:
                self.context.logger.warn("failed to send result: {}".format(job_id))
        except Exception as e:
            self.context.logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self.context.logger.error("[error] {}".format(sys.exc_info()))


class ModelLoader:
    def __init__(self, context):
        self.context = context
        self.ceph_path = self.context.cfg['actor']['ceph_model_path']

    def _get_model_ceph_path(self, model_id, teacher=False):
        return self.ceph_path + model_id

    def load_model(self, job, agent_no, model):
        """
        Overview: fetch a model from ceph
        Input:
            - job: a dict with description of how the game should be
            - agent_no: 0 or 1, labeling the two agents of game
            - model: the model in agent, should be modified here
        """
        path = self._get_model_ceph_path(job['model_id'][agent_no])
        t = time.time()
        # during testing, we don't necessary load the real weights
        if 'do_not_load' not in path:
            model_handle = read_file_ceph(path)
            helper = build_checkpoint_helper('')
            helper.load(model_handle, model, prefix_op='remove', prefix='module.')
        self.context.logger.info('Model {} loaded for agent {}, time:{}'.format(path, agent_no, time.time() - t))

    def load_teacher_model(self, job, model):
        if 'teacher_model_id' not in job.keys():
            self.context.logger.info('Teacher SL model eval on actor is disabled, not loading')
            return
        path = self._get_model_ceph_path(job['teacher_model_id'], teacher=True)
        t = time.time()
        # during testing, we don't necessary load the real weights
        if 'do_not_load' not in path:
            model_handle = read_file_ceph(path)
            helper = build_checkpoint_helper('')
            helper.load(model_handle, model, prefix_op='remove', prefix='module.')
        self.context.logger.info('Teacher Model {} loaded, time:{}'.format(path, time.time() - t))


class StatRequester:
    def __init__(self, context):
        self.context = context
        self.ceph_path = self.context.cfg['actor']['ceph_stat_path']

    def _get_stat_ceph_path(self, stat_id):
        return self.ceph_path + stat_id

    def request_stat(self, job, agent_no):
        path = self._get_stat_ceph_path(job['stat_id'][agent_no])
        if 'do_not_load' in path:
            return {}
        ceph_handle = read_file_ceph(path)
        stat = torch.load(ceph_handle, map_location='cpu')
        return stat


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    actor = AlphaStarActorWorker(cfg)
    actor.run()


if __name__ == '__main__':
    main()
