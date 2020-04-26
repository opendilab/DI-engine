import os
import sys
import torch
import numpy as np
import threading
import logging
import yaml
import traceback
import uuid
import time
import requests
from collections import OrderedDict
import random

from sc2learner.utils import read_file_ceph, save_file_ceph, get_step_data_decompressor, get_manager_node_ip
from sc2learner.utils import dist_init, broadcast


class LearnerCommunicationHelper(object):
    def __init__(self, cfg):
        super(LearnerCommunicationHelper, self).__init__()
        self.cfg = cfg

        # if want to rerun learner, learner_uid should be set in cfg.system
        if 'learner_uid' in self.cfg.system.keys() and self.cfg.system.learner_uid:
            self.learner_uid = str(self.cfg.system.learner_uid)
        else:
            self.learner_uid = str(os.environ.get('SLURM_JOB_ID', str(random.randint(0, 1000000))))

        # if want to rerun learner, learner_re_register should be set True in cfg.system
        if 'learner_re_register' in self.cfg.system.keys():
            self.learner_re_register = self.cfg.system.learner_re_register
        else:
            self.learner_re_register = False

        self.use_distributed = cfg.train.use_distributed
        if self.use_distributed:
            self.rank, self.world_size = dist_init()
            self.data_index = torch.tensor(0)
        else:
            self.rank, self.world_size = 0, 1

        if 'learner_ip' in self.cfg.system.keys():
            self.learner_ip = self.cfg.system.learner_ip
        else:
            self.learner_ip = os.environ.get('SLURMD_NODENAME', '')  # hostname like SH-IDC1-10-5-36-236
        if not self.learner_ip:
            raise ValueError('learner_ip must be ip address, but found {}'.format(self.learner_ip))

        self.learner_port = self.cfg.system.learner_port - int(self.rank)
        self.coordinator_ip = self.cfg.system.coordinator_ip
        self.coordinator_port = self.cfg.system.coordinator_port
        self.ceph_traj_path = self.cfg.system.ceph_traj_path
        self.ceph_model_path = self.cfg.system.ceph_model_path
        self.use_ceph = self.cfg.system.use_ceph
        self.learner_heartbeats_thread = self.cfg.system.learner_heartbeats_thread

        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)

        self._setup_comm_logger()
        self._register_learner_in_coordinator()

        self.batch_size = cfg.data.train.batch_size  # once len(self.trajectory_record) reach this value, train

        self.trajectory_record = OrderedDict()  # {trajectory_name: trajectory}
        self.delete_trajectory_record = False  # if delete trajectory_record
        self.delete_trajectory_record_num = 1000  # if larger than len(self.trajectory_record), delete by time sort

        # heartbeats
        self.stop_flag = False
        self.heartbeats_freq = self.cfg.system.learner_heartbeats_freq

        if self.learner_heartbeats_thread:
            check_send_learner_heartbeats_thread = threading.Thread(target=self.send_learner_heartbeats)
            check_send_learner_heartbeats_thread.setDaemon(True)
            check_send_learner_heartbeats_thread.start()
            self.comm_logger.info("[UP] send learner heartbeats thread ")

    def _setup_comm_logger(self):
        self.comm_logger = logging.getLogger("learner.log")
        self.comm_logger.info('learner_uid {}, learner_ip {}'.format(self.learner_uid, self.learner_ip))

    def get_ip_port(self):
        return self.learner_ip, self.learner_port

    def send_learner_heartbeats(self):
        '''
            Overview: actor sends heartbeats to coordinator
        '''
        d = {'learner_uid': self.learner_uid}
        while not self.stop_flag:
            try:
                response = requests.post(self.url_prefix + 'coordinator/get_heartbeats', json=d).json()
                for _ in range(self.heartbeats_freq):
                    if not self.stop_flag:
                        time.sleep(1)
                    else:
                        break
                if self.stop_flag:
                    break
            except Exception as e:
                self.comm_logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(1)
        self.comm_logger.info('check_send_learner_heartbeats_thread stop as job finished.')

    def _register_learner_in_coordinator(self):
        '''
            Overview: register learner in coordinator with learner_uid and learner_ip
        '''
        d = {
            'learner_uid': self.learner_uid,
            'learner_ip': self.learner_ip,
            'learner_port': self.learner_port,
            'learner_re_register': self.learner_re_register
        }
        while True:
            try:
                response = requests.post(self.url_prefix + "coordinator/register_learner", json=d).json()
                if response['code'] == 0:
                    # self.checkpoint_path = response['info']  # without s3://{}/ , only file name
                    return True
                else:
                    self.comm_logger.info(response['info'])
            except Exception as e:
                self.comm_logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def register_model_in_coordinator(self, model_name):
        '''
            Overview: register model in coordinator with model_name, should be called after saving model in ceph
        '''
        while True:
            try:
                d = {'learner_uid': self.learner_uid, 'model_name': model_name}
                response = requests.post(self.url_prefix + "coordinator/register_model", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.comm_logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def save_model_to_ceph(self, model_name, model):
        save_file_ceph(self.ceph_model_path, model_name, model)
        self.comm_logger.info("save model {} to ceph".format(model_name))

    def load_trajectory(self, metadata):
        '''
            Overview: load trajectory from ceph or lustre
            Returns:
                - (:obj`dict`): trajectory
        '''
        assert (isinstance(metadata, dict))
        decompressor = get_step_data_decompressor(metadata['step_data_compressor'])
        data = self._read_file(metadata['trajectory_path'])
        data = decompressor(data)
        for k, v in metadata.items():
            for i in range(len(data)):
                data[i][k] = v
        file_system = 'ceph' if self.use_ceph else 'lustre'
        self.comm_logger.info("load trajectory {} from {}".format(metadata['trajectory_path'], file_system))
        return data

    def _read_file(self, path):
        if self.use_ceph:
            ceph_traj_path = self.ceph_traj_path + path
            return read_file_ceph(ceph_traj_path, read_type='pickle')
        else:
            return path

    def deal_with_get_metadata(self, metadatas):
        assert (isinstance(metadatas, list))
        job_id = metadatas['job_id']
        self.data_queue.put(metadatas)

    def _get_trajectory_list(self, metadatas):
        '''
            Overview: get list of trajectory by trajectory_path in metadatas. Using OrderedDict,
                      and move_to_end() will set key to top to avoid being deleted.
            Arguments:
                - metadatas (:obj:`dict`): list of actor's metadata
            Returns:
                - (:obj`list`): list of trajectory
        '''
        train_trajectory_list = []
        for metadata in metadatas:
            job_id = metadata['job_id']
            trajectory_path = metadata['trajectory_path']
            if trajectory_path in self.trajectory_record:
                trajectory = self.trajectory_record[trajectory_path]
            else:
                trajectory = self.load_trajectory(trajectory_path)
                self.trajectory_record[trajectory_path] = trajectory
            self.trajectory_record.move_to_end(trajectory_path, last=False)  # avoid being deleted
            train_trajectory_list.append(trajectory)
        return train_trajectory_list

    def _delete_trajectory_record(self):
        '''
            Overview: delete trajectory from trajectory_record by time sort
            Arguments:
                - metadatas (:obj:`dict`): list of actor's metadata
            Returns:
                - (:obj`list`): list of trajectory
        '''
        if self.delete_trajectory_record and len(self.trajectory_record) > self.delete_trajectory_record_num:
            self.comm_logger.info("delete trajectory_record ... ")
            for k in list(self.trajectory_record.keys)[self.delete_trajectory_record_num:]:
                del self.trajectory_record[k]

    def update_info(self, info):
        """
            Overview: send some info to coordinator and update coordinator
            Arguments:
                - info (:obj:`dict`): info dict
        """
        try:
            d = {'update_info': info, 'learner_uid': self.learner_uid}
            response = requests.post(self.url_prefix + "coordinator/update_replay_buffer", json=d).json()
            if response['code'] == 0:
                return True
        except Exception as e:
            self.comm_logger.info("something wrong with coordinator, {}".format(e))

    def _get_sample_data(self):
        while True:
            try:
                if self.use_distributed:
                    broadcast(self.data_index, 0)
                data_index = self.data_index.item()
                d = {'learner_uid': self.learner_uid, 'batch_size': self.batch_size, 'data_index': data_index}
                response = requests.post(self.url_prefix + 'coordinator/ask_for_metadata', json=d).json()
                self.data_index += 1
                if response['code'] == 0:
                    metadatas = response['info']
                    if metadatas is not None:
                        yield metadatas
                        continue  # not sleep
            except Exception as e:
                self.comm_logger.info(''.join(traceback.format_tb(e.__traceback__)))
                self.comm_logger.info("[error] {}".format(sys.exc_info()))
            time.sleep(3)

    def sample(self):
        d = {'learner_uid': self.learner_uid, 'batch_size': self.batch_size}
        try:
            response = requests.post(self.url_prefix + 'coordinator/ask_for_metadata', json=d).json()
            if response['code'] == 0:
                metadatas = response['info']
                if metadatas is not None:
                    return metadatas
        except Exception as e:
            self.comm_logger.info(''.join(traceback.format_tb(e.__traceback__)))
            self.comm_logger.info("[error] {}".format(sys.exc_info()))
        return None

    def _load_checkpoint_from_ceph(self, checkpoint_path, read_type='pickle'):
        '''
            Overview: load model from ceph, using pickle as default
            Returns:
                - (:obj`dict`): model
        '''
        checkpoint = read_file_ceph(self.ceph_path + checkpoint_path, read_type=read_type)
        self.comm_logger.info("load checkpoint {} from ceph".format(checkpoint_path))
        return checkpoint

    def deal_with_reset(self, checkpoint_path):
        model = self._load_checkpoint_from_ceph(checkpoint_path)
        return True

    @property
    def data_iterator(self):
        return iter(self._get_sample_data())

    def deal_with_get_trajectory_record_len(self):
        return len(self.trajectory_record)

    def deal_with_set_delete_trajectory_record(self, state, num):
        if state:
            self.delete_trajectory_record = True
            self.delete_trajectory_record_num = int(num)
        else:
            self.delete_trajectory_record = False
        return True

    def deal_with_set_train_freq(self, freq):
        self.train_freq = int(freq)
        return True
