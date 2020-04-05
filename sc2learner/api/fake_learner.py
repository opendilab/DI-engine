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

from utils import read_file_ceph, save_file_ceph


class FakeLearner(object):
    def __init__(self, cfg):
        super(FakeLearner, self).__init__()

        self.cfg = cfg

        self.learner_uid = str(uuid.uuid1())
        self.learner_ip = os.environ.get('SLURMD_NODENAME', '')  # hostname like SH-IDC1-10-5-36-236
        if not self.learner_ip:
            raise ValueError('learner_ip must be ip address, but found {}'.format(self.learner_ip))
        self.coordinator_ip = self.cfg['api']['coordinator_ip']
        self.coordinator_port = self.cfg['api']['coordinator_port']
        self.ceph_path = self.cfg['api']['ceph_path']

        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)

        self._set_logger()
        self._register_learner_in_coordinator()

        self.batch_size = 4
        self.train_freq = 10  # ask for metadata and train every `train_freq` seconds

        self.trajectory_record = OrderedDict()  # {trajectory_name: trajectory}
        self.delete_trajectory_record = True  # if delete trajectory_record
        self.delete_trajectory_record_num = 1000  # if larger than len(self.trajectory_record), delete by time sort

        # threads
        check_train_thread = threading.Thread(target=self.check_train)
        check_train_thread.start()
        self.logger.info("[UP] check train thread ")

    def _set_logger(self):
        '''
            Overview: get logger, which is build by learner_api.py
        '''
        self.logger = logging.getLogger("learner.log")

    def _register_learner_in_coordinator(self):
        '''
            Overview: register learner in coordinator with learner_uid and learner_ip
        '''
        while True:
            try:
                d = {'learner_uid': self.learner_uid, 'learner_ip': self.learner_ip}
                response = requests.post(self.url_prefix + "coordinator/register_learner", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def _register_model_in_coordinator(self, model_name):
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
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def _save_model_to_ceph(self, model_name, model):
        '''
            Overview: save model in ceph, using pickle
        '''
        save_file_ceph(self.ceph_path, model_name, model)
        self.logger.info("save model {} to ceph".format(model_name))

    def _load_trajectory_from_ceph(self, trajectory_name):
        '''
            Overview: load trajectory from ceph, using pickle
            Returns:
                - (:obj`dict`): trajectory
        '''
        trajectory = read_file_ceph(self.ceph_path + trajectory_name, read_type='pickle')
        self.logger.info("load trajectory {} to ceph".format(trajectory_name))
        return trajectory

    def _get_trajectory_list(self, metadatas):
        '''
            Overview: get list of trajectory by trajectory_path in metadatas. Using OrderedDict, and move_to_end() will set key to top to avoid being deleted.
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
                trajectory = self._load_trajectory_from_ceph(trajectory_path)
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
            self.logger.info("delete trajectory_record ... ")
            for k in list(self.trajectory_record.keys)[self.delete_trajectory_record_num:]:
                del self.trajectory_record[k]

    def _get_update_info(self, metadatas):
        '''
            Overview: get update_info from metadatas
            Arguments:
                - metadatas (:obj:`dict`): list of actor's metadata
            Returns:
                - (:obj`dict`): update info
        '''
        update_info = {'replay_unique_id': [], 'replay_buffer_idx': [], 'priority': []}
        for metadata in metadatas:
            update_info['replay_unique_id'].append(metadata['replay_unique_id'])
            update_info['replay_buffer_idx'].append(metadata['replay_buffer_idx'])
            update_info['priority'].append(metadata['priority'])
        return update_info

    def train(self, metadatas):
        '''
            Overview: train process, get_trajectory -> train -> save model -> delete trajectory
            Arguments:
                - metadatas (:obj:`dict`): list of actor's metadata
        '''
        self.logger.info("getting trajectory data ... ")
        train_trajectory_list = self._get_trajectory_list(metadatas)
        self.logger.info("train_trajectory_list = {}".format(train_trajectory_list))
        self.logger.info("training ... ")
        model = {'model': torch.tensor([[1, 2, 3], [4, 5, 6]]), 'type': 'learner'}
        model_name = "learner-{}-{}.model".format(self.learner_uid, str(uuid.uuid1()))
        self.logger.info("saving model in ceph ... ")
        self._save_model_to_ceph(model_name, model)
        self._register_model_in_coordinator(model_name)
        self._delete_trajectory_record()
        update_info = self._get_update_info(metadatas)
        try:
            d = {'update_info': update_info}
            response = requests.post(self.url_prefix + "coordinator/update_replay_buffer", json=d).json()
            if response['code'] == 0:
                return True
        except Exception as e:
            self.logger.info("something wrong with coordinator, {}".format(e))

    ###################################################################################
    #                                     threads                                     #
    ###################################################################################

    def check_train(self):
        '''
            Overview: launch thread to check if train
        '''
        last_train_time = time.time()
        d = {'learner_uid': self.learner_uid, 'batch_size': self.batch_size}
        while True:
            now_time = time.time()
            metadatas = []
            if now_time - last_train_time > self.train_freq:
                try:
                    response = requests.post(self.url_prefix + 'coordinator/ask_for_metadata', json=d).json()
                    if response['code'] == 0:
                        metadatas = response['info']
                except Exception as e:
                    self.logger.info(''.join(traceback.format_tb(e.__traceback__)))
                    self.logger.info("[error] {}".format(sys.exc_info()))

                if metadatas:
                    self.train(metadatas)
                    last_train_time = time.time()
            time.sleep(self.train_freq)

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

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


if __name__ == '__main__':
    import subprocess

    def get_cls_info():
        ret_dict = {}
        info = subprocess.getoutput('sinfo -Nh').split('\n')
        for line in info:
            line = line.strip().split()
            if len(line) != 4:
                continue
            node, _, partition, state = line
            if partition not in ret_dict:
                ret_dict[partition] = []
            assert node not in ret_dict[partition]
            if state in ['idle', 'mix']:
                ret_dict[partition].append([node, state])
        return ret_dict

    def launch(partition, workstation):
        output = subprocess.getoutput(['bash', 'test.sh', partition, workstation])
        return output

    cls_dict = get_cls_info()
    for k, v in cls_dict.items():
        if k in ['VI_SP_Y_V100_B', 'VI_SP_Y_V100_A', 'VI_SP_VA_V100', 'VI_SP_VA_1080TI', 'VI_Face_1080TI',
                 'VI_ID_1080TI']:
            for vv in v:
                # output = launch(k, vv[0])
                # print(k, vv[0], output)
                print('srun -p {} -w {} python test.py'.format(k, vv[0]))
