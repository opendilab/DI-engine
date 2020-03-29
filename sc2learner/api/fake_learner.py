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
import socket
import requests

from utils import read_file_ceph, save_file_ceph


class FakeLearner(object):
    def __init__(self, cfg):
        super(FakeLearner, self).__init__()

        self.cfg = cfg

        self.learner_uid = str(uuid.uuid1())
        self.learner_ip = socket.gethostname()  # hostname like SH-IDC1-10-5-36-236
        self.coordinator_ip = self.cfg['api']['coordinator_ip']
        self.coordinator_port = self.cfg['api']['coordinator_port']
        self.ceph_path = self.cfg['api']['ceph_path']
        self.trajectory_record = []

        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)

        self._set_logger()
        self.register_learner_in_coordinator()

        self.batch_size = 4  # once len(self.trajectory_record) reach this value, train

        # threads
        check_train_thread = threading.Thread(target=self.check_train)
        check_train_thread.start()
        self.logger.info("[UP] check train thread ")

    def _set_logger(self):
        self.logger = logging.getLogger("learner.log")

    def register_learner_in_coordinator(self):
        # register learner in coordinator
        while True:
            try:
                d = {'learner_uid': self.learner_uid, 'learner_ip': self.learner_ip}
                response = requests.post(self.url_prefix + "coordinator/register_learner", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def register_model_in_coordinator(self, model_nama):
        # register model in coordinator
        while True:
            try:
                d = {'learner_uid': self.learner_uid, 'model_name': model_name}
                response = requests.post(self.url_prefix + "coordinator/register_model", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def save_model_to_ceph(self, model_name, model):
        save_file_ceph(self.ceph_path, model_name, model)
        self.logger.info("save model {} to ceph".format(model_name))

    def load_trajectory_from_ceph(self, trajectory_name):
        trajectory = read_file_ceph(self.ceph_path + trajectory_name, read_type='pickle')
        self.logger.info("load trajectory {} to ceph".format(trajectory_name))
        return trajectory

    def deal_with_get_metadata(self, metadata):
        job_id = metadata['job_id']
        trajectory_path = metadata['trajectory_path']
        trajectory = self.load_trajectory_from_ceph(trajectory_path)
        self.trajectory_record.append(trajectory)

    def train(self, trajectory_list):
        self.logger.info("training ... ")
        model = {'model': torch.tensor([[1, 2, 3], [4, 5, 6]]), 'type': 'learner'}
        model_name = "learner-{}-{}".format(self.learner_uid, str(uuid.uuid1()))
        self.save_model_to_ceph(model_name, model)

    ###################################################################################
    #                                     threads                                     #
    ###################################################################################

    def check_train(self):
        while True:
            if len(self.trajectory_record) >= self.batch_size:
                self.train(self.trajectory_record[:self.batch_size])
                del self.trajectory_record[:self.batch_size]
            time.sleep(10)

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    print(hostname)
    ip = socket.gethostbyname(hostname)
    print(ip)
