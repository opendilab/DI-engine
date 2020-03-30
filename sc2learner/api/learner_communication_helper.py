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
from queue import Queue

from utils import read_file_ceph, save_file_ceph


class LearnerCommunicationHelper(object):
    def __init__(self, cfg):
        super(LearnerCommunicationHelper, self).__init__()

        self.cfg = cfg

        self.learner_uid = str(uuid.uuid1())
        self.learner_ip = socket.gethostname()  # hostname like SH-IDC1-10-5-36-236
        self.coordinator_ip = self.cfg['api']['coordinator_ip']
        self.coordinator_port = self.cfg['api']['coordinator_port']
        self.ceph_path = self.cfg['api']['ceph_path']

        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)
        self.data_queue = Queue()
        self.data_iterator = iter(self.data_queue.get)

        self._setup_comm_logger()
        self.register_learner_in_coordinator()

        self.batch_size = cfg.train.batch_size  # once len(self.trajectory_record) reach this value, train

    def _setup_comm_logger(self):
        self.comm_logger = logging.getLogger("learner.log")

    def register_learner_in_coordinator(self):
        # register learner in coordinator
        while True:
            try:
                d = {'learner_uid': self.learner_uid, 'learner_ip': self.learner_ip}
                response = requests.post(self.url_prefix + "coordinator/register_learner", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self.comm_logger.info("something wrong with coordinator, {}".format(e))
            time.sleep(10)

    def register_model_in_coordinator(self, model_name):
        # register model in coordinator
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
        save_file_ceph(self.ceph_path, model_name, model)
        self.comm_logger.info("save model {} to ceph".format(model_name))

    def load_trajectory_from_ceph(self, metadata):
        assert (isinstance(metadata, dict))
        metadata['raw_data'] = read_file_ceph(self.ceph_path + metadata['trajectory_path'], read_type='pickle')
        self.comm_logger.info("load trajectory {} to ceph".format(metadata['trajectory_path']))
        return metadata

    def deal_with_get_metadata(self, metadatas):
        assert (isinstance(metadatas, list))
        job_id = metadatas['job_id']
        self.data_queue.put(metadatas)
