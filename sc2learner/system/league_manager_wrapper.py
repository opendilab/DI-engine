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
import requests

from sc2learner.data.online import ReplayBuffer
from sc2learner.utils import read_file_ceph, save_file_ceph
from sc2learner.league import LeagueManager


class LeagueManagerWrapper(object):
    def __init__(self, cfg):
        self.cfg = cfg

        if 'league_manager_ip' in self.cfg.system.keys():
            self.league_manager_ip = self.cfg.system.league_manager_ip
        else:
            self.league_manager_ip = os.environ.get('SLURMD_NODENAME', '')  # hostname like SH-IDC1-10-5-36-236
        if not self.league_manager_ip:
            raise ValueError('league_manager_ip must be ip address, but found {}'.format(self.learner_ip))
        self.coordinator_ip = self.cfg['system']['coordinator_ip']
        self.coordinator_port = self.cfg['system']['coordinator_port']
        self.ceph_traj_path = self.cfg['system']['ceph_traj_path']
        self.ceph_model_path = self.cfg['system']['ceph_model_path']
        self.use_ceph = self.cfg['system']['use_ceph']

        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)
        self.use_fake_data = cfg['coordinator']['use_fake_data']
        if self.use_fake_data:
            self.fake_model_path = cfg['system']['fake_model_path']
            self.fake_stat_path = cfg['system']['fake_stat_path']
        self._set_logger()
        self._init_league_manager()
        self._register_league_manager()


    def _set_logger(self, level=1):
        self.logger = logging.getLogger("league_manager.log")

    def _init_league_manager(self):
        def save_checkpoint_fn(src_checkpoint, dst_checkpoint, read_type='pickle'):
            '''
                Overview: copy src_checkpoint as dst_checkpoint
                Arguments:
                    - src_checkpoint (:obj:`str`): source checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
                    - dst_checkpoint (:obj:`str`): dst checkpoint's path, e.g. s3://alphastar_fake_data/ckpt.pth
            '''
            if self.use_fake_data:
                return
            src_checkpoint = os.path.join(self.ceph_model_path, src_checkpoint)
            dst_checkpoint = os.path.join(self.ceph_model_path, dst_checkpoint)
            checkpoint = read_file_ceph(src_checkpoint, read_type=read_type)
            ceph_path, file_name = dst_checkpoint.strip().rsplit('/', 1)
            save_file_ceph(ceph_path, file_name, checkpoint)
            self.logger.info('[league manager] load {} and resave to {}.'.format(src_checkpoint, dst_checkpoint))

        def load_checkpoint_fn(player_id, checkpoint_path):
            d = {'player_id': player_id, 'checkpoint_path': self.ceph_model_path + checkpoint_path}
            # need to be refine
            while True:
                try:
                    response = requests.post(self.url_prefix + "coordinator/ask_learner_to_reset", json=d).json()
                    if response['code'] == 0:
                        return True
                except Exception as e:
                    print("something wrong with coordinator, {}".format(e))
                time.sleep(10)
            return False

        def launch_match_fn(launch_info):
            d = {'launch_info': launch_info}
            # need to be refine
            while True:
                try:
                    response = requests.post(self.url_prefix + "coordinator/add_launch_info", json=d).json()
                    if response['code'] == 0:
                        return True
                except Exception as e:
                    print("something wrong with coordinator, {}".format(e))
                time.sleep(10)
            return False

        self.league_manager = LeagueManager(self.cfg, save_checkpoint_fn, load_checkpoint_fn, launch_match_fn)
        self.player_ids = self.league_manager.active_players_ids
        self.player_ckpts = self.league_manager.active_players_ckpts
        print('{} learners should be registered totally. '.format(len(self.player_ids)))

    def _register_league_manager(self):
        d = {
            'league_manager_ip': self.league_manager_ip,
            'player_ids': self.player_ids,
            'player_ckpts': self.player_ckpts
        }
        while True:
            try:
                response = requests.post(self.url_prefix + "coordinator/register_league_manager", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                print("something wrong with coordinator, {}".format(e))
            time.sleep(10)
        return False

    def deal_with_run_league(self):
        self.league_manager.run()
        return True

    def deal_with_finish_match(self, match_info):
        self.league_manager.finish_match(match_info)
        return True

    def get_ip(self):
        return self.league_manager_ip








