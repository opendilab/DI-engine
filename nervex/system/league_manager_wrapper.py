import logging
import os
import time

import requests

from nervex.league import create_league
from nervex.utils import read_file, save_file


class LeagueManagerWrapper(object):

    def __init__(self, cfg):
        self.cfg = cfg

        if 'league_manager_ip' not in self.cfg.system.keys() or self.cfg.system.league_manager_ip == 'auto':
            self.league_manager_ip = os.environ.get('SLURMD_NODENAME', '')  # hostname like SH-IDC1-10-5-36-236
        else:
            self.league_manager_ip = self.cfg.system.league_manager_ip
        if not self.league_manager_ip:
            raise ValueError('league_manager_ip must be ip address, but found {}'.format(self.league_manager_ip))
        self.coordinator_ip = self.cfg.system.coordinator_ip
        self.coordinator_port = self.cfg.system.coordinator_port
        self.path_agent = self.cfg.league.communication.path_agent

        self.url_prefix = 'http://{}:{}/'.format(self.coordinator_ip, self.coordinator_port)
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
            src_checkpoint = os.path.join(self.path_agent, src_checkpoint)
            dst_checkpoint = os.path.join(self.path_agent, dst_checkpoint)
            checkpoint = read_file(src_checkpoint)
            save_file(dst_checkpoint, checkpoint)
            self.logger.info('[league manager] load {} and resave to {}.'.format(src_checkpoint, dst_checkpoint))

        def load_checkpoint_fn(player_id, checkpoint_path):
            d = {'player_id': player_id, 'checkpoint_path': self.path_agent + checkpoint_path}
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

        self._setup_league_manager(save_checkpoint_fn, load_checkpoint_fn, launch_match_fn)
        self.player_ids = self.league_manager.active_players_ids
        self.player_ckpts = self.league_manager.active_players_ckpts
        print('{} learners should be registered totally. '.format(len(self.player_ids)))

    def _setup_league_manager(self, save_checkpoint_fn, load_checkpoint_fn, launch_match_fn):
        self.league_manager = create_league(self.cfg, save_checkpoint_fn, load_checkpoint_fn, launch_match_fn)

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
        # TODO launch learner job
        self.league_manager.run()
        return True

    def deal_with_finish_task(self, task_info):
        self.league_manager.finish_task(task_info)
        return True

    def deal_with_update_active_player(self, player_info):
        self.league_manager.update_active_player(player_info)
        return True

    @property
    def ip(self):
        return self.league_manager_ip

    def close(self):
        self.league_manager.close()
