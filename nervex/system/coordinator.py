import enum
import logging
import os
import threading
import time
import uuid
from queue import Queue

import requests
import torch
from easydict import EasyDict

from nervex.data.online import ReplayBuffer
from nervex.utils import LockContext


class JobState(enum.IntEnum):
    running = 1
    finish = 2


class LearnerState(enum.IntEnum):
    alive = 1
    dead = 2


class Coordinator(object):

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._setup_logger()

        # resume
        self._resume_dir = cfg.system.resume_dir
        self._resume_label = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self._save_resume_freq = 60 * 1
        self._load_resume()
        check_resume_thread = threading.Thread(target=self.check_resume)
        check_resume_thread.daemon = True
        check_resume_thread.start()
        self._logger.info("[UP] check resume thread ")

        # league manager
        self._league_manager_port = cfg.system.league_manager_port
        self._player_to_learner = {}
        self._learner_to_player = {}
        self._league_manager_flag = False
        launch_league_thread = threading.Thread(target=self._launch_league_manager)
        launch_league_thread.daemon = True
        launch_league_thread.data_index = True
        launch_league_thread.start()
        self._logger.info("[UP] launch league manager thread ")

        # manager:
        # {manager_uid: {actor_uid: [job_id]}}
        self.manager_record = {}
        # {job_id: {content: info, state: JobState}}
        self.job_record = {}
        # {learner_uid: {"learner_ip_port_list": [[learner_ip, learner_port]],
        #                "job_ids": [job_id],
        #                "world_size": world_size,
        #                "checkpoint_path": checkpoint_path,
        #                "replay_buffer": replay_buffer,
        #                "last_beats_time": last_beats_time,
        #                "state": LearnerState}

        # learner
        self._learner_record = {}
        self._check_dead_learner_freq = cfg.system.coordinator_check_dead_learner_freq
        check_learner_dead_thread = threading.Thread(target=self.check_learner_dead)
        check_learner_dead_thread.daemon = True
        check_learner_dead_thread.start()
        self._logger.info("[UP] check learner dead thread ")

        self.url_prefix_format = 'http://{}:{}/'
        self.lock = LockContext(lock_type='process')
        self.job_queue = Queue()

    def close(self):
        for k, v in self._learner_record.items():
            self._learner_record[k]['replay_buffer'].close()

    def _setup_logger(self, level=1):
        self._logger = logging.getLogger("coordinator.log")

    def _load_resume(self):
        if self.cfg.system.coordinator_resume_path and os.path.isfile(self.cfg.system.coordinator_resume_path):
            data = torch.load(self.cfg.system.coordinator_resume_path)
            self.manager_record, self.job_record, self._learner_record = data
            for k, v in self._learner_record.items():
                # launch new replay buffer
                self._learner_record[k]['replay_buffer'] = ReplayBuffer(EasyDict(self.cfg.replay_buffer))
                self._learner_record[k]['replay_buffer'].run()
                self._learner_record[k]['last_beats_time'] = time.time()

    def _save_resume(self):
        tmp = {}
        for k, v in self._learner_record.items():
            tmp[k] = {}
            tmp[k]['learner_ip_port_list'] = v['learner_ip_port_list']
            tmp[k]['job_ids'] = v['job_ids']
            tmp[k]['checkpoint_path'] = v['checkpoint_path']
            tmp[k]['state'] = v['state']
        data = [self.manager_record, self.job_record, tmp]
        torch.save(data, os.path.join(self._resume_dir, 'coordinator.resume.' + self._resume_label))

    def deal_with_register_manager(self, manager_uid):
        '''
            Overview: deal with register from manager
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
        '''
        # TODO re register manager
        if manager_uid not in self.manager_record:
            self.manager_record[manager_uid] = {}
        else:
            self._logger.info('manager({}) has been registered'.format(manager_uid))
        return True

    def deal_with_register_learner(self, learner_uid, learner_ip, learner_port, world_size, restore=False):
        '''
            Overview: deal with register from learner, make learner and player pairs
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - learner_ip (:obj:`str`): learner's ip
                - learner_port (:obj:`str`): learner's port
                - world_size (:obj:`int`): the number of processes in one learner
                - restore (:obj:`bool`): whether register the previous learner
        '''
        if hasattr(self, '_player_ids'):
            with self.lock:
                self._logger.info('now learner_record: {}'.format(self._learner_record))
                if learner_uid not in self._learner_record:
                    if len(self._player_to_learner) < len(self._player_ids):
                        self._learner_record[learner_uid] = {
                            "learner_ip_port_list": [[learner_ip, learner_port]],
                            "world_size": world_size,
                            "job_ids": [],
                            "checkpoint_path": '',
                            "replay_buffer": ReplayBuffer(EasyDict(self.cfg.replay_buffer)),
                            "last_beats_time": int(time.time()),
                            "state": LearnerState.alive
                        }
                        self._learner_record[learner_uid]['replay_buffer'].run()
                        for index, player_id in enumerate(self._player_ids):
                            if player_id not in self._player_to_learner:
                                self._player_to_learner[player_id] = learner_uid
                                self._learner_to_player[learner_uid] = player_id
                                self._learner_record[learner_uid]['checkpoint_path'] = self._player_ckpts[index]
                                self._logger.info('learner ({}) set to player ({})'.format(learner_uid, player_id))
                                break
                        self._logger.info(
                            '{}/{} learners have been registered'.format(
                                len(self._player_to_learner), len(self._player_ids)
                            )
                        )
                    else:
                        self._logger.info(
                            'learner {} try to register, but enough learners have been registered.'.format(learner_uid)
                        )
                        return False
                else:
                    if restore:
                        # if restore, empty learner_ip_port_list
                        self._learner_record[learner_uid]['learner_ip_port_list'] = []
                        self._learner_record[learner_uid]['state'] = LearnerState.alive
                    if [learner_ip, learner_port] not in self._learner_record[learner_uid]['learner_ip_port_list']:
                        self._learner_record[learner_uid]['learner_ip_port_list'].append([learner_ip, learner_port])

                self._logger.info('learner ({}) register, ip {}, port {}'.format(learner_uid, learner_ip, learner_port))
                return self._learner_record[learner_uid]['checkpoint_path']
        else:
            self._logger.info('learner can not register now, because league manager is not set up')
            return False

    def deal_with_ask_for_job(self, manager_uid, actor_uid):
        '''
            Overview: deal with job request from manager
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
                - actor_uid (:obj:`str`): actor's uid
        '''
        job = self.job_queue.get()
        assert isinstance(job, dict)
        job_id = job['job_id']
        if manager_uid not in self.manager_record:
            self.deal_with_register_manager(manager_uid)
        if actor_uid not in self.manager_record[manager_uid]:
            self.manager_record[manager_uid][actor_uid] = []
        # TODO deal with too many job id
        self.manager_record[manager_uid][actor_uid].append(job_id)
        self.job_record[job_id] = {'content': job, 'state': JobState.running}
        return job

    def deal_with_get_metadata(self, job_id, metadata):
        '''
            Overview: when receiving manager's request of sending metadata, return True/False
            Arguments:
                - job_id (:obj:`str`): job's id
                - metadata (:obj:`dict`): the trajectory metadata sent by actor
            Returns:
                - (:obj`bool`): state
        '''
        # assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        # keep related id info as func arguments, which is ready for further development
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        learner_uid = metadata['learner_uid']
        self._learner_record[learner_uid]['replay_buffer'].push_data(metadata)
        return True

    def deal_with_finish_job(self, manager_uid, actor_uid, job_id, result):
        '''
            Overview: when receiving actor's request of finishing job, return True/False
            Arguments:
                - manager_uid (:obj:`str`): manager's uid
                - actor_uid (:obj:`str`): actor's uid
                - job_id (:obj:`str`): job's id
                - result (:obj:`str`): job result
            Returns:
                - (:obj`bool`): state
        '''
        assert job_id in self.job_record, 'job_id ({}) not in job_record'.format(job_id)
        self.job_record[job_id]['state'] = JobState.finish
        url_prefix = self.url_prefix_format.format(self._league_manager_ip, self._league_manager_port)
        d = {'task_result': result}
        while True:
            try:
                response = requests.post(url_prefix + "league/finish_task", json=d).json()
                if response['code'] == 0:
                    return True
            except Exception as e:
                self._logger.info("something wrong with league_manager, {}".format(e))
            time.sleep(3)
        return False

    def deal_with_ask_for_metadata(self, learner_uid, batch_size):
        '''
            Overview: when receiving learner's request of asking for metadata, return metadatas
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - batch_size (:obj:`int`): batch size
            Returns:
                - (:obj`list`): metadata list
        '''
        assert learner_uid in self._learner_record, 'learner_uid ({}) not in learner_record'.format(learner_uid)
        self._learner_record[learner_uid]['last_beats_time'] = int(time.time())
        with self.lock:
            return self._learner_record[learner_uid]['replay_buffer'].sample(batch_size)

    def deal_with_train_info(self, learner_uid, train_info):
        '''
            Overview: when receiving learner's request of updating replay buffer, return True/False
            Arguments:
                - learner_uid (:obj:`str`): learner's uid
                - update_info (:obj:`dict`): info dict
            Returns:
                - (:obj`bool`): True
        '''
        self._learner_record[learner_uid]['last_beats_time'] = int(time.time())
        self._learner_record[learner_uid]['last_iter'] = train_info['iter']
        # update info for league
        player_id = self._learner_to_player.get(learner_uid)
        player_info = {'player_id': player_id, 'train_step': train_info['iter']}
        url_prefix = self.url_prefix_format.format(self._league_manager_ip, self._league_manager_port)
        d = {'player_info': player_info}
        while True:
            try:
                response = requests.post(url_prefix + "league/update_active_player", json=d).json()
                if response['code'] == 0:
                    break
            except Exception as e:
                self._logger.info("something wrong with league_manager, {}".format(e))
            time.sleep(3)
        # update info for buffer
        # TODO PER update
        # self._learner_record[learner_uid]['replay_buffer'].update(update_info)
        return True

    def deal_with_register_league_manager(self, league_manager_ip, player_ids, player_ckpts):
        self._league_manager_ip = league_manager_ip
        self._player_ids = player_ids
        self._player_ckpts = player_ckpts
        self._logger.info('register league_manager from {}'.format(self._league_manager_ip))
        self._league_manager_flag = True
        return True

    def deal_with_ask_learner_to_reset(self, player_id, checkpoint_path):
        learner_uid = self._player_to_learner[player_id]
        d = {'checkpoint_path': checkpoint_path}
        for learner_ip, learner_port in self._learner_record[learner_uid]['learner_ip_port_list']:
            url_prefix = self.url_prefix_format.format(learner_ip, learner_port)
            while True:
                try:
                    response = requests.post(url_prefix + "learner/reset", json=d).json()
                    if response['code'] == 0:
                        break
                except Exception as e:
                    self._logger.info("something wrong with learner {}, {}".format(learner_uid, e))
                time.sleep(2)
        return True

    def deal_with_add_launch_info(self, launch_info):
        """
            Overview: when receiving league manager's launch match request, prepare a new job and push it into job_queue
            Arguments:
                - launch_info (:obj:`dict`): launch match info, please refer to league manager for details
            Returns:
                - (:obj`bool`): state
        """
        learner_uid = [
            self._player_to_learner[p] if p in self._player_to_learner.keys() else None
            for p in launch_info['player_id']
        ]
        job = {
            'job_id': str(uuid.uuid1()),
            'learner_uid': learner_uid,
        }
        job.update(launch_info)
        self.job_queue.put(job)
        return True

    def deal_with_get_heartbeats(self, learner_uid):
        '''
            Overview: when receiving learner's heartbeats, update last_beats_time.
            Arguments:
                - actor_uid (:obj:`str`): learner's uid
            Returns:
                - (:obj`bool`): state
        '''
        assert learner_uid in self._learner_record
        self._learner_record[learner_uid]['last_beats_time'] = int(time.time())
        return True

    ###################################################################################
    #                                     threads                                     #
    ###################################################################################

    def _launch_league_manager(self):
        while True:
            # league_manager and enough learners have been registered
            if self._league_manager_flag and len(self._player_ids) == len(self._player_to_learner):
                try:
                    url_prefix = self.url_prefix_format.format(self._league_manager_ip, self._league_manager_port)
                    response = requests.get(url_prefix + "league/run_league").json()
                    if response['code'] == 0:
                        self._logger.info('league_manager run with table {}. '.format(self._player_to_learner))
                        break
                except Exception as e:
                    self._logger.info('launch_league_thread error {}'.format(e))
            time.sleep(10)

    def time_format(self, time_item):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_item))

    def deal_with_dead_learner(self, learner_uid, reuse=True):
        self._learner_record[learner_uid]['state'] = LearnerState.dead
        os.system('scancel ' + learner_uid)
        self._logger.info('[kill-dead] dead learner {} was killed'.format(learner_uid))

    def check_learner_dead(self):
        while True:
            nowtime = int(time.time())
            for learner_uid, learner_info in self._learner_record.items():
                if learner_info['state'] == LearnerState.alive and \
                        nowtime - learner_info['last_beats_time'] > self._check_dead_learner_freq:
                    # dead learner
                    self._logger.info(
                        "[coordinator][check_learner_dead] {} is dead, last_beats_time = {}".format(
                            learner_uid, self.time_format(learner_info['last_beats_time'])
                        )
                    )
                    self.deal_with_dead_learner(learner_uid)
            time.sleep(self._check_dead_learner_freq)

    def check_resume(self):
        self.lasttime = int(time.time())
        while True:
            nowtime = int(time.time())
            if nowtime - self.lasttime > self._save_resume_freq:
                self._save_resume()
                p = os.path.join(self._resume_dir, 'coordinator.resume.' + self._resume_label)
                self._logger.info('[resume] save to {}'.format(p))
                self.lasttime = nowtime
            time.sleep(self._save_resume_freq)

    ###################################################################################
    #                                      debug                                      #
    ###################################################################################

    def deal_with_get_all_manager(self):
        return self.manager_record

    def deal_with_get_all_learner(self):
        return self._learner_record

    def deal_with_get_all_job(self):
        return self.job_record

    def deal_with_get_replay_buffer(self, learner_uid):
        return self._learner_record[learner_uid]['replay_buffer']

    def deal_with_get_job_queue(self):
        pass

    def deal_with_push_data_to_replay_buffer(self, learner_uid):
        job_id = '8d2e8eda-83d9-11ea-8bb0-1be4f1872daf'
        # learner_uid = '3458436'
        trajectory_path = 'model_main_player_zerg_0_ckpt' \
                          '.pth_job_0098e642-841e-11ea-9918-6f27a4855242_agent_0_step_1159_' \
                          '0707b170-8423-11ea-99b0-db6573da5763.traj'
        self._learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        self._learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        self._learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        self._learner_record[learner_uid]['replay_buffer'].push_data(
            {
                'job_id': job_id,
                'trajectory_path': trajectory_path,
                'learner_uid': learner_uid,
                'data': [[1, 2, 3], [4, 5, 6]],
                'step_data_compressor': 'lz4'
            }
        )
        return True
