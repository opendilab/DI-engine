import os
import sys
import time
import traceback

import requests

from nervex.utils import read_file, save_file, get_rank, get_world_size
from .base_comm_learner import BaseCommLearner
from ..learner_hook import LearnerHook


class FlaskFileSystemLearner(BaseCommLearner):
    def __init__(self, cfg: dict) -> None:
        super(FlaskFileSystemLearner, self).__init__(cfg)
        self._url_prefix = 'http://{}:{}/'.format(cfg.upstream_ip, cfg.upstream_port)

        self._path_traj = cfg.path_traj
        self._path_agent = cfg.path_agent
        self._heartbeats_freq = cfg.heartbeats_freq
        self._send_agent_freq = cfg.send_agent_freq
        self._send_train_info_freq = cfg.send_train_info_freq
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._learner_ip = cfg.learner_ip
        self._learner_port = cfg.learner_port - self._rank
        self._restore = cfg.restore

    # override
    def register_learner(self) -> None:
        d = {
            'learner_uid': self._learner_uid,
            'learner_ip': self._learner_ip,
            'learner_port': self._learner_port,
            'world_size': self._world_size,
            'restore': self._restore
        }
        while True:  # only registeration succeeded `_active_flag` can be True
            result = self._flask_send(d, 'coordinator/register_learner')
            if result is not None and result['code'] == 0:
                self._agent_name = result['info']
                return
            else:
                time.sleep(3)

    # override
    def send_agent(self, state_dict: dict) -> None:
        path = os.path.join(self._path_agent, self._agent_name)
        save_file(path, state_dict)

    # override
    def get_data(self, batch_size: int) -> list:
        d = {'learner_uid': self._learner_uid, 'batch_size': batch_size}
        while self._active_flag:
            result = self._flask_send(d, 'coordinator/ask_for_metadata')
            if result is not None and result['code'] == 0:
                metadata = result['info']
                if metadata is not None:
                    assert isinstance(metadata, list)
                    stepdata = []
                    for m in metadata:
                        path = os.path.join(self._path_traj, m['traj_id'])
                        # due to read-write conflict, read_file may be error, therefore we circle this procedure
                        while True:
                            try:
                                s = read_file(path)
                                break
                            except Exception as e:
                                self._logger.info('read_file error: {}({})'.format(e, path))
                                time.sleep(0.5)
                        begin, end = m['unroll_split_begin'], m['unroll_split_begin'] + m['unroll_len']
                        if m['unroll_len'] == 1:
                            s = s[begin]
                            s.update(m)
                        else:
                            s = s[begin:end]
                            # add metdata key-value to stepdata
                            for i in range(len(s)):
                                s[i].update(m)
                        stepdata.append(s)
                    return stepdata
            time.sleep(5)

    # override
    def send_train_info(self, train_info: dict) -> None:
        d = {'train_info': train_info, 'learner_uid': self._learner_uid}
        while self._active_flag:
            result = self._flask_send(d, 'coordinator/send_train_info')
            if result is not None and result['code'] == 0:
                return
            else:
                time.sleep(1)

    # override
    def _send_learner_heartbeats(self) -> None:
        d = {'learner_uid': self._learner_uid}
        while self._active_flag:
            self._flask_send(d, 'coordinator/get_heartbeats')
            for _ in range(self._heartbeats_freq):
                if not self._active_flag:
                    break
                time.sleep(1)

    def _flask_send(self, data, api):
        response = None
        try:
            response = requests.post(self._url_prefix + api, json=data).json()
            name = self._learner_uid
            if response['code'] == 0:
                self._logger.info("{} succeed sending result: {}".format(api, name))
            else:
                self._logger.error("{} failed to send result: {}".format(api, name))
        except Exception as e:
            self._logger.error(''.join(traceback.format_tb(e.__traceback__)))
            self._logger.error("[error] api({}): {}".format(api, sys.exc_info()))
        return response

    @property
    def hooks4call(self) -> list:
        return [
            SendAgentHook('send_agent', 100, position='before_run', ext_args={}),
            SendAgentHook(
                'send_agent', 100, position='after_iter', ext_args={'send_agent_freq': self._send_agent_freq}
            ),
            SendTrainInfoHook(
                'send_train_info',
                100,
                position='after_iter',
                ext_args={'send_train_info_freq': self._send_train_info_freq}
            ),
        ]


class SendAgentHook(LearnerHook):
    def __init__(self, *args, ext_args: dict = {}, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if 'send_agent_freq' in ext_args:
            self._freq = ext_args['send_agent_freq']
        else:
            self._freq = 1

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        last_iter = engine.last_iter.val
        if engine.rank == 0 and last_iter % self._freq == 0:
            state_dict = {'model': engine.agent.model.state_dict(), 'iter': last_iter}
            engine.send_agent(state_dict)
            engine.info('{} save iter{} agent'.format(engine.name, last_iter))


class SendTrainInfoHook(LearnerHook):
    def __init__(self, *args, ext_args: dict = {}, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._freq = ext_args['send_train_info_freq']

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        last_iter = engine.last_iter.val
        if last_iter % self._freq == 0:
            state_dict = {'iter': last_iter}
            engine.send_train_info(state_dict)
            engine.info('{} save iter{} train_info'.format(engine.name, last_iter))
