import os
import sys
import time
import traceback

import requests
from typing import List

from nervex.utils import read_file, save_file, get_rank, get_world_size
from .base_comm_learner import BaseCommLearner
from ..learner_hook import LearnerHook


class FlaskFileSystemLearner(BaseCommLearner):
    """
    Overview:
        An implementation of CommLearner, using flask as the file system.
    Interfaces:
        __init__, register_learner, send_agent, get_data, send_train_info, start_heartbeats_thread
        init_service, close_service,
    Property:
        hooks4call
    """
    def __init__(self, cfg: 'EasyDict') -> None:  # noqa
        """
        Overview:
            Initialize file path(url, path of traj & agent), comm frequency, dist learner info according to cfg.
        Arguments:
            - cfg (:obj:`EasyDict`): config dict
        """
        super(FlaskFileSystemLearner, self).__init__(cfg)
        self._url_prefix = 'http://{}:{}/'.format(cfg.upstream_ip, cfg.upstream_port)

        self._path_traj = cfg.path_traj
        self._path_agent = cfg.path_agent
        # thread: _heartbeats_freq; hook: _send_agent_freq, _send_train_info_freq
        self._heartbeats_freq = cfg.heartbeats_freq
        self._send_agent_freq = cfg.send_agent_freq
        self._send_train_info_freq = cfg.send_train_info_freq
        self._rank = get_rank()
        self._world_size = get_world_size()
        if 'learner_ip' not in cfg.keys() or cfg.learner_ip == 'auto':
            self._learner_ip = os.environ.get('SLURMD_NODENAME', '')
        else:
            self._learner_ip = cfg.learner_ip
        self._learner_port = cfg.learner_port - self._rank
        self._restore = cfg.restore

    # override
    def register_learner(self) -> None:  # todo: 1 learner -> many agent?
        """
        Overview:
            Register learner's info in coordinator, called by ``self.init_service``.
            Will set property ``_agent_name`` to returned response.info. Registration will repeat until succeeds.
        """
        d = {
            'learner_uid': self._learner_uid,
            'learner_ip': self._learner_ip,
            'learner_port': self._learner_port,
            'world_size': self._world_size,
            'restore': self._restore
        }
        while True:  # only after registration succeeds, can ``_active_flag`` be set to True
            result = self._flask_send(d, 'coordinator/register_learner')
            if result is not None and result['code'] == 0:
                self._agent_name = result['info']
                return
            else:
                time.sleep(10)

    # override
    def send_agent(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's agent in corresponding path, called by ``SendAgentHook``.
        Arguments:
            - state_dict (:obj:`dict`): state dict of the runtime agent
        """
        path = os.path.join(self._path_agent, self._agent_name)
        save_file(path, state_dict)

    # override
    def get_data(self, batch_size: int) -> list:  # todo: doc not finished
        """
        Overview:
            Get batched data from coordinator.
        Arguments:
            - batch_size (:obj:`int`): size of one batch
        Returns:
            - stepdata (:obj:`list`): a list of train data, each element is one traj
        """
        d = {'learner_uid': self._learner_uid, 'batch_size': batch_size}
        sleep_count = 5
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
            time.sleep(sleep_count)
            sleep_count += 5

    # override
    def send_train_info(self, train_info: dict) -> None:
        """
        Overview:
            Send train info to coordinator, called by ``SendTrainInfoHook``.
            Sending will repeat until succeeds or ``_active_flag`` is set to False.
        Arguments:
            - train info (:obj:`dict`): train info in `dict` type, \
                including keys `train_info`(last iter), `learner_uid`
        """
        d = {'train_info': train_info, 'learner_uid': self._learner_uid}
        while self._active_flag:
            result = self._flask_send(d, 'coordinator/send_train_info')
            if result is not None and result['code'] == 0:
                return
            else:
                time.sleep(1)

    # override
    def _send_learner_heartbeats(self) -> None:
        """
        Overview:
            Send learner's heartbeats to coordinator, will start as a thread in ``self.start_heartbeats_thread``.
            Sending will take place every ``_heartbeats_freq`` seconds until ``_active_flag`` is set to False.
        """
        d = {'learner_uid': self._learner_uid}
        while self._active_flag:
            self._flask_send(d, 'coordinator/get_heartbeats')
            for _ in range(self._heartbeats_freq):
                if not self._active_flag:
                    break
                time.sleep(1)

    def _flask_send(self, data: dict, api: str) -> dict:
        """
        Overview:
            Send info via flask and return the response.
            Log corresponding info/error when succeeds, fails or raises an exception.
        Arguments:
            - data (:obj:`dict`): the data to send via ``requests`` api
            - api (:obj:`str`): the specific api which the data will be sent to, \
                should add prefix ([ip]:[port]) before when using.
        Returns:
            - response (:obj:`dict`): if no exception raises, return the json response
        """
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
    def hooks4call(self) -> List[LearnerHook]:
        """
        Overview:
            Initialize the hooks and return them.
        Returns:
            - hooks (:obj:`list`): the hooks which comm learner have, will be registered in learner as well.
        """
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
    """
    Overview:
        Hook to send agent
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """
    def __init__(self, *args, ext_args: dict = {}, **kwargs) -> None:
        """
        Overview:
            init SendAgentHook
        Arguments:
            - ext_args (:obj:`dict`): extended_args, use ext_args.freq to set send_agent_freq
        """
        super().__init__(*args, **kwargs)
        if 'send_agent_freq' in ext_args:
            self._freq = ext_args['send_agent_freq']
        else:
            self._freq = 1

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Save learner's agent in corresponding path at interval iterations, including model_state_dict, last_iter
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        last_iter = engine.last_iter.val
        if engine.rank == 0 and last_iter % self._freq == 0:
            state_dict = {'model': engine.agent.model.state_dict(), 'iter': last_iter}
            engine.send_agent(state_dict)
            engine.info('{} save iter{} agent'.format(engine.name, last_iter))


class SendTrainInfoHook(LearnerHook):
    """
    Overview:
        Hook to send train info
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """
    def __init__(self, *args, ext_args: dict, **kwargs) -> None:
        """
        Overview:
            init SendTrainInfoHook
        Arguments:
            - ext_args (:obj:`dict`): extended_args, use ext_args.freq to set send_train_info_freq
        """
        super().__init__(*args, **kwargs)
        self._freq = ext_args['send_train_info_freq']

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Send train info including last_iter at interval iterations, learner_uid (added in ``send_train_info``)
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        last_iter = engine.last_iter.val
        if last_iter % self._freq == 0:
            state_dict = {'iter': last_iter}
            engine.send_train_info(state_dict)
            engine.info('{} save iter{} train_info'.format(engine.name, last_iter))
