import os
import time
from typing import List, Union, Dict, Callable, Any
from functools import partial
from queue import Queue
from threading import Thread

from nervex.utils import read_file, save_file, get_data_decompressor
from nervex.interaction import Slave, TaskFail
from .base_comm_learner import BaseCommLearner, register_comm_learner
from ..learner_hook import LearnerHook


class LearnerSlave(Slave):

    def __init__(self, *args, callback_fn: Dict[str, Callable], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._callback_fn = callback_fn

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        task_name = task['name']
        if task_name == 'resource':
            return self._callback_fn['deal_with_resource']()
        elif task_name == 'learner_start_task':
            self._current_task_info = task['task_info']
            self._callback_fn['deal_with_learner_start'](self._current_task_info)
            return {'message': 'learner task has started'}
        elif task_name == 'learner_get_data_task':
            data_demand = self._callback_fn['deal_with_get_data']()
            ret = {
                'task_id': self._current_task_info['task_id'],
                'buffer_id': self._current_task_info['buffer_id'],
            }
            ret.update(data_demand)
            return ret
        elif task_name == 'learner_learn_task':
            learn_info = self._callback_fn['deal_with_learner_learn'](task['data'])
            ret = {
                'info': learn_info,
                'task_id': self._current_task_info['task_id'],
                'buffer_id': self._current_task_info['buffer_id']
            }
            ret['finished_task'] = learn_info.get('finished_task', None)
            finished_task = learn_info.get('finished_task', None)
            if finished_task is not None:
                finished_task['buffer_id'] = self._current_task_info['buffer_id']
                self._current_task_info = None
                ret['finished_task'] = finished_task
            else:
                ret['finished_task'] = None
            return ret
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal actor task <{}>'.format(task_name))


class FlaskFileSystemLearner(BaseCommLearner):
    """
    Overview:
        An implementation of CommLearner, using flask and the file system.
    Interfaces:
        __init__, send_policy, get_data, send_learn_info, start, close
    Property:
        hooks4call
    """

    def __init__(self, cfg: 'EasyDict') -> None:  # noqa
        """
        Overview:
            Initialize file path(url, path of data & policy), comm frequency, dist learner info according to cfg.
        Arguments:
            - cfg (:obj:`EasyDict`): config dict
        """
        BaseCommLearner.__init__(self, cfg)
        host, port = cfg.host, cfg.port
        if isinstance(port, list):
            port = port[self._rank]
        self._callback_fn = {
            'deal_with_resource': self.deal_with_resource,
            'deal_with_learner_start': self.deal_with_learner_start,
            'deal_with_get_data': self.deal_with_get_data,
            'deal_with_learner_learn': self.deal_with_learner_learn,
        }
        self._slave = LearnerSlave(host, port, callback_fn=self._callback_fn)

        self._path_data = cfg.path_data
        self._path_policy = cfg.path_policy
        self._send_policy_freq = cfg.send_policy_freq

        self._data_demand_queue = Queue(maxsize=1)
        self._data_result_queue = Queue(maxsize=1)
        self._learn_info_queue = Queue(maxsize=1)
        self._learner = None
        self._policy_id = None

    def start(self) -> None:
        BaseCommLearner.start(self)
        self._slave.start()

    def close(self) -> None:
        if self._end_flag:
            return
        if hasattr(self, '_learner_thread'):
            self._learner_thread.join()
        if self._learner is not None:
            self._learner.close()
        self._slave.close()
        BaseCommLearner.close(self)

    def __del__(self) -> None:
        self.close()

    def deal_with_resource(self) -> dict:
        return {'gpu': self._world_size}

    def deal_with_learner_start(self, task_info: dict) -> None:
        self._policy_id = task_info['policy_id']
        self._learner = self._create_learner(task_info)
        for h in self.hooks4call:
            self._learner.register_hook(h)
        self._learner_thread = Thread(target=self._learner.start, args=(), daemon=True)
        self._learner_thread.start()

    def deal_with_get_data(self) -> Any:
        data_demand = self._data_demand_queue.get()
        return data_demand

    def deal_with_learner_learn(self, data: dict) -> dict:
        self._data_result_queue.put(data)
        learn_info = self._learn_info_queue.get()
        finished_task = learn_info.get('finished_task', None)
        if finished_task is not None:
            self._learner.close()
            self._learner = None
            self._policy_id = None
        return learn_info

    # override
    def send_policy(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's policy in corresponding path, called by ``SendpolicyHook``.
        Arguments:
            - state_dict (:obj:`dict`): state dict of the runtime policy
        """
        path = os.path.join(self._path_policy, self._policy_id)
        save_file(path, state_dict)

    @staticmethod
    def load_data_fn(path, meta, decompressor):
        # due to read-write conflict, read_file may be error, therefore we circle this procedure
        while True:
            try:
                s = read_file(path)
                s = decompressor(s)
                break
            except Exception:
                time.sleep(0.01)
        unroll_len = meta.get('unroll_len', 1)
        if 'unroll_split_begin' in meta:
            begin = meta['unroll_split_begin']
            if unroll_len == 1:
                s = s[begin]
                s.update(meta)
            else:
                end = begin + unroll_len
                s = s[begin:end]
                # add metdata key-value to stepdata
                for i in range(len(s)):
                    s[i].update(meta)
        else:
            s.update(meta)
        return s

    # override
    def get_data(self, batch_size: int) -> list:  # todo: doc not finished
        """
        Overview:
            Get batched data from coordinator.
        Arguments:
            - batch_size (:obj:`int`): size of one batch
        Returns:
            - data (:obj:`list`): a list of train data, each element is one data
        """
        while self._learner is None:
            time.sleep(1)
        assert self._data_demand_queue.qsize() == 0
        self._data_demand_queue.put({'batch_size': batch_size, 'cur_learner_iter': self._learner.last_iter.val})
        data = self._data_result_queue.get()
        assert isinstance(data, list)
        assert len(data) == batch_size, '{}/{}'.format(len(data), batch_size)
        decompressor = get_data_decompressor(data[0].get('compressor', 'none'))
        data = [
            partial(
                FlaskFileSystemLearner.load_data_fn,
                path=os.path.join(self._path_data, m['data_id']),
                meta=m,
                decompressor=decompressor,
            ) for m in data
        ]
        return data

    # override
    def send_learn_info(self, learn_info: dict) -> None:
        """
        Overview:
            Send learn info to coordinator, called by ``SendLearnInfoHook``.
            Sending will repeat until succeeds or ``_active_flag`` is set to False.
        Arguments:
            - learn info (:obj:`dict`): learn info in `dict` type, including keys `learn_info`\
                (e.g. last iter, priority info)
        """
        assert self._learn_info_queue.qsize() == 0
        self._learn_info_queue.put(learn_info)

    @property
    def hooks4call(self) -> List[LearnerHook]:
        """
        Overview:
            Initialize the hooks and return them.
        Returns:
            - hooks (:obj:`list`): the hooks which comm learner have, will be registered in learner as well.
        """
        return [
            SendPolicyHook('send_policy', 100, position='before_run', ext_args={}),
            SendPolicyHook(
                'send_policy', 100, position='after_iter', ext_args={'send_policy_freq': self._send_policy_freq}
            ),
            SendLearnInfoHook(
                'send_learn_info',
                100,
                position='after_iter',
                ext_args={'freq': 10},
            ),
            SendLearnInfoHook(
                'send_learn_info',
                100,
                position='after_run',
                ext_args={'freq': 1},
            ),
        ]


class SendPolicyHook(LearnerHook):
    """
    Overview:
        Hook to send policy
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: dict = {}, **kwargs) -> None:
        """
        Overview:
            init SendpolicyHook
        Arguments:
            - ext_args (:obj:`dict`): extended_args, use ext_args.freq to set send_policy_freq
        """
        super().__init__(*args, **kwargs)
        if 'send_policy_freq' in ext_args:
            self._freq = ext_args['send_policy_freq']
        else:
            self._freq = 1

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Save learner's policy in corresponding path at interval iterations, including model_state_dict, last_iter
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        last_iter = engine.last_iter.val
        if engine.rank == 0 and last_iter % self._freq == 0:
            state_dict = {'model': engine.policy.state_dict_handle()['model'].state_dict(), 'iter': last_iter}
            engine.send_policy(state_dict)
            engine.info('{} save iter{} policy'.format(engine.name, last_iter))


class SendLearnInfoHook(LearnerHook):
    """
    Overview:
        Hook to send learn info
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: dict, **kwargs) -> None:
        """
        Overview:
            init SendLearnInfoHook
        Arguments:
            - ext_args (:obj:`dict`): extended_args, use ext_args.freq
        """
        super().__init__(*args, **kwargs)
        self._freq = ext_args['freq']

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Send learn info including last_iter at interval iterations and priority info
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner
        """
        last_iter = engine.last_iter.val
        engine.send_learn_info(engine.learn_info)
        if last_iter % self._freq == 0:
            engine.info('{} save iter{} learn_info'.format(engine.name, last_iter))


register_comm_learner('flask_fs', FlaskFileSystemLearner)
