import os
import time
from typing import List, Union, Dict, Callable, Any
from functools import partial
from queue import Queue
from threading import Thread

from ding.utils import read_file, save_file, get_data_decompressor, COMM_LEARNER_REGISTRY
from ding.utils.file_helper import read_from_di_store
from ding.interaction import Slave, TaskFail
from .base_comm_learner import BaseCommLearner
from ..learner_hook import LearnerHook


class LearnerSlave(Slave):
    """
    Overview:
        A slave, whose master is coordinator.
        Used to pass message between comm learner and coordinator.
    """

    def __init__(self, *args, callback_fn: Dict[str, Callable], **kwargs) -> None:
        """
        Overview:
            Init callback functions additionally. Callback functions are methods in comm learner.
        """
        super().__init__(*args, **kwargs)
        self._callback_fn = callback_fn

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        """
        Overview:
            Process a task according to input task info dict, which is passed in by master coordinator.
            For each type of task, you can refer to corresponding callback function in comm learner for details.
        Arguments:
            - cfg (:obj:`EasyDict`): Task dict. Must contain key "name".
        Returns:
            - result (:obj:`Union[dict, TaskFail]`): Task result dict, or task fail exception.
        """
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
            info = self._callback_fn['deal_with_learner_learn'](task['data'])
            data = {'info': info}
            data['buffer_id'] = self._current_task_info['buffer_id']
            data['task_id'] = self._current_task_info['task_id']
            return data
        elif task_name == 'learner_close_task':
            self._callback_fn['deal_with_learner_close']()
            return {
                'task_id': self._current_task_info['task_id'],
                'buffer_id': self._current_task_info['buffer_id'],
            }
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal learner task <{}>'.format(task_name))


@COMM_LEARNER_REGISTRY.register('flask_fs')
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
            Init method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict.
        """
        BaseCommLearner.__init__(self, cfg)

        # Callback functions for message passing between comm learner and coordinator.
        self._callback_fn = {
            'deal_with_resource': self.deal_with_resource,
            'deal_with_learner_start': self.deal_with_learner_start,
            'deal_with_get_data': self.deal_with_get_data,
            'deal_with_learner_learn': self.deal_with_learner_learn,
            'deal_with_learner_close': self.deal_with_learner_close,
        }
        # Learner slave to implement those callback functions. Host and port is used to build connection with master.
        host, port = cfg.host, cfg.port
        if isinstance(port, list):
            port = port[self._rank]
        elif isinstance(port, int) and self._world_size > 1:
            port = port + self._rank
        self._slave = LearnerSlave(host, port, callback_fn=self._callback_fn)

        self._path_data = cfg.path_data  # path to read data from
        self._path_policy = cfg.path_policy  # path to save policy

        # Queues to store info dicts. Only one info is needed to pass between learner and coordinator at a time.
        self._data_demand_queue = Queue(maxsize=1)
        self._data_result_queue = Queue(maxsize=1)
        self._learn_info_queue = Queue(maxsize=1)

        # Task-level learner and policy will only be set once received the task.
        self._learner = None
        self._policy_id = None

    def start(self) -> None:
        """
        Overview:
            Start comm learner itself and the learner slave.
        """
        BaseCommLearner.start(self)
        self._slave.start()

    def close(self) -> None:
        """
        Overview:
            Join learner thread and close learner if still running.
            Then close learner slave and comm learner itself.
        """
        if self._end_flag:
            return
        if self._learner is not None:
            self.deal_with_learner_close()
        self._slave.close()
        BaseCommLearner.close(self)

    def __del__(self) -> None:
        """
        Overview:
            Call ``close`` for deletion.
        """
        self.close()

    def deal_with_resource(self) -> dict:
        """
        Overview:
            Callback function. Return how many resources are needed to start current learner.
        Returns:
            - resource (:obj:`dict`): Resource info dict, including ["gpu"].
        """
        return {'gpu': self._world_size}

    def deal_with_learner_start(self, task_info: dict) -> None:
        """
        Overview:
            Callback function. Create a learner and help register its hooks. Start a learner thread of the created one.
        Arguments:
            - task_info (:obj:`dict`): Task info dict.

        .. note::
            In ``_create_learner`` method in base class ``BaseCommLearner``, 3 methods
            ('get_data', 'send_policy', 'send_learn_info'), dataloader and policy are set.
            You can refer to it for details.
        """
        self._policy_id = task_info['policy_id']
        self._league_save_checkpoint_path = task_info.get('league_save_checkpoint_path', None)
        self._learner = self._create_learner(task_info)
        for h in self.hooks4call:
            self._learner.register_hook(h)
        self._learner_thread = Thread(target=self._learner.start, args=(), daemon=True, name='learner_start')
        self._learner_thread.start()

    def deal_with_get_data(self) -> Any:
        """
        Overview:
            Callback function. Get data demand info dict from ``_data_demand_queue``,
            which will be sent to coordinator afterwards.
        Returns:
            - data_demand (:obj:`Any`): Data demand info dict.
        """
        data_demand = self._data_demand_queue.get()
        return data_demand

    def deal_with_learner_learn(self, data: dict) -> dict:
        """
        Overview:
            Callback function. Put training data info dict (i.e. meta data), which is received from coordinator, into
            ``_data_result_queue``, and wait for ``get_data`` to retrieve. Wait for learner training and
            get learn info dict from ``_learn_info_queue``. If task is finished, join the learner thread and
            close the learner.
        Returns:
            - learn_info (:obj:`Any`): Learn info dict.
        """
        self._data_result_queue.put(data)
        learn_info = self._learn_info_queue.get()
        return learn_info

    def deal_with_learner_close(self) -> None:
        self._learner.close()
        self._learner_thread.join()
        del self._learner_thread
        self._learner = None
        self._policy_id = None

    # override
    def send_policy(self, state_dict: dict) -> None:
        """
        Overview:
            Save learner's policy in corresponding path, called by ``SendPolicyHook``.
        Arguments:
            - state_dict (:obj:`dict`): State dict of the policy.
        """
        if not os.path.exists(self._path_policy):
            os.mkdir(self._path_policy)
        path = self._policy_id
        if self._path_policy not in path:
            path = os.path.join(self._path_policy, path)
        setattr(self, "_latest_policy_path", path)
        save_file(path, state_dict, use_lock=True)

        if self._league_save_checkpoint_path is not None:
            save_file(self._league_save_checkpoint_path, state_dict, use_lock=True)

    @staticmethod
    def load_data_fn(path, meta: Dict[str, Any], decompressor: Callable) -> Any:
        """
        Overview:
            The function that is used to load data file.
        Arguments:
            - meta (:obj:`Dict[str, Any]`): Meta data info dict.
            - decompressor (:obj:`Callable`): Decompress function.
        Returns:
            - s (:obj:`Any`): Data which is read from file.
        """
        # Due to read-write conflict, read_file raise an error, therefore we set a while loop.
        while True:
            try:
                s = read_from_di_store(path) if read_from_di_store else read_file(path, use_lock=False)
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
                # add metadata key-value to stepdata
                for i in range(len(s)):
                    s[i].update(meta)
        else:
            s.update(meta)
        return s

    # override
    def get_data(self, batch_size: int) -> List[Callable]:
        """
        Overview:
            Get a list of data loading function, which can be implemented by dataloader to read data from files.
        Arguments:
            - batch_size (:obj:`int`): Batch size.
        Returns:
            - data (:obj:`List[Callable]`): A list of callable data loading function.
        """
        while self._learner is None:
            time.sleep(1)
        # Tell coordinator that we need training data, by putting info dict in data_demand_queue.
        assert self._data_demand_queue.qsize() == 0
        self._data_demand_queue.put({'batch_size': batch_size, 'cur_learner_iter': self._learner.last_iter.val})
        # Get a list of meta data (data info dict) from coordinator, by getting info dict from data_result_queue.
        data = self._data_result_queue.get()
        assert isinstance(data, list)
        assert len(data) == batch_size, '{}/{}'.format(len(data), batch_size)
        # Transform meta data to callable data loading function (partial ``load_data_fn``).
        decompressor = get_data_decompressor(data[0].get('compressor', 'none'))
        data = [
            partial(
                FlaskFileSystemLearner.load_data_fn,
                path=m['object_ref'] if read_from_di_store else os.path.join(self._path_data, m['data_id']),
                meta=m,
                decompressor=decompressor,
            ) for m in data
        ]
        return data

    # override
    def send_learn_info(self, learn_info: dict) -> None:
        """
        Overview:
            Store learn info dict in queue, which will be retrieved by callback function "deal_with_learner_learn"
            in learner slave, then will be sent to coordinator.
        Arguments:
            - learn_info (:obj:`dict`): Learn info in `dict` type. Keys are like 'learner_step', 'priority_info' \
                'finished_task', etc. You can refer to ``learn_info``(``worker/learner/base_learner.py``) for details.
        """
        assert self._learn_info_queue.qsize() == 0
        self._learn_info_queue.put(learn_info)

    @property
    def hooks4call(self) -> List[LearnerHook]:
        """
        Overview:
            Return the hooks that are related to message passing with coordinator.
        Returns:
            - hooks (:obj:`list`): The hooks which comm learner has. Will be registered in learner as well.
        """
        return [
            SendPolicyHook('send_policy', 100, position='before_run', ext_args={}),
            SendPolicyHook('send_policy', 100, position='after_iter', ext_args={'send_policy_freq': 1}),
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
            - ext_args (:obj:`dict`): Extended arguments. Use ``ext_args.freq`` to set send_policy_freq
        """
        super().__init__(*args, **kwargs)
        if 'send_policy_freq' in ext_args:
            self._freq = ext_args['send_policy_freq']
        else:
            self._freq = 1

    def __call__(self, engine: 'BaseLearner') -> None:  # noqa
        """
        Overview:
            Save learner's policy in corresponding path at interval iterations by calling ``engine``'s ``send_policy``.
            Saved file includes model_state_dict, learner_last_iter.
        Arguments:
            - engine (:obj:`BaseLearner`): The BaseLearner.

        .. note::
            Only rank == 0 learner will save policy.
        """
        last_iter = engine.last_iter.val
        if engine.rank == 0 and last_iter % self._freq == 0:
            state_dict = {'model': engine.policy.state_dict()['model'], 'iter': last_iter}
            engine.send_policy(state_dict)
            engine.debug('{} save iter{} policy'.format(engine.instance_name, last_iter))


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
            engine.debug('{} save iter{} learn_info'.format(engine.instance_name, last_iter))
