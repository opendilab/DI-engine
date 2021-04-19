from abc import ABC, abstractmethod
from collections import defaultdict
from easydict import EasyDict
import copy

from nervex.utils import import_module, COMMANDER_REGISTRY
from nervex.league import create_league


class BaseCommander(ABC):
    r"""
    Overview:
        Base parallel commander abstract class.
    Interface:
        get_collector_task
    """

    @abstractmethod
    def get_collector_task(self) -> dict:
        raise NotImplementedError

    def judge_collector_finish(self, task_id: str, info: dict) -> bool:
        collector_done = info.get('collector_done', False)
        cur_episode = info['cur_episode']
        cur_sample = info['cur_sample']
        if collector_done:
            return True
        return False

    def judge_learner_finish(self, task_id: str, info: dict) -> bool:
        learner_done = info.get('learner_done', False)
        cur_step = info['learner_step']
        if learner_done:
            return True
        return False


@COMMANDER_REGISTRY.register('naive')
class NaiveCommander(BaseCommander):
    r"""
    Overview:
        A naive implementation of parallel commander.
    Interface:
        __init__, get_collector_task, get_learner_task, finsh_collector_task, finish_learner_task,
        notify_fail_collector_task, notify_fail_learner_task, get_learner_info
    """

    def __init__(self, cfg: dict) -> None:
        r"""
        Overview:
            Init the naive commander according to config
        Arguments:
            - cfg (:obj:`dict`): The config to init commander. Should include \
                "collector_task_space" and "learner_task_space".
        """
        self._cfg = cfg
        self.collector_task_space = cfg.collector_task_space
        self.learner_task_space = cfg.learner_task_space
        self.collector_task_count = 0
        self.learner_task_count = 0
        self._learner_info = defaultdict(list)
        self._learner_task_finish_count = 0
        self._collector_task_finish_count = 0

    def get_collector_task(self) -> dict:
        r"""
        Overview:
            Get a new collector task when ``collector_task_count`` is smaller than ``collector_task_space``.
        Return:
            - task (:obj:`dict`): New collector task.
        """
        if self.collector_task_count < self.collector_task_space:
            self.collector_task_count += 1
            collector_cfg = self._cfg.collector_cfg
            collector_cfg.collect_setting = {'eps': 0.9}
            collector_cfg.eval_flag = False
            return {
                'task_id': 'collector_task_id{}'.format(self.collector_task_count),
                'buffer_id': 'test',
                'collector_cfg': collector_cfg,
                'policy': copy.deepcopy(self._cfg.policy),
            }
        else:
            return None

    def get_learner_task(self) -> dict:
        r"""
        Overview:
            Get the new learner task when task_count is less than task_space
        Return:
            - task (:obj:`dict`): the new learner task
        """
        if self.learner_task_count < self.learner_task_space:
            self.learner_task_count += 1
            learner_cfg = self._cfg.learner_cfg
            learner_cfg.max_iterations = self._cfg.max_iterations
            return {
                'task_id': 'learner_task_id{}'.format(self.learner_task_count),
                'policy_id': 'test.pth',
                'buffer_id': 'test',
                'learner_cfg': learner_cfg,
                'replay_buffer_cfg': self._cfg.replay_buffer_cfg,
                'policy': copy.deepcopy(self._cfg.policy),
            }
        else:
            return None

    def finish_collector_task(self, task_id: str, finished_task: dict) -> None:
        r"""
        Overview:
            finish collector task will add the collector_task_finish_count
        """
        self._collector_task_finish_count += 1

    def finish_learner_task(self, task_id: str, finished_task: dict) -> str:
        r"""
        Overview:
            finish learner task will add the learner_task_finish_count and get the buffer_id of task to close the buffer
        Return:
            the finished_task buffer_id
        """
        self._learner_task_finish_count += 1
        return finished_task['buffer_id']

    def notify_fail_collector_task(self, task: dict) -> None:
        r"""
        Overview:
            naive coordinator will pass when need to notify_fail_collector_task
        """
        pass

    def notify_fail_learner_task(self, task: dict) -> None:
        r"""
        Overview:
            naive coordinator will pass when need to notify_fail_learner_task
        """
        pass

    def get_learner_info(self, task_id: str, info: dict) -> None:
        r"""
        Overview:
            append the info to learner:
        Arguments:
            - task_id (:obj:`str`): the learner task_id
            - info (:obj:`dict`): the info to append to learner
        """
        self._learner_info[task_id].append(info)


def create_parallel_commander(cfg: dict) -> BaseCommander:
    r"""
    Overview:
        create the commander according to cfg
    Arguments:
        - cfg (:obj:`dict`): the commander cfg to create, should include import_names and parallel_commander_type
    """
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return COMMANDER_REGISTRY.build(cfg.parallel_commander_type, cfg=cfg)
