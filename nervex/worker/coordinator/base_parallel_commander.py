from abc import ABC, abstractmethod
from collections import defaultdict
from easydict import EasyDict
from nervex.utils import import_module


class BaseCommander(ABC):
    pass


class NaiveCommander(BaseCommander):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self.actor_task_space = cfg.actor_task_space
        self.learner_task_space = cfg.learner_task_space
        self.actor_task_count = 0
        self.learner_task_count = 0
        self._learner_info = defaultdict(list)
        self._learner_task_finish_count = 0
        self._actor_task_finish_count = 0

    def get_actor_task(self) -> dict:
        if self.actor_task_count < self.actor_task_space:
            self.actor_task_count += 1
            actor_cfg = self._cfg.actor_cfg
            actor_cfg.collect_setting = {'eps': 0.9}
            return {
                'task_id': 'actor_task_id{}'.format(self.actor_task_count),
                'buffer_id': 'test',
                'actor_cfg': actor_cfg,
                'policy': self._cfg.policy
            }
        else:
            return None

    def get_learner_task(self) -> dict:
        if self.learner_task_count < self.learner_task_space:
            self.learner_task_count += 1
            learner_cfg = self._cfg.learner_cfg
            learner_cfg.max_iterations = self._cfg.max_iterations
            return {
                'task_id': 'learner_task_id{}'.format(self.learner_task_count),
                'policy_id': 'test.pth',
                'buffer_id': 'test',
                'learner_cfg': learner_cfg,
                'policy': self._cfg.policy
            }
        else:
            return None

    def finish_actor_task(self, task_id: str, finished_task: dict) -> None:
        self._actor_task_finish_count += 1

    def finish_learner_task(self, task_id: str, finished_task: dict) -> None:
        self._learner_task_finish_count += 1
        return finished_task['buffer_id']

    def notify_fail_actor_task(self, task: dict) -> None:
        pass

    def notify_fail_learner_task(self, task: dict) -> None:
        pass

    def get_learner_info(self, task_id: str, info: dict) -> None:
        self._learner_info[task_id].append(info)


commander_map = {'naive': NaiveCommander}


def register_parallel_commander(name: str, commander: type) -> None:
    assert isinstance(name, str)
    assert issubclass(commander, BaseCommander)
    commander_map[name] = BaseCommander


def create_parallel_commander(cfg: dict) -> BaseCommander:
    cfg = EasyDict(cfg)
    import_module(cfg.import_names)
    commander_type = cfg.parallel_commander_type
    if commander_type not in commander_map:
        raise KeyError("not support parallel commander type: {}".format(commander_type))
    else:
        return commander_map[commander_type](cfg)
