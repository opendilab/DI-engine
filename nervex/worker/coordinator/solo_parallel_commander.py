from typing import Union
from collections import defaultdict
from nervex.utils import LimitedSpaceContainer, get_task_uid
from .base_parallel_commander import register_parallel_commander, BaseCommander


class SoloCommander(BaseCommander):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._actor_task_space = LimitedSpaceContainer(0, cfg.actor_task_space)
        self._learner_task_space = LimitedSpaceContainer(0, cfg.learner_task_space)
        self._learner_info = defaultdict(list)
        self._current_buffer_id = None
        self._current_policy_id = None

    def get_actor_task(self) -> Union[None, dict]:
        if self._actor_task_space.acquire_space():
            if self._current_buffer_id is None or self._current_policy_id is None:
                return None
            actor_cfg = self._cfg.actor_cfg
            actor_cfg.collect_setting = {'eps': 0.9}
            actor_cfg.policy_update_path = self._current_policy_id
            return {
                'task_id': 'actor_task_{}'.format(get_task_uid()),
                'buffer_id': self._current_buffer_id,
                'actor_cfg': actor_cfg,
                'policy': self._cfg.policy,
            }
        else:
            return None

    def get_learner_task(self) -> Union[None, dict]:
        if self._learner_task_space.acquire_space():
            learner_cfg = self._cfg.learner_cfg
            learner_cfg.max_iterations = self._cfg.max_iterations
            return {
                'task_id': 'learner_task_{}'.format(get_task_uid()),
                'policy_id': self._init_policy_id(),
                'buffer_id': self._init_buffer_id(),
                'learner_cfg': learner_cfg,
                'policy': self._cfg.policy,
            }
        else:
            return None

    def finish_actor_task(self, task_id: str, finished_task: dict) -> None:
        self._actor_task_space.release_space()

    def finish_learner_task(self, task_id: str, finished_task: dict) -> str:
        self._learner_task_space.release_space()
        self._learner_info = defaultdict(list)
        buffer_id = finished_task['buffer_id']
        self._current_buffer_id = None
        self._current_policy_id = None
        return buffer_id

    def notify_fail_actor_task(self, task: dict) -> None:
        self._actor_task_space.release_space()

    def notify_fail_learner_task(self, task: dict) -> None:
        self._learner_task_space.release_space()
        self._learner_info = defaultdict(list)

    def get_learner_info(self, task_id: str, info: dict) -> None:
        self._learner_info[task_id].append(info)

    def _init_policy_id(self) -> str:
        policy_id = 'policy_{}'.format(get_task_uid())
        self._current_policy_id = policy_id
        return policy_id

    def _init_buffer_id(self) -> str:
        buffer_id = 'buffer_{}'.format(get_task_uid())
        self._current_buffer_id = buffer_id
        return buffer_id


register_parallel_commander('solo', SoloCommander)
