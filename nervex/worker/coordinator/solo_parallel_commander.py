import time
import sys
from typing import Union
from collections import defaultdict

from nervex.policy import create_policy
from nervex.utils import LimitedSpaceContainer, get_task_uid
from .base_parallel_commander import register_parallel_commander, BaseCommander


class SoloCommander(BaseCommander):
    r"""
    Overview:
        the solo type commander
    Interface:
        __init__, get_actor_task, get_learner_task, finish_actor_task, finish_learner_task, \
            notify_fail_actor_task, notify_fail_learner_task, get_learner_info
    """

    def __init__(self, cfg: dict) -> None:
        r"""
        Overview:
            init the solo commander according to config
        Arguments:
            - cfg (:obj:`dict`): the config file of solo commander
        """
        self._cfg = cfg
        self._actor_task_space = LimitedSpaceContainer(0, cfg.actor_task_space)
        self._learner_task_space = LimitedSpaceContainer(0, cfg.learner_task_space)
        self._learner_info = [{'learner_step': 0}]
        self._evaluator_info = []
        self._current_buffer_id = None
        self._current_policy_id = None
        self._last_eval_time = 0
        self._policy = create_policy(self._cfg.policy, enable_field=['command']).command_mode

    def get_actor_task(self) -> Union[None, dict]:
        r"""
        Overview:
            Get the new actor task when there is space
        Return:
            - task (:obj:`dict`): the new actor task
        """
        if self._actor_task_space.acquire_space():
            if self._current_buffer_id is None or self._current_policy_id is None:
                self._actor_task_space.release_space()
                return None
            cur_time = time.time()
            if cur_time - self._last_eval_time > self._cfg.eval_interval:
                eval_flag = True
            else:
                eval_flag = False
            actor_cfg = self._cfg.actor_cfg
            actor_cfg.collect_setting = self._policy.get_setting_collect(self._learner_info[-1])
            actor_cfg.policy_update_path = self._current_policy_id
            actor_cfg.eval_flag = eval_flag
            return {
                'task_id': 'actor_task_{}'.format(get_task_uid()),
                'buffer_id': self._current_buffer_id,
                'actor_cfg': actor_cfg,
                'policy': self._cfg.policy,
            }
        else:
            return None

    def get_learner_task(self) -> Union[None, dict]:
        r"""
        Overview:
            Get the new learner task when there is space
        Return:
            - task (:obj:`dict`): the new learner task
        """
        if self._learner_task_space.acquire_space():
            learner_cfg = self._cfg.learner_cfg
            learner_cfg.max_iterations = self._cfg.max_iterations
            return {
                'task_id': 'learner_task_{}'.format(get_task_uid()),
                'policy_id': self._init_policy_id(),
                'buffer_id': self._init_buffer_id(),
                'learner_cfg': learner_cfg,
                'replay_buffer_cfg': self._cfg.replay_buffer_cfg,
                'policy': self._cfg.policy,
            }
        else:
            return None

    def finish_actor_task(self, task_id: str, finished_task: dict) -> bool:
        r"""
        Overview:
            finish the actor task and release space
        Arguments:
            - task_id (:obj:`str`): the actor task_id
            - finished_task (:obj:`dict`): the finished task
        Returns:
            - converge (:obj:`bool`): whether the stop val is reached and the algorithm is converged
        """
        self._actor_task_space.release_space()
        if finished_task['eval_flag']:
            self._last_eval_time = time.time()
            self._evaluator_info.append(finished_task)
            eval_stop_val = self._cfg.actor_cfg.env_kwargs.eval_stop_val
            if eval_stop_val is not None and finished_task['reward_mean'] >= eval_stop_val:
                print(
                    "[nerveX parallel pipeline] current eval_reward: {} is greater than the stop_val: {}, so the total \
                    training program is over.\nPress CTRL+C to quit"
                    .format(finished_task['reward_mean'], eval_stop_val)
                )
                return True
        return False

    def finish_learner_task(self, task_id: str, finished_task: dict) -> str:
        r"""
        Overview:
            finish the learner task and release space
        Arguments:
            - task_id (:obj:`str`): the learner task_id
            - finished_task (:obj:`dict`): the finished task
        Returns:
            - buffer_id (:obj:`str`): the buffer_id of the finished learner
        """
        self._learner_task_space.release_space()
        buffer_id = finished_task['buffer_id']
        self._current_buffer_id = None
        self._current_policy_id = None
        self._learner_info = [{'learner_step': 0}]
        self._evaluator_info = []
        self._last_eval_time = 0
        return buffer_id

    def notify_fail_actor_task(self, task: dict) -> None:
        r"""
        Overview:
            release space when actor task failed
        """
        self._actor_task_space.release_space()

    def notify_fail_learner_task(self, task: dict) -> None:
        r"""
        Overview:
            release space when learner task failed
        """
        self._learner_task_space.release_space()

    def get_learner_info(self, task_id: str, info: dict) -> None:
        r"""
        Overview:
            append the info to learner:
        Arguments:
            - task_id (:obj:`str`): the learner task_id
            - info (:obj:`dict`): the info to append to learner
        """
        self._learner_info.append(info)

    def _init_policy_id(self) -> str:
        r"""
        Overview:
            init the policy id
        Returns:
            - policy_id (:obj:`str`): the policy id uesd
        """
        policy_id = 'policy_{}'.format(get_task_uid())
        self._current_policy_id = policy_id
        return policy_id

    def _init_buffer_id(self) -> str:
        r"""
        Overview:
            init the buffer id
        Returns:
            - buffer_id (:obj:`str`): the buffer id uesd
        """
        buffer_id = 'buffer_{}'.format(get_task_uid())
        self._current_buffer_id = buffer_id
        return buffer_id


register_parallel_commander('solo', SoloCommander)
