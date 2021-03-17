import time
import sys
import copy
from typing import Optional, Union
from collections import defaultdict

from nervex.policy import create_policy
from nervex.utils import LimitedSpaceContainer, get_task_uid, build_logger, COMMANDER_REGISTRY
from .base_parallel_commander import register_parallel_commander, BaseCommander


@COMMANDER_REGISTRY.register('solo')
class SoloCommander(BaseCommander):
    r"""
    Overview:
        Parallel commander for solo games.
    Interface:
        __init__, get_actor_task, get_learner_task, finish_actor_task, finish_learner_task,
        notify_fail_actor_task, notify_fail_learner_task, get_learner_info
    """

    def __init__(self, cfg: dict) -> None:
        r"""
        Overview:
            Init the solo commander according to config.
        Arguments:
            - cfg (:obj:`dict`): Dict type config file.
        """
        self._cfg = cfg
        self._actor_task_space = LimitedSpaceContainer(0, cfg.actor_task_space)
        self._learner_task_space = LimitedSpaceContainer(0, cfg.learner_task_space)
        self._learner_info = [{'learner_step': 0}]
        self._evaluator_info = []
        self._current_buffer_id = None
        self._current_policy_id = None
        self._last_eval_time = 0
        # policy_cfg must be deepcopyed
        policy_cfg = copy.deepcopy(self._cfg.policy)
        self._policy = create_policy(policy_cfg, enable_field=['command']).command_mode
        self._logger, self._tb_logger = build_logger("./log/commander", "commander", need_tb=True)
        for tb_var in [
                'episode_count',
                'step_count',
                'avg_step_per_episode',
                'avg_time_per_step',
                'avg_time_per_episode',
                'reward_mean',
                'reward_std',
        ]:
            self._tb_logger.register_var('evaluator/' + tb_var)
        self._eval_step = -1
        self._end_flag = False

    def get_actor_task(self) -> Optional[dict]:
        r"""
        Overview:
            Return the new actor task when there is residual task space; Otherwise return None.
        Return:
            - task (:obj:`Optional[dict]`): New actor task.
        """
        if self._end_flag:
            return None
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
                'policy': copy.deepcopy(self._cfg.policy),
            }
        else:
            return None

    def get_learner_task(self) -> Optional[dict]:
        r"""
        Overview:
            Return the new learner task when there is residual task space; Otherwise return None.
        Return:
            - task (:obj:`Optional[dict]`): New learner task.
        """
        if self._end_flag:
            return None
        if self._learner_task_space.acquire_space():
            learner_cfg = self._cfg.learner_cfg
            learner_cfg.max_iterations = self._cfg.max_iterations
            return {
                'task_id': 'learner_task_{}'.format(get_task_uid()),
                'policy_id': self._init_policy_id(),
                'buffer_id': self._init_buffer_id(),
                'learner_cfg': learner_cfg,
                'replay_buffer_cfg': self._cfg.replay_buffer_cfg,
                'policy': copy.deepcopy(self._cfg.policy),
            }
        else:
            return None

    def finish_actor_task(self, task_id: str, finished_task: dict) -> bool:
        r"""
        Overview:
            Get actor's finish_task_info and release actor_task_space.
            If actor's task is evaluation, judge the convergence and return it.
        Arguments:
            - task_id (:obj:`str`): the actor task_id
            - finished_task (:obj:`dict`): the finished task
        Returns:
            - convergence (:obj:`bool`): Whether the stop val is reached and the algorithm is converged. \
                If True, the pipeline can be finished.
        """
        self._actor_task_space.release_space()
        if finished_task['eval_flag']:
            self._eval_step += 1
            self._last_eval_time = time.time()
            self._evaluator_info.append(finished_task)
            # TODO real train_iter from evaluator
            train_iter = self._eval_step
            info = {
                'train_iter': train_iter,
                'episode_count': finished_task['real_episode_count'],
                'step_count': finished_task['step_count'],
                'avg_step_per_episode': finished_task['avg_time_per_episode'],
                'avg_time_per_step': finished_task['avg_time_per_step'],
                'avg_time_per_episode': finished_task['avg_step_per_episode'],
                'reward_mean': finished_task['reward_mean'],
                'reward_std': finished_task['reward_std'],
            }
            self._logger.info(
                "[EVALUATOR]evaluate end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
            )
            tb_vars = [['evaluator/' + k, v, train_iter] for k, v in info.items() if k not in ['train_iter']]
            self._tb_logger.add_val_list(tb_vars, viz_type='scalar')
            eval_stop_val = self._cfg.actor_cfg.env_kwargs.eval_stop_val
            if eval_stop_val is not None and finished_task['reward_mean'] >= eval_stop_val:
                self._logger.info(
                    "[nerveX parallel pipeline] current eval_reward: {} is greater than the stop_val: {}".
                    format(finished_task['reward_mean'], eval_stop_val) + ", so the total training program is over."
                )
                self._end_flag = True
                return True
        return False

    def finish_learner_task(self, task_id: str, finished_task: dict) -> str:
        r"""
        Overview:
            Get learner's finish_task_info, release learner_task_space, reset corresponding variables.
        Arguments:
            - task_id (:obj:`str`): Learner task_id
            - finished_task (:obj:`dict`): Learner's finish_learn_info.
        Returns:
            - buffer_id (:obj:`str`): Buffer id of the finished learner.
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
            Release task space when actor task fails.
        """
        self._actor_task_space.release_space()

    def notify_fail_learner_task(self, task: dict) -> None:
        r"""
        Overview:
            Release task space when learner task fails.
        """
        self._learner_task_space.release_space()

    def get_learner_info(self, task_id: str, info: dict) -> None:
        r"""
        Overview:
            Append the info to learner_info:
        Arguments:
            - task_id (:obj:`str`): Learner task_id
            - info (:obj:`dict`): Dict type learner info.
        """
        self._learner_info.append(info)

    def _init_policy_id(self) -> str:
        r"""
        Overview:
            Init the policy id and return it.
        Returns:
            - policy_id (:obj:`str`): New initialized policy id.
        """
        policy_id = 'policy_{}'.format(get_task_uid())
        self._current_policy_id = policy_id
        return policy_id

    def _init_buffer_id(self) -> str:
        r"""
        Overview:
            Init the buffer id and return it.
        Returns:
            - buffer_id (:obj:`str`): New initialized buffer id.
        """
        buffer_id = 'buffer_{}'.format(get_task_uid())
        self._current_buffer_id = buffer_id
        return buffer_id
