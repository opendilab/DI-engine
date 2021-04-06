import time
import sys
from typing import Optional, Union
from collections import defaultdict
from functools import partial

from nervex.policy import create_policy
from nervex.utils import LimitedSpaceContainer, get_task_uid, build_logger, COMMANDER_REGISTRY
from nervex.league import create_league
from .base_parallel_commander import register_parallel_commander, BaseCommander


@COMMANDER_REGISTRY.register('one_vs_one')
class OneVsOneCommander(BaseCommander):
    r"""
    Overview:
        Parallel commander for battle games.
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

        # League
        self._league = create_league(cfg.league)
        self._active_player = self._league.active_players[0]
        self._current_player_id = None

        self._learner_info = [{'learner_step': 0}]
        self._evaluator_info = []
        self._current_buffer_id = None
        self._current_policy_id = []
        self._last_eval_time = 0
        self._policy = create_policy(self._cfg.policy, enable_field=['command']).command_mode
        self._logger, self._tb_logger = build_logger("./log/commander", "commander", need_tb=True)
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
            if self._current_buffer_id is None or len(self._current_policy_id) == 0:
                self._actor_task_space.release_space()
                return None
            cur_time = time.time()
            if cur_time - self._last_eval_time > self._cfg.eval_interval:
                eval_flag = True
            else:
                eval_flag = False
            actor_cfg = self._cfg.actor_cfg
            actor_cfg.collect_setting = self._policy.get_setting_collect(self._learner_info[-1])
            league_job_dict = self._league.get_job_info(self._active_player.player_id, eval_flag)
            self._current_player_id = league_job_dict['player_id']
            actor_cfg.policy_update_path = league_job_dict['checkpoint_path']
            actor_cfg.policy_update_flag = league_job_dict['player_active_flag']
            actor_cfg.eval_flag = eval_flag
            if eval_flag:
                policy = [self._cfg.policy]
                actor_cfg.env_kwargs.eval_opponent = league_job_dict['eval_opponent']
            else:
                policy = [self._cfg.policy for _ in range(2)]
            actor_command = {
                'task_id': 'actor_task_{}'.format(get_task_uid()),
                'buffer_id': self._current_buffer_id,
                'actor_cfg': actor_cfg,
                'policy': policy,
            }
            log_str = "EVALUATOR" if eval_flag else "ACTOR"
            self._logger.info(
                "[{}] Task starts:\n{}".format(
                    log_str, '\n'.join(
                        ['{}: {}'.format(k, v) for k, v in actor_command.items() if k not in ['actor_cfg', 'policy']]
                    )
                )
            )
            return actor_command
        else:
            # self._logger.info("[{}] Fails to start because of no launch space".format(log_str.upper()))
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
            learner_command = {
                'task_id': 'learner_task_{}'.format(get_task_uid()),
                'policy_id': self._init_policy_id(),
                'buffer_id': self._init_buffer_id(),
                'learner_cfg': learner_cfg,
                'replay_buffer_cfg': self._cfg.replay_buffer_cfg,
                'policy': self._cfg.policy,
                'league_save_checkpoint_path': self._active_player.checkpoint_path,
            }
            self._logger.info(
                "[LEARNER] Task starts:\n{}".format(
                    '\n'.join(
                        [
                            '{}: {}'.format(k, v) for k, v in learner_command.items()
                            if k not in ['learner_cfg', 'replay_buffer_cfg', 'policy']
                        ]
                    )
                )
            )
            return learner_command
        else:
            # self._logger.info("[LEARNER] Fails to start because of no launch space")
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
                If True, the pipeline can be finished. It is only effective for an evaluator finish task.
        """
        self._actor_task_space.release_space()
        if not finished_task['eval_flag']:
            # If actor task ends, league payoff should be updated.
            payoff_update_dict = {
                'player_id': self._current_player_id,
                'result': finished_task['game_result'],
            }
            self._league.finish_job(payoff_update_dict)  # issue
            self._logger.info("[ACTOR] Task ends")
        if finished_task['eval_flag']:
            # If evaluator task ends, whether to stop training should be judged.
            self._eval_step += 1
            self._last_eval_time = time.time()
            self._evaluator_info.append(finished_task)
            # Evaluate difficulty increment
            game_result = finished_task['game_result']
            wins, games = 0, 0
            for i in game_result:
                for j in i:
                    if j == "win":
                        wins += 1
                    games += 1
            eval_win = True if wins / games > 0.7 else False
            player_update_info = {
                'player_id': self._active_player.player_id,
                'eval_win': eval_win,
            }
            difficulty_inc = self._league.update_active_player(player_update_info)
            is_hardest = eval_win and difficulty_inc
            # Print log
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
                'game_result': game_result,
            }
            self._logger.info(
                "[EVALUATOR] Task ends:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
            )
            for k, v in info.items():
                if k in ['train_iter', 'game_result']:
                    continue
                self._tb_logger.add_scalar('evaluator/' + k, v, train_iter)
            eval_stop_value = self._cfg.actor_cfg.env_kwargs.eval_stop_value
            if eval_stop_value is not None and finished_task['reward_mean'] >= eval_stop_value and is_hardest:
                self._logger.info(
                    "[nerveX parallel pipeline] Current eval_reward: {} is greater than the stop_value: {}".
                    format(finished_task['reward_mean'], eval_stop_value) + ", so the total training program is over."
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
        self._current_policy_id = []
        self._learner_info = [{'learner_step': 0}]
        self._evaluator_info = []
        self._last_eval_time = 0
        self._logger.info("[LEARNER] Task ends.")
        return buffer_id

    def notify_fail_actor_task(self, task: dict) -> None:
        r"""
        Overview:
            Release task space when actor task fails.
        """
        self._actor_task_space.release_space()
        self._logger.info("[ACTOR/EVALUATOR] Task fails.")

    def notify_fail_learner_task(self, task: dict) -> None:
        r"""
        Overview:
            Release task space when learner task fails.
        """
        self._learner_task_space.release_space()
        self._logger.info("[LEARNER] Task fails.")

    def get_learner_info(self, task_id: str, info: dict) -> None:
        r"""
        Overview:
            Get learner info dict, use it to update commander record and league record.
        Arguments:
            - task_id (:obj:`str`): Learner task_id
            - info (:obj:`dict`): Dict type learner info.
        """
        self._learner_info.append(info)
        player_update_info = {
            'player_id': self._active_player.player_id,
            'train_iteration': info['learner_step'],
        }
        self._league.update_active_player(player_update_info)
        self._logger.info("[LEARNER] Update info at step {}".format(player_update_info['train_iteration']))
        snapshot = self._league.judge_snapshot(self._active_player.player_id)  # todo sequence of ckpt and snapshot
        if snapshot:
            self._logger.info(
                "[LEAGUE] Player {} snapshot at step {}".format(
                    player_update_info['player_id'], player_update_info['train_iteration']
                )
            )

    def _init_policy_id(self) -> str:
        r"""
        Overview:
            Init the policy id and return it.
        Returns:
            - policy_id (:obj:`str`): New initialized policy id.
        """
        policy_id = 'policy_{}'.format(get_task_uid())
        self._current_policy_id.append(policy_id)
        assert len(self._current_policy_id) <= 2
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
