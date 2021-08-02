from typing import Optional
import time
import copy

from ding.utils import deep_merge_dicts
from ding.policy import create_policy
from ding.utils import LimitedSpaceContainer, get_task_uid, build_logger, COMMANDER_REGISTRY
from ding.league import create_league, OneVsOneLeague
from .base_parallel_commander import BaseCommander


@COMMANDER_REGISTRY.register('one_vs_one')
class OneVsOneCommander(BaseCommander):
    r"""
    Overview:
        Parallel commander for battle games.
    Interface:
        __init__, get_collector_task, get_learner_task, finish_collector_task, finish_learner_task,
        notify_fail_collector_task, notify_fail_learner_task, get_learner_info
    """
    config = dict(
        collector_task_space=2,
        learner_task_space=1,
        eval_interval=60,
    )

    def __init__(self, cfg: dict) -> None:
        r"""
        Overview:
            Init the 1v1 commander according to config.
        Arguments:
            - cfg (:obj:`dict`): Dict type config file.
        """
        self._cfg = cfg
        self._exp_name = cfg.exp_name
        commander_cfg = self._cfg.policy.other.commander
        self._commander_cfg = commander_cfg

        self._collector_env_cfg = copy.deepcopy(self._cfg.env)
        self._collector_env_cfg.pop('collector_episode_num')
        self._collector_env_cfg.pop('evaluator_episode_num')
        self._collector_env_cfg.manager.episode_num = self._cfg.env.collector_episode_num
        self._evaluator_env_cfg = copy.deepcopy(self._cfg.env)
        self._evaluator_env_cfg.pop('collector_episode_num')
        self._evaluator_env_cfg.pop('evaluator_episode_num')
        self._evaluator_env_cfg.manager.episode_num = self._cfg.env.evaluator_episode_num

        self._collector_task_space = LimitedSpaceContainer(0, commander_cfg.collector_task_space)
        self._learner_task_space = LimitedSpaceContainer(0, commander_cfg.learner_task_space)
        self._learner_info = [{'learner_step': 0}]
        # TODO accumulate collect info
        self._collector_info = []
        self._total_collector_env_step = 0
        self._evaluator_info = []
        self._current_buffer_id = None
        self._current_policy_id = []  # 1v1 commander has multiple policies
        self._last_eval_time = 0
        # policy_cfg must be deepcopyed
        policy_cfg = copy.deepcopy(self._cfg.policy)
        self._policy = create_policy(policy_cfg, enable_field=['command']).command_mode
        self._logger, self._tb_logger = build_logger(
            "./{}/log/commander".format(self._exp_name), "commander", need_tb=True
        )
        self._collector_logger, _ = build_logger(
            "./{}/log/commander".format(self._exp_name), "commander_collector", need_tb=False
        )
        self._evaluator_logger, _ = build_logger(
            "./{}/log/commander".format(self._exp_name), "commander_evaluator", need_tb=False
        )
        self._sub_logger = {
            'collector': self._collector_logger,
            'evaluator': self._evaluator_logger,
        }
        self._end_flag = False

        # League
        path_policy = commander_cfg.path_policy
        self._path_policy = path_policy
        commander_cfg.league.path_policy = path_policy
        commander_cfg.league = deep_merge_dicts(OneVsOneLeague.default_config(), commander_cfg.league)
        self._league = create_league(commander_cfg.league)
        self._active_player = self._league.active_players[0]
        self._current_player_id = {}

    def get_collector_task(self) -> Optional[dict]:
        r"""
        Overview:
            Return the new collector task when there is residual task space; Otherwise return None.
        Return:
            - task (:obj:`Optional[dict]`): New collector task.
        """
        if self._end_flag:
            return None
        if self._collector_task_space.acquire_space():
            if self._current_buffer_id is None or len(self._current_policy_id) == 0:
                self._collector_task_space.release_space()
                return None
            cur_time = time.time()
            if cur_time - self._last_eval_time > self._commander_cfg.eval_interval:
                eval_flag = True
                self._last_eval_time = time.time()
            else:
                eval_flag = False
            collector_cfg = copy.deepcopy(self._cfg.policy.collect.collector)
            info = self._learner_info[-1]
            info['envstep'] = self._total_collector_env_step
            collector_cfg.collect_setting = self._policy.get_setting_collect(info)
            eval_or_collect = "EVALUATOR" if eval_flag else "COLLECTOR"
            task_id = '{}_task_{}'.format(eval_or_collect.lower(), get_task_uid())
            league_job_dict = self._league.get_job_info(self._active_player.player_id, eval_flag)
            # `self._current_player_id`: For eval, [id1, id2]; For collect, [id1].
            self._current_player_id[task_id] = league_job_dict['player_id']
            collector_cfg.policy_update_path = league_job_dict['checkpoint_path']
            collector_cfg.policy_update_flag = league_job_dict['player_active_flag']
            collector_cfg.eval_flag = eval_flag
            collector_cfg.exp_name = self._exp_name
            if eval_flag:
                collector_cfg.policy = copy.deepcopy([self._cfg.policy])
                collector_cfg.env = self._evaluator_env_cfg
                collector_cfg.env.eval_opponent = league_job_dict['eval_opponent']
            else:
                collector_cfg.policy = copy.deepcopy([self._cfg.policy for _ in range(2)])
                collector_cfg.env = self._collector_env_cfg
            collector_command = {
                'task_id': task_id,
                'buffer_id': self._current_buffer_id,
                'collector_cfg': collector_cfg,
            }
            # self._logger.info(
            #     "[{}] Task starts:\n{}".format(
            #         eval_or_collect, '\n'.join(
            #             [
            #                 '{}: {}'.format(k, v) for k, v in collector_command.items()
            #                 if k not in ['collector_cfg', 'policy']
            #             ]
            #         )
            #     )
            # )
            return collector_command
        else:
            # self._logger.info("[{}] Fails to start because of no launch space".format(eval_or_collect.upper()))
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
            learner_cfg = copy.deepcopy(self._cfg.policy.learn.learner)
            learner_cfg.exp_name = self._exp_name
            learner_command = {
                'task_id': 'learner_task_{}'.format(get_task_uid()),
                'policy_id': self._init_policy_id(),
                'buffer_id': self._init_buffer_id(),
                'learner_cfg': learner_cfg,
                'replay_buffer_cfg': self._cfg.policy.other.replay_buffer,
                'policy': copy.deepcopy(self._cfg.policy),
                'league_save_checkpoint_path': self._active_player.checkpoint_path,
            }
            # self._logger.info(
            #     "[LEARNER] Task starts:\n{}".format(
            #         '\n'.join(
            #             [
            #                 '{}: {}'.format(k, v) for k, v in learner_command.items()
            #                 if k not in ['learner_cfg', 'replay_buffer_cfg', 'policy']
            #             ]
            #         )
            #     )
            # )
            return learner_command
        else:
            # self._logger.info("[LEARNER] Fails to start because of no launch space")
            return None

    def finish_collector_task(self, task_id: str, finished_task: dict) -> bool:
        r"""
        Overview:
            Get collector's finish_task_info and release collector_task_space.
            If collector's task is evaluation, judge the convergence and return it.
        Arguments:
            - task_id (:obj:`str`): the collector task_id
            - finished_task (:obj:`dict`): the finished task
        Returns:
            - convergence (:obj:`bool`): Whether the stop val is reached and the algorithm is converged. \
                If True, the pipeline can be finished. It is only effective for an evaluator finish task.
        """
        self._collector_task_space.release_space()
        if finished_task['eval_flag']:
            self._evaluator_info.append(finished_task)
            # Evaluate difficulty increment
            wins, games = 0, 0
            game_result = finished_task['game_result']
            for i in game_result:
                for j in i:
                    if j == "wins":
                        wins += 1
                    games += 1
            eval_win = True if wins / games > 0.7 else False
            player_update_info = {
                'player_id': self._active_player.player_id,
                'eval_win': eval_win,
            }
            difficulty_inc = self._league.update_active_player(player_update_info)
            is_hardest = eval_win and not difficulty_inc
            # Print log
            train_iter = self._learner_info[-1]['learner_step']
            info = {
                'train_iter': train_iter,
                'episode_count': finished_task['real_episode_count'],
                'step_count': finished_task['step_count'],
                'avg_step_per_episode': finished_task['avg_time_per_episode'],
                'avg_time_per_step': finished_task['avg_time_per_step'],
                'avg_time_per_episode': finished_task['avg_step_per_episode'],
                'reward_mean': finished_task['reward_mean'],
                'reward_std': finished_task['reward_std'],
                'game_result': finished_task['game_result'],
                'eval_win': eval_win,
                'difficulty_inc': difficulty_inc,
            }
            self._sub_logger['evaluator'].info(
                "[EVALUATOR] Task ends:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
            )
            for k, v in info.items():
                if k in ['train_iter', 'game_result', 'eval_win', 'difficulty_inc']:
                    continue
                self._tb_logger.add_scalar('evaluator_iter/' + k, v, train_iter)
                self._tb_logger.add_scalar('evaluator_step/' + k, v, self._total_collector_env_step)
            # If evaluator task ends, whether to stop training should be judged.
            eval_stop_value = self._cfg.env.stop_value
            print('===', eval_stop_value)
            print('===', finished_task['reward_mean'])
            print('===', eval_win, difficulty_inc)
            if eval_stop_value is not None and finished_task['reward_mean'] >= eval_stop_value and is_hardest:
                self._logger.info(
                    "[DI-engine parallel pipeline] Current eval_reward: {} is greater than the stop_value: {}".
                    format(finished_task['reward_mean'], eval_stop_value) + ", so the total training program is over."
                )
                self._end_flag = True
                return True
        else:
            self._collector_info.append(finished_task)
            self._total_collector_env_step += finished_task['step_count']
            # If collector task ends, league payoff should be updated.
            payoff_update_dict = {
                'player_id': self._current_player_id.pop(task_id),
                'result': finished_task['game_result'],
            }
            self._league.finish_job(payoff_update_dict)
            # Print log
            train_iter = self._learner_info[-1]['learner_step']
            info = {
                'train_iter': train_iter,
                'episode_count': finished_task['real_episode_count'],
                'step_count': finished_task['step_count'],
                'avg_step_per_episode': finished_task['avg_time_per_episode'],
                'avg_time_per_step': finished_task['avg_time_per_step'],
                'avg_time_per_episode': finished_task['avg_step_per_episode'],
                'reward_mean': finished_task['reward_mean'],
                'reward_std': finished_task['reward_std'],
                'game_result': finished_task['game_result'],
            }
            self._sub_logger['collector'].info(
                "[COLLECTOR] Task ends:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
            )
            for k, v in info.items():
                if k in ['train_iter', 'game_result']:
                    continue
                self._tb_logger.add_scalar('collector_iter/' + k, v, train_iter)
                self._tb_logger.add_scalar('collector_step/' + k, v, self._total_collector_env_step)
            return False
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
        self._current_player_id = {}
        # self._logger.info("[LEARNER] Task ends.")
        return buffer_id

    def notify_fail_collector_task(self, task: dict) -> None:
        r"""
        Overview:
            Release task space when collector task fails.
        """
        self._collector_task_space.release_space()
        # self._logger.info("[COLLECTOR/EVALUATOR] Task fails.")

    def notify_fail_learner_task(self, task: dict) -> None:
        r"""
        Overview:
            Release task space when learner task fails.
        """
        self._learner_task_space.release_space()
        # self._logger.info("[LEARNER] Task fails.")

    def update_learner_info(self, task_id: str, info: dict) -> None:
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
        snapshot = self._league.judge_snapshot(self._active_player.player_id)
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
        self._current_buffer_id = buffer_id  # todo(why policy 2, buffer 1)
        # assert len(self._current_buffer_id) <= 2
        return buffer_id

    def increase_collector_task_space(self):
        r""""
        Overview:
        Increase task space when a new collector has added dynamically.
        """
        self._collector_task_space.increase_space()

    def decrease_collector_task_space(self):
        r""""
        Overview:
        Decrease task space when a new collector has removed dynamically.
        """
        self._collector_task_space.decrease_space()
