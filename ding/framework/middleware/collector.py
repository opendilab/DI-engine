from easydict import EasyDict
from typing import Dict, TYPE_CHECKING
import time
from ditk import logging

from ding.policy import get_random_policy
from ding.envs import BaseEnvManager
from ding.utils import log_every_sec
from ding.framework import task
from ding.framework.middleware.functional import PlayerModelInfo
from .functional import inferencer, rolloutor, TransitionList, BattleTransitionList, \
    battle_inferencer, battle_rolloutor

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, BattleContext

WAIT_MODEL_TIME = float('inf')


class BattleStepCollector:

    def __init__(
        self, cfg: EasyDict, env: BaseEnvManager, unroll_len: int, model_dict: Dict, model_info_dict: Dict,
        player_policy_collect_dict: Dict, agent_num: int
    ):
        self.cfg = cfg
        self.end_flag = False
        # self._reset(env)
        self.env = env
        self.env_num = self.env.env_num

        self.total_envstep_count = 0
        self.unroll_len = unroll_len
        self.model_dict = model_dict
        self.model_info_dict = model_info_dict
        self.player_policy_collect_dict = player_policy_collect_dict
        self.agent_num = agent_num

        self._battle_inferencer = task.wrap(battle_inferencer(self.cfg, self.env))
        self._transitions_list = [
            BattleTransitionList(self.env.env_num, self.unroll_len) for _ in range(self.agent_num)
        ]
        self._battle_rolloutor = task.wrap(
            battle_rolloutor(self.cfg, self.env, self._transitions_list, self.model_info_dict)
        )

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.env.close()

    def _update_policies(self, player_id_set) -> None:
        for player_id in player_id_set:
            # for this player, in the beginning of actor's lifetime, actor didn't recieve any new model, use initial model instead.
            if self.model_info_dict.get(player_id) is None:
                self.model_info_dict[player_id] = PlayerModelInfo(
                    get_new_model_time=time.time(), update_new_model_time=None
                )

        update_player_id_set = set()
        for player_id in player_id_set:
            if 'historical' not in player_id:
                update_player_id_set.add(player_id)
        while True:
            time_now = time.time()
            time_list = [time_now - self.model_info_dict[player_id].get_new_model_time for player_id in update_player_id_set]
            if any(x >= WAIT_MODEL_TIME for x in time_list):
                for index, player_id in enumerate(update_player_id_set):
                    if time_list[index] >= WAIT_MODEL_TIME:
                        #TODO: log_every_sec can only print the first model that not updated
                        log_every_sec(
                            logging.WARNING, 5,
                            'In actor {}, model for {} is not updated for {} senconds, and need new model'.format(
                                task.router.node_id, player_id, time_list[index]
                            )
                        )
                time.sleep(1)
            else:
                break

        for player_id in update_player_id_set:
            if self.model_dict.get(player_id) is None:
                continue
            else:
                learner_model = self.model_dict.get(player_id)
                policy = self.player_policy_collect_dict.get(player_id)
                assert policy, "for player{}, policy should have been initialized already"
                # update policy model
                policy.load_state_dict(learner_model.state_dict)
                self.model_info_dict[player_id].update_new_model_time = time.time()
                self.model_info_dict[player_id].update_train_iter = learner_model.train_iter
                self.model_dict[player_id] = None

    def __call__(self, ctx: "BattleContext") -> None:
        """
        Input of ctx:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - collect_kwargs (:obj:`dict`): the keyword args for policy forward
        Output of ctx:
            -  ctx.train_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
        """
        ctx.total_envstep_count = self.total_envstep_count
        old = ctx.env_step

        while True:
            if self.env.closed:
                self.env.launch()
                # TODO(zms): only runnable when 1 actor has exactly one env, need to write more general
                for policy_id, policy in enumerate(ctx.current_policies):
                    policy.reset(self.env.ready_obs[0][policy_id])
            self._update_policies(set(ctx.player_id_list))
            try:
                self._battle_inferencer(ctx)
                self._battle_rolloutor(ctx)
            except Exception as e:
                # TODO(zms): need to handle the exception cleaner
                logging.error("[Actor {}] got an exception: {} when collect data".format(task.router.node_id, e))
                self.env.close()
                for env_id in range(self.env_num):
                    for policy_id, policy in enumerate(ctx.current_policies):
                        self._transitions_list[policy_id].clear_newest_episode(env_id, before_append=True)

            self.total_envstep_count = ctx.total_envstep_count

            only_finished = True if ctx.env_episode >= ctx.n_episode else False
            if (self.unroll_len > 0 and ctx.env_step - old >= self.unroll_len) or ctx.env_episode >= ctx.n_episode:
                for transitions in self._transitions_list:
                    trajectories = transitions.to_trajectories(only_finished=only_finished)
                    ctx.trajectories_list.append(trajectories)
                if ctx.env_episode >= ctx.n_episode:
                    self.env.close()
                    ctx.job_finish = True
                    for transitions in self._transitions_list:
                        transitions.clear()
                break


# class BattleEpisodeCollector:

#     def __init__(
#         self, cfg: EasyDict, env: BaseEnvManager, n_rollout_samples: int, model_dict: Dict, player_policy_collect_dict: Dict,
#         agent_num: int
#     ):
#         self.cfg = cfg
#         self.end_flag = False
#         # self._reset(env)
#         self.env = env
#         self.env_num = self.env.env_num

#         self.total_envstep_count = 0
#         self.n_rollout_samples = n_rollout_samples
#         self.model_dict = model_dict
#         self.player_policy_collect_dict = player_policy_collect_dict
#         self.agent_num = agent_num

#         self._battle_inferencer = task.wrap(battle_inferencer(self.cfg, self.env))
#         self._transitions_list = [TransitionList(self.env.env_num) for _ in range(self.agent_num)]
#         self._battle_rolloutor = task.wrap(battle_rolloutor(self.cfg, self.env, self._transitions_list))

#     def __del__(self) -> None:
#         """
#         Overview:
#             Execute the close command and close the collector. __del__ is automatically called to \
#                 destroy the collector instance when the collector finishes its work
#         """
#         if self.end_flag:
#             return
#         self.end_flag = True
#         self.env.close()

#     def _update_policies(self, player_id_list) -> None:
#         for player_id in player_id_list:
#             if self.model_dict.get(player_id) is None:
#                 continue
#             else:
#                 learner_model = self.model_dict.get(player_id)
#                 policy = self.player_policy_collect_dict.get(player_id)
#                 assert policy, "for player {}, policy should have been initialized already".format(player_id)
#                 # update policy model
#                 policy.load_state_dict(learner_model.state_dict)
#                 self.model_dict[player_id] = None

#     def __call__(self, ctx: "BattleContext") -> None:
#         """
#         Input of ctx:
#             - n_episode (:obj:`int`): the number of collecting data episode
#             - train_iter (:obj:`int`): the number of training iteration
#             - collect_kwargs (:obj:`dict`): the keyword args for policy forward
#         Output of ctx:
#             -  ctx.train_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
#                 the former is a list containing collected episodes if not get_train_sample, \
#                 otherwise, return train_samples split by unroll_len.
#         """
#         ctx.total_envstep_count = self.total_envstep_count
#         old = ctx.env_episode
#         while True:
#             if self.env.closed:
#                 self.env.launch()
#             self._update_policies(ctx.player_id_list)
#             self._battle_inferencer(ctx)
#             self._battle_rolloutor(ctx)

#             self.total_envstep_count = ctx.total_envstep_count

#             if (self.n_rollout_samples > 0
#                     and ctx.env_episode - old >= self.n_rollout_samples) or ctx.env_episode >= ctx.n_episode:
#                 for transitions in self._transitions_list:
#                     ctx.episodes.append(transitions.to_episodes())
#                     transitions.clear()
#                 if ctx.env_episode >= ctx.n_episode:
#                     self.env.close()
#                     ctx.job_finish = True
#                 break


class StepCollector:
    """
    Overview:
        The class of the collector running by steps, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
            - random_collect_size (:obj:`int`): The count of samples that will be collected randomly, \
                typically used in initial runs.
        """
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = TransitionList(self.env.env_num)
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing \
                the target number of steps.
        Input of ctx:
            - env_step (:obj:`int`): The env steps which will increase during collection.
        """
        old = ctx.env_step
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            # compatible with old config, a train sample = unroll_len step
            target_size = self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_step - old >= target_size:
                ctx.trajectories, ctx.trajectory_end_idx = self._transitions.to_trajectories()
                self._transitions.clear()
                break


class EpisodeCollector:
    """
    Overview:
        The class of the collector running by episodes, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
            - random_collect_size (:obj:`int`): The count of samples that will be collected randomly, \
                typically used in initial runs.
        """
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = TransitionList(self.env.env_num)
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing the \
                target number of episodes.
        Input of ctx:
            - env_episode (:obj:`int`): The env env_episode which will increase during collection.
        """
        old = ctx.env_episode
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            target_size = self.cfg.policy.collect.n_episode
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_episode - old >= target_size:
                ctx.episodes = self._transitions.to_episodes()
                self._transitions.clear()
                break


# TODO battle collector
