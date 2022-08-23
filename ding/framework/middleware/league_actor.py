from typing import TYPE_CHECKING, Dict, Callable
from threading import Lock
import queue
from easydict import EasyDict
import time
from ditk import logging
import torch
import gc

from ding.policy import Policy
from ding.framework import task, EventEnum
from ding.framework.middleware import BattleStepCollector
from ding.framework.middleware.functional import ActorData, ActorDataMeta, PlayerModelInfo
from ding.league.player import PlayerMeta
from ding.utils.sparse_logging import log_every_sec

if TYPE_CHECKING:
    from ding.league.v2.base_league import Job
    from ding.framework import BattleContext
    from ding.framework.middleware.league_learner import LearnerModel


class StepLeagueActor:

    def __init__(self, cfg: EasyDict, env_fn: Callable, policy_fn: Callable):
        self.cfg = cfg
        self.env_fn = env_fn
        self.env_num = env_fn().env_num
        self.policy_fn = policy_fn
        self.unroll_len = self.cfg.policy.collect.unroll_len
        self._collectors: Dict[str, BattleStepCollector] = {}
        self.player_policy_dict: Dict[str, "Policy"] = {}
        self.player_policy_collect_dict: Dict[str, "Policy.collect_function"] = {}

        task.on(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=task.router.node_id), self._on_league_job)
        task.on(EventEnum.LEARNER_SEND_MODEL, self._on_learner_model)

        self.job_queue = queue.Queue()
        self.model_dict = {}
        self.model_dict_lock = Lock()
        self.model_info_dict = {}

        self.traj_num = 0
        self.total_time = 0
        self.total_episode_num = 0

        self.agent_num = 2

        self.collect_time = {}

        # self._gae_estimator = gae_estimator(cfg, policy_fn().collect_mode)

    def _on_learner_model(self, learner_model: "LearnerModel"):
        """
        If get newest learner model, put it inside model_queue.
        """
        log_every_sec(
            logging.INFO, 5, '[Actor {}] recieved model {}'.format(task.router.node_id, learner_model.player_id)
        )
        with self.model_dict_lock:
            self.model_dict[learner_model.player_id] = learner_model
            if self.model_info_dict.get(learner_model.player_id):
                self.model_info_dict[learner_model.player_id].get_new_model_time = time.time()
                self.model_info_dict[learner_model.player_id].update_new_model_time = None
            else:
                self.model_info_dict[learner_model.player_id] = PlayerModelInfo(
                    get_new_model_time=time.time(), update_new_model_time=None
                )

    def _on_league_job(self, job: "Job"):
        """
        Deal with job distributed by coordinator, put it inside job_queue.
        """
        self.job_queue.put(job)

    def _get_collector(self, player_id: str):
        if self._collectors.get(player_id):
            return self._collectors.get(player_id)
        cfg = self.cfg
        env = self.env_fn()
        collector = task.wrap(
            BattleStepCollector(
                cfg.policy.collect.collector, env, self.unroll_len, self.model_dict, self.model_info_dict,
                self.player_policy_collect_dict, self.agent_num
            )
        )
        self._collectors[player_id] = collector
        self.collect_time[player_id] = 0
        return collector

    def _get_policy(self, player: "PlayerMeta", duplicate: bool = False) -> "Policy.collect_function":
        player_id = player.player_id

        if self.player_policy_collect_dict.get(player_id):
            player_policy_collect_mode = self.player_policy_collect_dict.get(player_id)
            if duplicate is False:
                return player_policy_collect_mode
            else:
                player_policy = self.player_policy_dict.get(player_id)
                duplicate_policy: "Policy.collect_function" = self.policy_fn()
                del duplicate_policy._collect_model
                del duplicate_policy.teacher_model
                duplicate_policy._collect_model = player_policy._collect_model
                duplicate_policy.teacher_model = player_policy.teacher_model
                return duplicate_policy.collect_mode
        else:
            policy: "Policy.collect_function" = self.policy_fn()
            self.player_policy_dict[player_id] = policy

            policy_collect_mode = policy.collect_mode
            self.player_policy_collect_dict[player_id] = policy_collect_mode
            if "historical" in player.player_id:
                print(
                    '[Actor {}] recieved historical player {}, checkpoint.path is {}'.format(
                        task.router.node_id, player.player_id, player.checkpoint.path
                    )
                )
                policy_collect_mode.load_state_dict(player.checkpoint.load())

            return policy_collect_mode

    def _get_job(self):
        if self.job_queue.empty():
            task.emit(EventEnum.ACTOR_GREETING, task.router.node_id)
        job = None

        try:
            job = self.job_queue.get(timeout=10)
        except queue.Empty:
            logging.warning("[Actor {}] no Job got from coordinator.".format(task.router.node_id))

        return job

    def _get_current_policies(self, job):
        current_policies = []
        main_player: "PlayerMeta" = None
        player_set = set()
        for player in job.players:
            if player.player_id not in player_set:
                current_policies.append(self._get_policy(player, duplicate=False))
                player_set.add(player.player_id)
            else:
                current_policies.append(self._get_policy(player, duplicate=True))
            if player.player_id == job.launch_player:
                main_player = player
        assert main_player, "[Actor {}] can not find active player.".format(task.router.node_id)

        if current_policies is not None:
            #TODO(zms): make it more general, should we have the restriction of 1 policies
            # assert len(current_policies) > 1, "[Actor {}] battle collector needs more than 1 policies".format(
            #     task.router.node_id
            # )
            pass
        else:
            raise RuntimeError('[Actor {}] current_policies should not be None'.format(task.router.node_id))

        return main_player, current_policies

    def __call__(self, ctx: "BattleContext"):

        job = self._get_job()
        if job is None:
            return
        print('[Actor {}] recieve job {}'.format(task.router.node_id, job))
        log_every_sec(
            logging.INFO, 5, '[Actor {}] job of player {} begins.'.format(task.router.node_id, job.launch_player)
        )
        # TODO(zms): when get job, update the policies to the checkpoint in job
        ctx.player_id_list = [player.player_id for player in job.players]
        main_player_idx = [idx for idx, player in enumerate(job.players) if player.player_id == job.launch_player]
        self.agent_num = len(job.players)
        collector = self._get_collector(job.launch_player)

        main_player, ctx.current_policies = self._get_current_policies(job)

        ctx.n_episode = self.cfg.policy.collect.n_episode
        assert ctx.n_episode >= self.env_num, "[Actor {}] Please make sure n_episode >= env_num".format(
            task.router.node_id
        )

        ctx.n_episode = self.cfg.policy.collect.n_episode
        assert ctx.n_episode >= self.env_num, "Please make sure n_episode >= env_num"

        ctx.episode_info = [[] for _ in range(self.agent_num)]

        while True:
            time_begin = time.time()
            old_envstep = ctx.total_envstep_count
            collector(ctx)

            if ctx.job_finish is True:
                logging.info('[Actor {}] finish current job !'.format(task.router.node_id))

            for idx in main_player_idx:
                if not job.is_eval and len(ctx.trajectories_list[idx]) > 0:
                    trajectories = ctx.trajectories_list[idx]
                    self.traj_num += len(trajectories)
                    meta_data = ActorDataMeta(
                        player_total_env_step=ctx.total_envstep_count,
                        actor_id=task.router.node_id,
                        send_wall_time=time.time()
                    )
                    actor_data = ActorData(meta=meta_data, train_data=trajectories)
                    task.emit(EventEnum.ACTOR_SEND_DATA.format(player=job.launch_player), actor_data)

            ctx.trajectories_list = []

            time_end = time.time()
            self.total_time += time_end - time_begin
            log_every_sec(
                logging.INFO, 5,
                '[Actor {}] sent {} trajectories till now, total trajectory send speed is {} traj/s'.format(
                    task.router.node_id,
                    self.traj_num,
                    self.traj_num / self.total_time,
                )
            )

            self.collect_time[job.launch_player] += time_end - time_begin
            total_collect_speed = ctx.total_envstep_count / self.collect_time[job.launch_player] if self.collect_time[
                job.launch_player] != 0 else 0
            envstep_passed = ctx.total_envstep_count - old_envstep
            real_time_speed = envstep_passed / (time_end - time_begin)
            gc.collect()

            if ctx.job_finish is True:
                job.result = []
                for idx in main_player_idx:
                    for e in ctx.episode_info[idx]:
                        job.result.append(e['result'])
                task.emit(EventEnum.ACTOR_FINISH_JOB, job)
                ctx.episode_info = [[] for _ in range(self.agent_num)]
                logging.info('[Actor {}] job finish, send job\n'.format(task.router.node_id))
                break
        self.total_episode_num += ctx.env_episode
        logging.info(
            '[Actor {}] finish {} episodes till now, speed is {} episode/s'.format(
                task.router.node_id, self.total_episode_num, self.total_episode_num / self.total_time
            )
        )
        logging.info(
            '[Actor {}] sent {} trajectories till now, the episode trajectory speed is {} traj/episode'.format(
                task.router.node_id, self.traj_num, self.traj_num / self.total_episode_num
            )
        )


# class LeagueActor:

#     def __init__(self, cfg: EasyDict, env_fn: Callable, policy_fn: Callable):
#         self.cfg = cfg
#         self.env_fn = env_fn
#         self.env_num = env_fn().env_num
#         self.policy_fn = policy_fn
#         self.n_rollout_samples = self.cfg.policy.collect.n_rollout_samples
#         self._collectors: Dict[str, BattleEpisodeCollector] = {}
#         self.player_policy_collect_dict: Dict[str, "Policy.collect_function"] = {}
#         task.on(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=task.router.node_id), self._on_league_job)
#         task.on(EventEnum.LEARNER_SEND_MODEL, self._on_learner_model)
#         self.job_queue = queue.Queue()
#         self.model_dict = {}
#         self.model_dict_lock = Lock()

#         self.agent_num = 2
#         self.collect_time = {}

#     def _on_learner_model(self, learner_model: "LearnerModel"):
#         """
#         If get newest learner model, put it inside model_queue.
#         """
#         log_every_sec(logging.INFO, 5, "[Actor {}] receive model from learner \n".format(task.router.node_id))
#         with self.model_dict_lock:
#             self.model_dict[learner_model.player_id] = learner_model

#     def _on_league_job(self, job: "Job"):
#         """
#         Deal with job distributed by coordinator, put it inside job_queue.
#         """
#         self.job_queue.put(job)

#     def _get_collector(self, player_id: str):
#         if self._collectors.get(player_id):
#             return self._collectors.get(player_id)
#         cfg = self.cfg
#         env = self.env_fn()
#         collector = task.wrap(
#             BattleEpisodeCollector(
#                 cfg.policy.collect.collector, env, self.n_rollout_samples, self.model_dict, self.player_policy_collect_dict,
#                 self.agent_num
#             )
#         )
#         self._collectors[player_id] = collector
#         self.collect_time[player_id] = 0
#         return collector

#     def _get_policy(self, player: "PlayerMeta") -> "Policy.collect_function":
#         player_id = player.player_id
#         if self.player_policy_collect_dict.get(player_id):
#             return self.player_policy_collect_dict.get(player_id)
#         policy: "Policy.collect_function" = self.policy_fn().collect_mode
#         self.player_policy_collect_dict[player_id] = policy
#         if "historical" in player.player_id:
#             policy.load_state_dict(player.checkpoint.load())

#         return policy

#     def _get_job(self):
#         if self.job_queue.empty():
#             task.emit(EventEnum.ACTOR_GREETING, task.router.node_id)
#         job = None

#         try:
#             job = self.job_queue.get(timeout=10)
#         except queue.Empty:
#             logging.warning("[Actor {}] no Job got from coordinator".format(task.router.node_id))

#         return job

#     def _get_current_policies(self, job):
#         current_policies = []
#         main_player: "PlayerMeta" = None
#         for player in job.players:
#             current_policies.append(self._get_policy(player))
#             if player.player_id == job.launch_player:
#                 main_player = player
#         assert main_player, "[Actor {}] cannot find active player.".format(task.router.node_id)

#         if current_policies is not None:
#             assert len(current_policies) > 1, "battle collector needs more than 1 policies"
#             for p in current_policies:
#                 p.reset()
#         else:
#             raise RuntimeError('current_policies should not be None')

#         return main_player, current_policies

#     def __call__(self, ctx: "BattleContext"):

#         job = self._get_job()
#         if job is None:
#             return

#         ctx.player_id_list = [player.player_id for player in job.players]
#         self.agent_num = len(job.players)
#         collector = self._get_collector(job.launch_player)

#         main_player, ctx.current_policies = self._get_current_policies(job)

#         ctx.n_episode = self.cfg.policy.collect.n_episode
#         assert ctx.n_episode >= self.env_num, "[Actor {}] Please make sure n_episode >= env_num".format(
#             task.router.node_id
#         )

#         ctx.episode_info = [[] for _ in range(self.agent_num)]
#         ctx.remain_episode = ctx.n_episode
#         while True:
#             old_envstep = ctx.total_envstep_count
#             time_begin = time.time()
#             collector(ctx)
#             meta_data = ActorDataMeta(
#                 player_total_env_step=ctx.total_envstep_count, actor_id=task.router.node_id, send_wall_time=time.time()
#             )

#             if not job.is_eval and len(ctx.episodes[0]) > 0:
#                 actor_data = ActorData(meta=meta_data, train_data=ctx.episodes[0])
#                 task.emit(EventEnum.ACTOR_SEND_DATA.format(player=job.launch_player), actor_data)
#             ctx.episodes = []
#             time_end = time.time()
#             self.collect_time[job.launch_player] += time_end - time_begin
#             total_collect_speed = ctx.total_envstep_count / self.collect_time[job.launch_player] if self.collect_time[
#                 job.launch_player] != 0 else 0
#             envstep_passed = ctx.total_envstep_count - old_envstep
#             real_time_speed = envstep_passed / (time_end - time_begin)
#             log_every_sec(
#                 logging.INFO, 5,
#                 '[Actor {}] total_env_step:{}, current job env_step: {}, total_collect_speed: {} env_step/s, real-time collect speed: {} env_step/s'
#                 .format(
#                     task.router.node_id, ctx.total_envstep_count, ctx.env_step, total_collect_speed, real_time_speed
#                 )
#             )

#             if ctx.job_finish is True:
#                 job.result = [e['result'] for e in ctx.episode_info[0]]
#                 task.emit(EventEnum.ACTOR_FINISH_JOB, job)
#                 ctx.episode_info = [[] for _ in range(self.agent_num)]
#                 break
