from ding.framework import task, EventEnum
from time import sleep
import logging

from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from abc import abstractmethod

from easydict import EasyDict
from ding.envs import BaseEnvManager

from ding.framework import BattleContext
from ding.league.v2.base_league import Job
from ding.policy import Policy
from ding.framework.middleware.league_learner import LearnerModel
from ding.framework.middleware import BattleCollector
from ding.framework.middleware.functional import policy_resetter
from ding.league.player import PlayerMeta
from threading import Lock
import queue

class LeagueActor:

    def __init__(self, cfg: EasyDict, env_fn: Callable, policy_fn: Callable):
        self.cfg = cfg
        self.env_fn = env_fn
        self.env_num = env_fn().env_num
        self.policy_fn = policy_fn
        # self.n_rollout_samples = self.cfg.policy.collect.get("n_rollout_samples") or 0
        self.n_rollout_samples = 64
        self._collectors: Dict[str, BattleCollector] = {}
        self.all_policies: Dict[str, "Policy.collect_function"] = {}
        task.on(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=task.router.node_id), self._on_league_job)
        task.on(EventEnum.LEARNER_SEND_MODEL, self._on_learner_model)
        self._policy_resetter = task.wrap(policy_resetter(self.env_num))
        self.job_queue = queue.Queue()
        self.model_dict = {}
        self.model_dict_lock = Lock()

    def _on_learner_model(self, learner_model: "LearnerModel"):
        """
        If get newest learner model, put it inside model_queue.
        """
        with self.model_dict_lock:
            self.model_dict[learner_model.player_id] = learner_model

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
        collector = task.wrap(BattleCollector(cfg.policy.collect.collector, env, self.n_rollout_samples, self.model_dict, self.all_policies))
        self._collectors[player_id] = collector
        return collector

    def _get_policy(self, player: "PlayerMeta") -> "Policy.collect_function":
        player_id = player.player_id
        if self.all_policies.get(player_id):
            return self.all_policies.get(player_id)
        policy: "Policy.collect_function" = self.policy_fn().collect_mode
        self.all_policies[player_id] = policy
        if "historical" in player.player_id:
            policy.load_state_dict(player.checkpoint.load())

        return policy

    def _get_job(self):
        if self.job_queue.empty():
            task.emit(EventEnum.ACTOR_GREETING, task.router.node_id)
        job = None

        try:
            job = self.job_queue.get(timeout=10)
        except queue.Empty:
            logging.warning("For actor_{}, no Job get from coordinator".format(task.router.node_id))
        
        return job 

    def _get_current_policies(self, job):
        current_policies = []
        main_player: "PlayerMeta" = None
        for player in job.players:
            current_policies.append(self._get_policy(player))
            if player.player_id == job.launch_player:
                main_player = player
        return main_player, current_policies


    def __call__(self, ctx: "BattleContext"):

        ctx.job = self._get_job()
        if ctx.job is None:
            return
        
        collector = self._get_collector(ctx.job.launch_player)

        main_player, ctx.current_policies = self._get_current_policies(ctx.job)
        assert main_player, "can not find active player, on actor: {}".format(task.router.node_id)

        self._policy_resetter(ctx)

        ctx.n_episode = None
        ctx.train_iter = main_player.total_agent_step
        ctx.collect_kwargs = None

        collector(ctx)


