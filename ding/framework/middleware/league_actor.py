from ding.framework import task
from time import sleep
import logging

from typing import Dict, List, Any, Callable 
from dataclasses import dataclass, field
from abc import abstractmethod

from easydict import EasyDict
from ding.envs import BaseEnvManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext
    from ding.league.v2.base_league import Job
    from ding.policy import Policy
    from ding.framework.middleware.league_learner import LearnerModel
    from ding.framework.middleware import BattleCollector

@dataclass
class ActorData:
    train_data: Any
    env_step: int = 0

class Storage:

    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def save(self, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError

@dataclass
class PlayerMeta:
    player_id: str
    checkpoint: "Storage"
    total_agent_step: int = 0

class LeagueActor:

    def __init__(self, cfg: EasyDict, env_fn: Callable, policy_fn: Callable):
        self.cfg = cfg
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        # self.n_rollout_samples = self.cfg.policy.collect.get("n_rollout_samples") or 0
        self.n_rollout_samples = 0
        self._running = False
        self._collectors: Dict[str, BattleCollector] = {}
        self._policies: Dict[str, "Policy.collect_function"] = {}
        self._model_updated = True
        task.on("league_job_actor_{}".format(task.router.node_id), self._on_league_job)
        task.on("learner_model", self._on_learner_model)

    def _on_learner_model(self, learner_model: "LearnerModel"):
        """
        If get newest learner model, update this actor's model.
        """
        player_meta = PlayerMeta(player_id=learner_model.player_id, checkpoint=None)
        policy = self._get_policy(player_meta)
        policy.load_state_dict(learner_model.state_dict)
        self._model_updated = True

        # update policy model

    def _on_league_job(self, job: "Job"):
        """
        Deal with job distributed by coordinator. Load historical model, generate traj and emit data.
        """
        self._running = True

        # Wait new active model for 10 seconds
        for _ in range(10):
            if self._model_updated:
                self._model_updated = False
                break
            logging.info(
                "Waiting for new model on actor: {}, player: {}".format(task.router.node_id, job.launch_player)
            )
            sleep(1)
        
        collector = self._get_collector(job.launch_player)
        policies = []
        main_player: "PlayerMeta" = None
        for player in job.players:
            policies.append(self._get_policy(player))
            if player.player_id == job.launch_player:
                main_player = player
                # inferencer,rolloutor = self._get_collector(player.player_id)

        assert main_player, "Can not find active player"
        collector.reset_policy(policies)

        def send_actor_job(episode_info: List):
            job.result = [e['result'] for e in episode_info]
            task.emit("actor_job", job)

        def send_actor_data(train_data: List):
            # Don't send data in evaluation mode
            if job.is_eval:
                return
            for d in train_data:
                d["adv"] = d["reward"]

            actor_data = ActorData(env_step=collector.envstep, train_data=train_data)
            task.emit("actor_data_player_{}".format(job.launch_player), actor_data)

        ctx = OnlineRLContext()
        ctx.n_episode = None
        ctx.train_iter = main_player.total_agent_step
        ctx.policy_kwargs = None
        
        train_data, episode_info = collector(ctx)
        train_data, episode_info = train_data[0], episode_info[0]  # only use main player data for training
        send_actor_data(train_data)
        send_actor_job(episode_info)
        
        task.emit("actor_greeting", task.router.node_id)
        self._running = False
    
    def _get_collector(self, player_id: str):
        if self._collectors.get(player_id):
            return self._collectors.get(player_id)
        cfg = self.cfg
        env = self.env_fn()
        collector = BattleCollector(
            cfg.policy.collect.collector,
            env
        )
        self._collectors[player_id] = collector
        return collector

    def _get_policy(self, player: "PlayerMeta") -> "Policy.collect_function":
        player_id = player.player_id
        if self._policies.get(player_id):
            return self._policies.get(player_id)
        policy: "Policy.collect_function" = self.policy_fn().collect_mode
        self._policies[player_id] = policy
        if "historical" in player.player_id:
            policy.load_state_dict(player.checkpoint.load())

        return policy

    def __call__(self):
        if not self._running:
            task.emit("actor_greeting", task.router.node_id)
        sleep(3)

# used for test
if __name__ == '__main__':
    actor = LeagueActor()



