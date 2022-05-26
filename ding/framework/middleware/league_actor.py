from ding.framework import task
from time import sleep
import logging

from typing import Dict, List, Any, Callable 
from dataclasses import dataclass, field
from abc import abstractmethod

from easydict import EasyDict
from ding.envs import BaseEnvManager

from .functional import inferencer, rolloutor, TransitionList

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext
    from ding.policy import Policy
    from ding.framework.middleware.league_learner import LearnerModel

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

@dataclass
class Job:
    launch_player: str
    players: List["PlayerMeta"]
    result: list = field(default_factory=list)
    job_no: int = 0  # Serial number of  job, not required
    train_iter: int = None
    is_eval: bool = False

# class Agent:
#     HAS_MODEL = False
#     HAS_TEACHER_MODEL = False
#     HAS_SUCCESSIVE_MODEL = False
#     def __init__(self, cfg=None, env_id=0):
#         pass

#     def reset(self, map_name, race, game_info, obs):
#         pass

#     def step(self, obs):
#         action = {'func_id': 0, 'skip_steps': 1, 
#         'queued': False, 'unit_tags': [0], 
#         'target_unit_tag': 0, 'location': [0, 0]}
#         return [action]


class LeagueActor:

    def __init__(self, cfg: EasyDict, env_fn: Callable, policy_fn: Callable):
        self._running = True
        self._model_updated = True

        self.cfg = cfg
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self._policies: Dict[str, "Policy.collect_function"] = {}
        self._inferencers: Dict[str, "inferencer"] = {}
        self._rolloutors: Dict[str, "rolloutor"] = {}
        # self._transitions = TransitionList(self.env.env_num)
        # self._inferencer = task.wrap(inferencer(cfg, policy, env))
        # self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

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

        policies = []
        main_player: "PlayerMeta" = None
        for player in job.players:
            policies.append(self._get_policy(player))
            if player.player_id == job.launch_player:
                main_player = player
                inferencer,rolloutor = self._get_collector(player.player_id)

        assert main_player, "Can not find active player"
        for p in self._policies:
            p.reset()
        
        # initialize env
        # self._envs.reset()

        # 参考StepCollector 
        ctx = OnlineRLContext()
        ctx.env_step = 0
        ctx.env_episode = 0
        ctx.train_iter = main_player.total_agent_step
        #TODO: what is ctx.collect_kwargs 

        old = ctx.env_step
        
        target_size = self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len

        while True:
            # model interaction with env
            # collector
            inferencer(ctx)
            rolloutor(ctx)
            if ctx.env_step - old >= target_size:
                ctx.trajectories, ctx.trajectory_end_idx = self._transitions.to_trajectories()
                self._transitions.clear()
                break

        # generate traj
        # get traj from collector

        # emit traj data 
        # rolloutor 实现
        
        def send_actor_job():
            """
            Q:When send actor job? 
            """
            raise NotImplementedError
        
        def send_actor_data():
            """
            Send actor traj for each iter.
            """
            raise NotImplementedError

        task.emit("actor_greeting", task.router.node_id)
        self._running = False
    
    def _get_collector(self, player_id: str):
        if self._inferencers.get(player_id) and self._rolloutors.get(player_id):
            return self._inferencers[player_id], self._rolloutors[player_id]
        cfg = self.cfg
        env = self.env_fn()
        policy = self._policies[player_id]

        self._inferencers[player_id] = task.wrap(inferencer(cfg, policy, env))
        self._rolloutors[player_id] = task.wrap(rolloutor(cfg, policy, env, self._transitions))
        return self._inferencers[player_id], self._rolloutors[player_id]

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



