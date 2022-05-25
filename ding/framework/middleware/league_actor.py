from ding.framework import task
from time import sleep
import logging

from typing import List, Any
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from abc import abstractmethod

from easydict import EasyDict
from ding.envs import BaseEnvManager

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

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager):
        self._running = True
        self._model_updated = True
        task.on("league_job_actor_{}".format(task.router.node_id), self._on_league_job)
        task.on("learner_model", self._on_learner_model)

    def _on_learner_model(self):
        """
        If get newest learner model, update this actor's model.
        """
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
        
        # initialize env
        self._envs.reset()

        # 参考StepCollector 

        # model interaction with env
        # collector

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
        self._envs.close()
        self._running = False

    def __call__(self):
        if not self._running:
            task.emit("actor_greeting", task.router.node_id)
        sleep(3)

# used for test
if __name__ == '__main__':
    actor = LeagueActor()



