from time import sleep
import pytest
from copy import deepcopy
from ding.envs import BaseEnvManager
from ding.framework.middleware.league_learner import LearnerModel
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware.league_actor import ActorData, LeagueActor, PlayerMeta
from ding.framework.storage import FileStorage

from ding.framework.task import task, Parallel
from ding.league.v2.base_league import Job
from ding.model import VAC
from ding.policy.ppo import PPOPolicy
from dizoo.league_demo.game_env import GameEnv

from dataclasses import dataclass

def prepare_test():
    global cfg
    cfg = deepcopy(cfg)

    def env_fn():
        env = BaseEnvManager(
            env_fn=[lambda: GameEnv(cfg.env.env_type) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn():
        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        return policy

    return cfg, env_fn, policy_fn


def test_league_actor():
    cfg, env_fn, policy_fn = prepare_test()
    policy = policy_fn()
    task.start()

    job = Job(
        launch_player='main_player_default_0', 
        players=[
            PlayerMeta(
                player_id='main_player_default_0', 
                checkpoint=FileStorage(path = None), 
                total_agent_step=0
            ), 
            PlayerMeta(
                player_id='main_player_default_1', 
                checkpoint=FileStorage(path = None), 
                total_agent_step=0)
            ]
    )
    ACTOR_ID = 0
    
    if task.router.node_id == ACTOR_ID:
        sleep(2)
        league_actor = LeagueActor(cfg, env_fn, policy_fn)
        league_actor()

    elif task.router.node_id == 1:
        
        testcases = {
            "on_actor_greeting": False,
            "on_actor_job": False,
            "on_actor_data": False,
        }

        def on_actor_greeting(actor_id):
            assert actor_id == ACTOR_ID
            task.emit("league_job_actor_{}".format(ACTOR_ID), job)
            testcases["on_actor_greeting"] = True
    
        def on_actor_job(job_: Job):
            assert job_.launch_player == job.launch_player
            testcases["on_actor_job"] = True

        
        def on_actor_data(actor_data):
            assert isinstance(actor_data, ActorData)
            testcases["on_actor_data"] = True
        
        task.on('actor_greeting', on_actor_greeting)
        task.on("actor_job", on_actor_job)
        task.on("actor_data_player_{}".format(job.launch_player), on_actor_data)

        sleep(0.3)
        # assert league_actor._model_updated == False

        task.emit(
            "learner_model",
            LearnerModel(
                player_id='main_player_default_0', state_dict=policy.learn_mode.state_dict(), train_iter=0
            )
        )
        sleep(5)
        # assert league_actor._model_updated == True

        try:
            print(testcases)
            assert all(testcases.values())
        finally:
            task.finish = True

if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=2, protocol="tcp", topology="mesh")(test_league_actor)