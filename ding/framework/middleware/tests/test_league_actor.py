from time import sleep
import pytest
from copy import deepcopy
from ding.envs import BaseEnvManager
from ding.framework.middleware.league_learner import LearnerModel
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware.league_actor import ActorData, LeagueActor

from ding.framework.task import task, Parallel
from ding.league.v2.base_league import BaseLeague, Job
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

    league = BaseLeague(cfg.policy.other.league)
    return cfg, env_fn, policy_fn, league


def main():
    cfg, env_fn, policy_fn, league = prepare_test()
    policy = policy_fn()
    league: BaseLeague
    task.start()

    job: Job = league.get_job_info()
    ACTOR_ID = 0

    def on_actor_greeting(actor_id):
        assert actor_id == ACTOR_ID
    
    def on_actor_job(job_: Job):
        assert job_.launch_player == job.launch_player
    
    def on_actor_data(actor_data):
        assert isinstance(actor_data, ActorData)
    
    if task.router.node_id == ACTOR_ID:
        league_actor = LeagueActor(cfg, env_fn, policy_fn)

    elif task.router.node_id == 1:
        task.on('actor_greeting', on_actor_greeting)
        task.on("actor_job", on_actor_job)
        task.on("actor_data_player_{}".format(job.launch_player), on_actor_data)

        sleep(0.3)
        task.emit("league_job_actor_{}".format(task.router.node_id), job)
        sleep(0.3)
        assert league_actor._model_updated == False

        task.emit(
            "learner_model",
            LearnerModel(
                player_id=league.active_players_ids[0], state_dict=policy.learn_mode.state_dict(), train_iter=0
            )
        )
        sleep(0.3)
        assert league_actor._model_updated == True

        task.finish = True

if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=2, protocol="tcp", topology="star")(main)