from time import sleep
import pytest
from copy import deepcopy
from ding.envs import BaseEnvManager
from ding.framework.middleware.league_learner import LearnerModel
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware.league_actor import ActorData, LeagueActor, Job

from ding.framework.task import Task
from ding.model import VAC
from ding.policy.ppo import PPOPolicy
from dizoo.league_demo.game_env import GameEnv

class MockLeague:
    def __init__(self):
        self.active_players_ids = ["test_node_1", "test_node_2", "test_node_3"]
        self.update_payoff_cnt = 0
        self.update_active_player_cnt = 0
        self.create_historical_player_cnt = 0
        self.get_job_info_cnt = 0

    def update_payoff(self, job):
        self.update_payoff_cnt += 1
        # print("update_payoff: {}".format(job))
    
    def update_active_player(self, meta):
        self.update_active_player_cnt += 1
        # print("update_active_player: {}".format(meta))
    
    def create_historical_player(self, meta):
        self.create_historical_player_cnt += 1
        # print("create_historical_player: {}".format(meta))
    
    def get_job_info(self, player_id):
        self.get_job_info_cnt += 1
        # print("get_job_info")
        return Job(231890, player_id, False)

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
    cfg, env_fn, policy_fn = prepare_test()
    task.start()
    
    if task.router.node_id == 0:
        LeagueActor()
    elif task.router.node_id == 1:
        pass

if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="star")(main)