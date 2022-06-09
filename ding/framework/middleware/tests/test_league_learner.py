from copy import deepcopy
from time import sleep
import torch
import pytest
import logging

from ding.envs import BaseEnvManager
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.framework import EventEnum
from ding.framework.task import task, Parallel
from ding.framework.middleware import LeagueLearner
from ding.framework.middleware.functional.actor_data import ActorData
from ding.framework.middleware.tests import cfg, MockLeague, MockLogger
from dizoo.league_demo.game_env import GameEnv

logging.getLogger().setLevel(logging.INFO)

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

def coordinator_mocker():
    task.on(EventEnum.LEARNER_SEND_META, lambda x: print("test:", x))
    task.on(EventEnum.LEARNER_SEND_MODEL, lambda x: print("test: send model success"))
    def _coordinator_mocker(ctx):
        sleep(1)
    return _coordinator_mocker

def actor_mocker(league):
    def _actor_mocker(ctx):
        sleep(1)
        obs_size = cfg.policy.model.obs_shape
        data = [{
            'obs': torch.rand(*obs_size),
            'next_obs': torch.rand(*obs_size),
            'done': False,
            'reward': torch.rand(1),
            'logit': torch.rand(1),
            'action': torch.randint(low=0, high=2, size=(1,)),
        } for _ in range(32)]
        actor_data = ActorData(env_step=0, train_data=data)
        n_players = len(league.active_players_ids)
        player = league.active_players[(task.router.node_id + 2) % n_players]
        print("actor player:", player.player_id)
        task.emit(EventEnum.ACTOR_SEND_DATA.format(player=player.player_id), actor_data)
    return _actor_mocker

def _main():
    cfg, env_fn, policy_fn = prepare_test()
    league = MockLeague(cfg.policy.other.league)

    with task.start():
        if task.router.node_id == 0:
            task.use(coordinator_mocker())
        elif task.router.node_id <= 2:
            task.use(actor_mocker(league))
        else:
            n_players = len(league.active_players_ids)
            player = league.active_players[task.router.node_id % n_players]
            learner = LeagueLearner(cfg, policy_fn, player)
            learner._learner._tb_logger = MockLogger()
            task.use(learner)
        task.run(max_step=3)


@pytest.mark.unittest
def test_league_learner():
    Parallel.runner(n_parallel_workers=5, protocol="tcp", topology="mesh")(_main)


if __name__ == '__main__':
    Parallel.runner(n_parallel_workers=5, protocol="tcp", topology="mesh")(_main)
