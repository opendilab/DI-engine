import logging
import pytest
from easydict import EasyDict
from copy import deepcopy
from ding.data import DequeBuffer
from ding.envs import BaseEnvManager
from ding.framework.context import BattleContext
from ding.framework.middleware import StepLeagueActor, LeagueCoordinator, LeagueLearnerExchanger, data_pusher, OffPolicyLearner
from ding.framework.middleware.tests.mock_for_test import DIStarMockPolicy, DIStarMockPolicyCollect, battle_inferencer_for_distar, battle_rolloutor_for_distar
from ding.framework.task import task, Parallel
from ding.league.v2 import BaseLeague
from dizoo.distar.config import distar_cfg
from dizoo.distar.envs.distar_env import DIStarEnv
from unittest.mock import patch

env_cfg = dict(
    actor=dict(job_type='train', ),
    env=dict(
        map_name='KingsCove',
        player_ids=['agent1', 'agent2'],
        races=['zerg', 'zerg'],
        map_size_resolutions=[True, True],  # if True, ignore minimap_resolutions
        minimap_resolutions=[[160, 152], [160, 152]],
        realtime=False,
        replay_dir='.',
        random_seed='none',
        game_steps_per_episode=100000,
        update_bot_obs=False,
        save_replay_episodes=1,
        update_both_obs=False,
        version='4.10.0',
    ),
)


def prepare_test():
    global distar_cfg, env_cfg
    env_cfg = EasyDict(env_cfg)
    cfg = deepcopy(distar_cfg)

    def env_fn():
        # subprocess env manager
        env = BaseEnvManager(
            env_fn=[lambda: DIStarEnv(env_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn():
        policy = DIStarMockPolicy(DIStarMockPolicy.default_config(), enable_field=['learn'])
        return policy

    def collect_policy_fn():
        policy = DIStarMockPolicyCollect()
        return policy

    return cfg, env_fn, policy_fn, collect_policy_fn


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg, env_fn, policy_fn, collect_policy_fn = prepare_test()
    league = BaseLeague(cfg.policy.other.league)
    N_PLAYERS = len(league.active_players_ids)
    print("League: n_players =", N_PLAYERS)

    with task.start(async_mode=True, ctx=BattleContext()),\
      patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar),\
      patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
        print("node id:", task.router.node_id)
        if task.router.node_id == 0:
            task.use(LeagueCoordinator(cfg, league))
        elif task.router.node_id <= N_PLAYERS:
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            player = league.active_players[task.router.node_id % N_PLAYERS]
            policy = policy_fn()
            task.use(LeagueLearnerExchanger(cfg, policy.learn_mode, player))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        else:
            task.use(StepLeagueActor(cfg, env_fn, collect_policy_fn))

        task.run()


if __name__ == "__main__":
    Parallel.runner(n_parallel_workers=4, protocol="tcp", topology="mesh")(main)
