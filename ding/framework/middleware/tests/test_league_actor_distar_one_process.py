from time import sleep
import pytest
from copy import deepcopy
from unittest.mock import patch
from easydict import EasyDict

from dizoo.distar.config import distar_cfg
from dizoo.distar.envs.distar_env import DIStarEnv

from ding.envs import EnvSupervisor
from ding.league.player import PlayerMeta
from ding.league.v2.base_league import Job
from ding.framework import EventEnum
from ding.framework.storage import FileStorage
from ding.framework.task import task
from ding.framework.context import BattleContext

from ding.framework.supervisor import ChildType
from ding.framework.middleware import StepLeagueActor
from ding.framework.middleware.functional import ActorData
from ding.framework.middleware.tests import DIStarMockPolicy
from ding.framework.middleware.functional.collector import battle_inferencer_for_distar, battle_rolloutor_for_distar

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
env_cfg = EasyDict(env_cfg)
cfg = deepcopy(distar_cfg)


class PrepareTest():

    @classmethod
    def get_env_fn(cls):
        return DIStarEnv(env_cfg)

    @classmethod
    def get_env_supervisor(cls):
        for _ in range(10):
            try:
                env = EnvSupervisor(
                    type_=ChildType.THREAD,
                    env_fn=[cls.get_env_fn for _ in range(cfg.env.collector_env_num)],
                    **cfg.env.manager
                )
                env.seed(cfg.seed)
                return env
            except Exception as e:
                print(e)
                continue

    @classmethod
    def learn_policy_fn(cls):
        policy = DIStarMockPolicy(DIStarMockPolicy.default_config(), enable_field=['learn'])
        return policy

    @classmethod
    def collect_policy_fn(cls):
        # policy = DIStarMockPolicyCollect()
        policy = DIStarMockPolicy(DIStarMockPolicy.default_config(), enable_field=['collect'])
        return policy


@pytest.mark.unittest
def test_league_actor():
    with task.start(async_mode=True, ctx=BattleContext()):
        policy = PrepareTest.learn_policy_fn().learn_mode

        def test_actor():
            job = Job(
                launch_player='main_player_default_0',
                players=[
                    PlayerMeta(
                        player_id='main_player_default_0', checkpoint=FileStorage(path=None), total_agent_step=0
                    ),
                    PlayerMeta(
                        player_id='main_player_default_1', checkpoint=FileStorage(path=None), total_agent_step=0
                    )
                ]
            )
            testcases = {
                "on_actor_greeting": False,
                "on_actor_job": False,
                "on_actor_data": False,
            }

            def on_actor_greeting(actor_id):
                assert actor_id == task.router.node_id
                testcases["on_actor_greeting"] = True

            def on_actor_job(job_: Job):
                assert job_.launch_player == job.launch_player
                print(job)
                testcases["on_actor_job"] = True

            def on_actor_data(actor_data):
                print('got actor_data')
                assert isinstance(actor_data, ActorData)
                testcases["on_actor_data"] = True

            task.on(EventEnum.ACTOR_GREETING, on_actor_greeting)
            task.on(EventEnum.ACTOR_FINISH_JOB, on_actor_job)
            task.on(EventEnum.ACTOR_SEND_DATA.format(player=job.launch_player), on_actor_data)

            def _test_actor(ctx):
                sleep(0.3)
                task.emit(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=task.router.node_id), job)
                sleep(0.3)

                task.emit(
                    EventEnum.LEARNER_SEND_MODEL,
                    LearnerModel(player_id='main_player_default_0', state_dict=policy.state_dict(), train_iter=0)
                )
                # sleep(100)
                # try:
                #     print(testcases)
                #     assert all(testcases.values())
                # finally:
                #     task.finish = True

            return _test_actor

        with patch("ding.framework.middleware.collector.battle_inferencer", battle_inferencer_for_distar):
            with patch("ding.framework.middleware.collector.battle_rolloutor", battle_rolloutor_for_distar):
                league_actor = StepLeagueActor(
                    cfg=cfg, env_fn=PrepareTest.get_env_supervisor, policy_fn=PrepareTest.collect_policy_fn
                )
                task.use(test_actor())
                task.use(league_actor)
                task.run()


if __name__ == '__main__':
    test_league_actor()
