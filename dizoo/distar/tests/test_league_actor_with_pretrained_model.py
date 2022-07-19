from time import sleep
import pytest
from copy import deepcopy
from unittest.mock import patch
from easydict import EasyDict

from dizoo.distar.config import distar_cfg
from dizoo.distar.envs.distar_env import DIStarEnv
from dizoo.distar.policy.distar_policy import DIStarPolicy

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
from ding.framework.middleware.league_learner import LearnerModel
from ding.framework.middleware.functional.collector import battle_inferencer_for_distar, battle_rolloutor_for_distar


def battle_rolloutor_for_distar2(cfg, env, transitions_list, model_info_dict):

    def _battle_rolloutor(ctx: "BattleContext"):
        timesteps = env.step(ctx.actions)

        ctx.total_envstep_count += len(timesteps)
        ctx.env_step += len(timesteps)

        for env_id, timestep in enumerate(timesteps):
            if timestep.info.get('abnormal'):
                for policy_id, policy in enumerate(ctx.current_policies):
                    policy.reset(env.ready_obs[0][policy_id])
                continue

            for policy_id, policy in enumerate(ctx.current_policies):
                if timestep.done:
                    policy.reset(env.ready_obs[0][policy_id])
                    ctx.episode_info[policy_id].append(timestep.info[policy_id])

            if timestep.done:
                ctx.env_episode += 1

    return _battle_rolloutor


env_cfg = dict(
    actor=dict(job_type='train', ),
    env=dict(
        map_name='random',
        player_ids=['agent1', 'bot7'],
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
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])
        return policy

    @classmethod
    def collect_policy_fn(cls):
        # policy = DIStarMockPolicyCollect()
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['collect'])
        return policy


total_games = 0
win_games = 0
draw_games = 0
loss_games = 0


@pytest.mark.unittest
def test_league_actor():
    with task.start(async_mode=True, ctx=BattleContext()):

        def test_actor():
            job = Job(
                launch_player='main_player_default_0',
                players=[
                    PlayerMeta(
                        player_id='main_player_default_0', checkpoint=FileStorage(path=None), total_agent_step=0
                    )
                ]
            )

            def on_actor_job(job_: Job):
                assert job_.launch_player == job.launch_player
                print(job)
                global total_games
                global win_games
                global draw_games
                global loss_games

                for r in job_.result:
                    total_games += 1
                    if r == 'wins':
                        win_games += 1
                    elif r == 'draws':
                        draw_games += 1
                    elif r == 'losses':
                        loss_games += 1
                    else:
                        raise NotImplementedError

                print(
                    'total games {}, win games {}, draw_games {}, loss_games {}'.format(
                        total_games, win_games, draw_games, loss_games
                    )
                )
                if total_games >= 100:
                    task.finish = True
                    exit(0)

            def on_actor_data(actor_data):
                print('got actor_data')
                assert isinstance(actor_data, ActorData)

            task.on(EventEnum.ACTOR_FINISH_JOB, on_actor_job)
            task.on(EventEnum.ACTOR_SEND_DATA.format(player=job.launch_player), on_actor_data)

            def _test_actor(ctx):
                sleep(0.3)
                for _ in range(20):
                    task.emit(EventEnum.COORDINATOR_DISPATCH_ACTOR_JOB.format(actor_id=task.router.node_id), job)
                    sleep(0.3)

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