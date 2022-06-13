from time import sleep
import pytest
from copy import deepcopy
from ding.envs import BaseEnvManager
from ding.framework.context import BattleContext
from ding.framework.middleware.league_learner import LearnerModel
from ding.framework.middleware.tests.league_config import cfg
from ding.framework.middleware import LeagueActor
from ding.framework.middleware.functional import ActorData
from ding.league.player import PlayerMeta
from ding.framework.storage import FileStorage
from easydict import EasyDict

from ding.framework.task import task
from ding.league.v2.base_league import Job
from dizoo.distar.envs.distar_env import DIStarEnv
from unittest.mock import Mock, patch

from ding.framework import EventEnum
from typing import Dict, Any, List, Optional
from collections import namedtuple
from distar.ctools.utils import read_config
import treetensor.torch as ttorch
from easydict import EasyDict

class LearnMode:
    def __init__(self) -> None:
        pass

    def state_dict(self):
        return {}

class CollectMode:
    def __init__(self) -> None:
        self._cfg = EasyDict(dict(
            collect = dict(
                n_episode = 64
            )
        ))

    def load_state_dict(self, state_dict):
        return
    
    def forward(self, data: Dict):
        return_data = {}
        return_data['action'] = DIStarEnv.random_action(data)
        return_data['logit'] = [1]
        return_data['value'] = [0]

        return return_data
    
    def process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition
    
    def reset(self, data_id: Optional[List[int]] = None) -> None:
        pass
    
    def get_attribute(self, name: str) -> Any:
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError
    

class MockActorDIstarPolicy():
    
    def __init__(self):
        
        self.learn_mode = LearnMode()
        self.collect_mode = CollectMode()


def prepare_test():
    global cfg
    cfg = deepcopy(cfg)

    env_cfg = read_config('./test_distar_config.yaml')
    env_cfg.env.map_name = 'KingsCove'

    def env_fn():
        env = BaseEnvManager(
            env_fn=[lambda: DIStarEnv(env_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        env.seed(cfg.seed)
        return env

    def policy_fn():
        policy = MockActorDIstarPolicy()
        return policy

    return cfg, env_fn, policy_fn



@pytest.mark.unittest
def test_league_actor():
    cfg, env_fn, policy_fn = prepare_test()
    policy = policy_fn()
    with task.start(async_mode=True, ctx = BattleContext()):
        league_actor = LeagueActor(cfg=cfg, env_fn=env_fn, policy_fn=policy_fn)

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
                testcases["on_actor_job"] = True

            def on_actor_data(actor_data):
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
                    LearnerModel(
                        player_id='main_player_default_0', state_dict=policy.learn_mode.state_dict(), train_iter=0
                    )
                )
                sleep(150)
                # try:
                #     print(testcases)
                #     assert all(testcases.values())
                # finally:
                #     task.finish = True

            return _test_actor

        task.use(test_actor())
        task.use(league_actor)
        task.run()

if __name__ == '__main__':
    test_league_actor()