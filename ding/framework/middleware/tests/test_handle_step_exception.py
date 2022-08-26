from ding.framework.context import BattleContext
from ding.framework.middleware.functional.collector import BattleTransitionList
from ding.framework.middleware.functional import battle_rolloutor
import pytest
from unittest.mock import Mock
from ding.envs import BaseEnvTimestep
from easydict import EasyDict


class MockEnvManager:

    def __init__(self) -> None:
        self.ready_obs = [[[]]]

    def step(self, actions):
        timesteps = {}
        for env_id in actions.keys():
            timesteps[env_id] = BaseEnvTimestep(obs=[1], reward=[1, 1], done=False, info={'abnormal': True})
        return timesteps


class MockPolicy:

    def __init__(self) -> None:
        pass

    def reset(self, data):
        pass


@pytest.mark.unittest
def test_handle_step_exception():
    ctx = BattleContext()
    ctx.total_envstep_count = 10
    ctx.env_step = 20
    transitions_list = [BattleTransitionList(env_num=2, unroll_len=5)]
    ctx.current_policies = [MockPolicy()]
    for _ in range(5):
        transitions_list[0].append(env_id=0, transition=BaseEnvTimestep(obs=[1], reward=[1, 1], done=False, info={}))
        transitions_list[0].append(env_id=1, transition=BaseEnvTimestep(obs=[1], reward=[1, 1], done=False, info={}))

    ctx.actions = {0: {}}
    ctx.obs = {0: {0: {}}}
    rolloutor = battle_rolloutor(cfg=EasyDict(), env=MockEnvManager(), transitions_list=transitions_list, model_info_dict=None)
    rolloutor(ctx)

    assert len(transitions_list[0]._transitions[0]) == 0
    assert len(transitions_list[0]._transitions[1]) == 1
