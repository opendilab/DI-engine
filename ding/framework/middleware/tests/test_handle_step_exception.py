from ding.framework.context import BattleContext
from ding.framework.middleware.functional.collector import TransitionList
from ding.framework.middleware.tests import battle_rolloutor_for_distar
import pytest
from unittest.mock import Mock
from ding.envs import BaseEnvTimestep
from easydict import EasyDict


class MockEnvManager(Mock):

    def step(self, actions):
        timesteps = {}
        for env_id in actions.keys():
            timesteps[env_id] = BaseEnvTimestep(obs=[1], reward=[1, 1], done=False, info={'step_error': True})
        return timesteps


@pytest.mark.unittest
def test_handle_step_exception():
    ctx = BattleContext()
    ctx.total_envstep_count = 10
    ctx.env_step = 20
    transitions_list = [TransitionList(env_num=2)]
    for _ in range(5):
        transitions_list[0].append(env_id=0, transition=BaseEnvTimestep(obs=[1], reward=[1, 1], done=False, info={}))
        transitions_list[0].append(env_id=1, transition=BaseEnvTimestep(obs=[1], reward=[1, 1], done=False, info={}))

    ctx.actions = {0: {}}
    ctx.obs = {0: {0: {}}}
    rolloutor = battle_rolloutor_for_distar(cfg=EasyDict(), env=MockEnvManager(), transitions_list=transitions_list)
    rolloutor(ctx)

    assert ctx.total_envstep_count == 5
    assert ctx.env_step == 15
    assert transitions_list[0].length(0) == 0
    assert transitions_list[0].length(1) == 5


if __name__ == '__main__':
    test_handle_step_exception()