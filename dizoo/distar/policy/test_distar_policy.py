import pytest

from dizoo.distar.policy import DIStarPolicy
from dizoo.distar.envs import get_fake_rl_trajectory, get_fake_env_reset_data, get_fake_env_step_data


@pytest.mark.envtest
class TestDIStarPolicy:

    def test_forward_learn(self):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])
        policy = policy.learn_mode
        data = get_fake_rl_trajectory(batch_size=3)
        output = policy.forward(data)
        print(output)

    def test_forward_collect(self):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['collect'])
        policy = policy.collect_mode
        data = get_fake_env_reset_data()
        policy.reset(data)
        data = get_fake_env_step_data()
        output = policy.forward(data)
        print(output)
