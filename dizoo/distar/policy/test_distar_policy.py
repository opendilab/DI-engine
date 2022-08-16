import pytest

from dizoo.distar.policy import DIStarPolicy
from dizoo.distar.envs import get_fake_rl_batch, get_fake_env_reset_data, get_fake_env_step_data

import torch
from ding.utils import set_pkg_seed


@pytest.mark.envtest
class TestDIStarPolicy:

    def test_forward_learn(self):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['learn'])
        policy = policy.learn_mode
        data = get_fake_rl_batch(batch_size=3)
        output = policy.forward(data)
        print(output)

    def test_forward_collect(self):
        policy = DIStarPolicy(DIStarPolicy.default_config(), enable_field=['collect'])
        policy = policy.collect_mode

        rl_model = torch.load('./rl_model.pth')
        policy.load_state_dict(rl_model)

        data = get_fake_env_reset_data()
        policy.reset(data)
        output = policy.forward(data)
        print(output)


if __name__ == '__main__':
    set_pkg_seed(0, use_cuda=False)
    torch.set_printoptions(profile="full")
    test = TestDIStarPolicy()
    test.test_forward_collect()