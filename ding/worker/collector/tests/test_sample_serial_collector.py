import pytest
from ding.worker import SampleSerialCollector
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.model import DQN
from dizoo.classic_control.cartpole.envs import CartPoleEnv


@pytest.mark.unittest
@pytest.mark.parametrize('env_manager_type', [BaseEnvManager, SyncSubprocessEnvManager])
def test_collect(env_manager_type):
    env = env_manager_type([lambda: CartPoleEnv({}) for _ in range(8)], env_manager_type.default_config())
    env.seed(0)
    model = DQN(obs_shape=4, action_shape=1)
    policy = DQNPolicy(DQNPolicy.default_config(), model=model).collect_mode
    collector = SampleSerialCollector(SampleSerialCollector.default_config(), env, policy)

    collected_sample = collector.collect(
        n_sample=1000,
        train_iter=collector._collect_print_freq,
        record_random_collect=False,
        policy_kwargs={'eps': 0.5}
    )
    assert len(collected_sample) == 1000


@pytest.mark.unittest
@pytest.mark.parametrize('env_manager_type', [BaseEnvManager, SyncSubprocessEnvManager])
def test_random_collect(env_manager_type):
    env = env_manager_type([lambda: CartPoleEnv({}) for _ in range(8)], env_manager_type.default_config())
    env.seed(0)
    model = DQN(obs_shape=4, action_shape=1)
    policy = DQNPolicy(DQNPolicy.default_config(), model=model).collect_mode
    collector = SampleSerialCollector(SampleSerialCollector.default_config(), env, policy)

    collected_sample = collector.collect(
        n_sample=1000, train_iter=collector._collect_print_freq, record_random_collect=True, policy_kwargs={'eps': 0.5}
    )
    assert len(collected_sample) == 1000
