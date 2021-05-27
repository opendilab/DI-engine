import pytest
from nervex.worker import EpisodeCollector
from nervex.envs import BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager
from nervex.policy import DQNPolicy
from nervex.model import FCDiscreteNet
from app_zoo.classic_control.cartpole.envs import CartPoleEnv


@pytest.mark.unittest
@pytest.mark.parametrize('env_manager_type', [BaseEnvManager, SyncSubprocessEnvManager, AsyncSubprocessEnvManager])
def test_collect(env_manager_type):
    env = env_manager_type([lambda: CartPoleEnv({}) for _ in range(8)], env_manager_type.default_config())
    env.seed(0)
    model = FCDiscreteNet(obs_shape=4, action_shape=1)
    policy = DQNPolicy(DQNPolicy.default_config(), model=model).collect_mode
    collector = EpisodeCollector(EpisodeCollector.default_config(), env, policy)

    collected_episode = collector.collect(
        n_episode=18, train_iter=collector._collect_print_freq, policy_kwargs={'eps': 0.5}
    )
    assert len(collected_episode) == 18
    assert all([e[-1]['done'] for e in collected_episode])
    assert all([len(c) == 0 for c in collector._traj_cache.values()])
