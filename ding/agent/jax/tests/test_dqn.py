import pytest
import copy
import gym
import jax
import jax.numpy as jnp
from ding.agent.jax import NNState
from ding.agent.jax import DQNNetwork, DQNAgent
from ding.jax_utils import collate_fn_jax, to_raw


@pytest.mark.unittest
def test_dqn_network():
    rng_key = jax.random.PRNGKey(42)
    x = jax.random.uniform(rng_key, [4, 10])
    dummy_data = jnp.ones([1, 10])

    model = NNState(rng_key, dummy_data, create_model_fn=lambda: DQNNetwork(2))

    output = model.forward(x)
    assert output.shape == (4, 2)


@pytest.mark.unittest
def test_dqn_agent_learn():
    B = 3
    rng_key = jax.random.PRNGKey(42)
    env = gym.make('CartPole-v1')
    env.reset()
    cfg = DQNAgent.default_config()
    rng_key, model_rng_key = jax.random.split(rng_key, 2)
    model = DQNAgent.default_model(model_rng_key, env.observation_space, env.action_space)
    agent = DQNAgent(cfg, model)

    data_rng_key = jax.random.split(rng_key, B)
    data = [agent.random_transition(data_rng_key[b], env.observation_space, env.action_space) for b in range(B)]
    data = collate_fn_jax(data)

    info = agent.learn(data)
    info = to_raw(info)
    assert jnp.isscalar(info['total_loss'])
    assert len(info['priority']) == B

    action = agent.eval(data.obs)
    assert action.shape == (B, )

    action = agent.collect(data.obs, eps=0.5)
    assert action.shape == (B, )
    timestep = env.step(env.action_space.sample())
    transition = agent.process_transition(data.obs[0], action[0], timestep)
    assert isinstance(transition, dict)

    obs = env.reset()
    while True:
        action = agent.eval(obs)
        raw_action = to_raw(action)
        obs, reward, done, info = env.step(raw_action)
        if done:
            break

    obs = env.reset()
    trajectory = []
    while True:
        action = agent.collect(obs, eps=0.5)
        raw_action = to_raw(action)
        timestep = env.step(raw_action)

        transition = agent.process_transition(obs, action, timestep)
        trajectory.append(transition)
        if transition['done']:
            break

    batch = collate_fn_jax(trajectory)
    old_param = copy.deepcopy(agent.model.param)
    info = agent.learn(batch)
    info = to_raw(info)
    assert jnp.isscalar(info['total_loss'])
    assert len(info['priority']) == len(trajectory)
    assert (agent.model.param['dqn_network/~/linear']['w'] != old_param['dqn_network/~/linear']['w']).any()
