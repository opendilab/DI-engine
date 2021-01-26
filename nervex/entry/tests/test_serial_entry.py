import pytest
import time
import os
from nervex.entry import serial_pipeline
from nervex.utils import read_config


@pytest.mark.unittest
def test_dqn():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_dqn_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ddpg():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_ddpg_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_td3():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_td3_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_a2c_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_rainbow_dqn():
    path = os.path.join(
        os.path.dirname(__file__),
        '../../../app_zoo/classic_control/cartpole/entry/cartpole_rainbowdqn_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_iqn():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_rainbowdqn_iqn_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_dqn_vanilla():
    path = os.path.join(
        os.path.dirname(__file__),
        '../../../app_zoo/classic_control/cartpole/entry/cartpole_dqnvanilla_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_ppo_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo_vanilla():
    path = os.path.join(
        os.path.dirname(__file__),
        '../../../app_zoo/classic_control/cartpole/entry/cartpole_ppovanilla_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo_vanilla_continous():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_ppo_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/pendulum/entry/pendulum_sac_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_r2d2():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_r2d2_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_qmix():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/smac/entry/smac_qmix_default_config.yaml')
    config = read_config(path)
    config.env.env_type = 'fake_smac'
    config.env.import_names = ['app_zoo.smac.envs.fake_smac_env']
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_collaQ():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/smac/entry/smac_collaQ_default_config.yaml')
    config = read_config(path)
    config.env.env_type = 'fake_smac'
    config.env.import_names = ['app_zoo.smac.envs.fake_smac_env']
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_coma():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/smac/entry/smac_coma_default_config.yaml')
    config = read_config(path)
    config.env.env_type = 'fake_smac'
    config.env.import_names = ['app_zoo.smac.envs.fake_smac_env']
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c_with_nstep_return():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_a2c_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.algo.use_nstep_return = True
    config.policy.learn.algo.discount_factor = config.policy.collect.algo.discount_factor
    config.policy.learn.algo.nstep = 3
    config.policy.collect.algo.use_nstep_return = config.policy.learn.algo.use_nstep_return
    config.policy.collect.algo.nstep = config.policy.learn.algo.nstep
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_impala():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/cartpole/entry/cartpole_impala_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_her_dqn():
    path = os.path.join(
        os.path.dirname(__file__), '../../../app_zoo/classic_control/bitflip/entry/bitflip_dqn_default_config.yaml'
    )
    config = read_config(path)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

@pytest.mark.unittest
def test_collaQ_particle():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/multiagent_particle/entry/cooperative_navigation_collaQ_default_config.yaml')
    config = read_config(path)
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

@pytest.mark.unittest
def test_coma_particle():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/multiagent_particle/entry/cooperative_navigation_coma_default_config.yaml')
    config = read_config(path)
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

@pytest.mark.unittest
def test_qmix_particle():
    path = os.path.join(os.path.dirname(__file__), '../../../app_zoo/multiagent_particle/entry/cooperative_navigation_qmix_default_config.yaml')
    config = read_config(path)
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
