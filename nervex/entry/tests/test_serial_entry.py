import pytest
import time
import os
from copy import deepcopy

from nervex.entry import serial_pipeline
from app_zoo.classic_control.bitflip.entry import bitflip_dqn_default_config
from app_zoo.classic_control.cartpole.entry import \
    cartpole_a2c_default_config, cartpole_dqn_default_config, cartpole_dqnvanilla_default_config, \
    cartpole_impala_default_config, cartpole_ppo_default_config, cartpole_ppovanilla_default_config, \
    cartpole_r2d2_default_config, cartpole_rainbowdqn_default_config, cartpole_rainbowdqn_iqn_config
from app_zoo.classic_control.pendulum.entry import pendulum_ddpg_default_config, pendulum_ppo_default_config, \
    pendulum_sac_auto_alpha_config, pendulum_sac_default_config, pendulum_td3_default_config
from app_zoo.smac.entry import smac_collaQ_default_config, smac_coma_default_config, smac_qmix_default_config
from app_zoo.multiagent_particle.entry import cooperative_navigation_collaQ_default_config, \
    cooperative_navigation_coma_default_config, cooperative_navigation_iql_default_config, \
    cooperative_navigation_qmix_default_config


@pytest.mark.unittest
def test_dqn():
    config = deepcopy(cartpole_dqn_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ddpg():
    config = deepcopy(pendulum_ddpg_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_td3():
    config = deepcopy(pendulum_td3_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c():
    config = deepcopy(cartpole_a2c_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_rainbow_dqn():
    config = deepcopy(cartpole_rainbowdqn_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_iqn():
    config = deepcopy(cartpole_rainbowdqn_iqn_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_dqn_vanilla():
    config = deepcopy(cartpole_dqnvanilla_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo():
    config = deepcopy(cartpole_ppo_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo_vanilla():
    config = deepcopy(cartpole_ppovanilla_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo_vanilla_continous():
    config = deepcopy(pendulum_ppo_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac():
    config = deepcopy(pendulum_sac_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac_auto_alpha():
    config = deepcopy(pendulum_sac_auto_alpha_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_r2d2():
    config = deepcopy(cartpole_r2d2_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_qmix():
    config = deepcopy(smac_qmix_default_config)
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
    config = deepcopy(smac_collaQ_default_config)
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
    config = deepcopy(smac_coma_default_config)
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
    config = deepcopy(cartpole_a2c_default_config)
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
    config = deepcopy(cartpole_impala_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_her_dqn():
    config = deepcopy(bitflip_dqn_default_config)
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_collaQ_particle():
    config = deepcopy(cooperative_navigation_collaQ_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_coma_particle():
    config = deepcopy(cooperative_navigation_coma_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_qmix_particle():
    config = deepcopy(cooperative_navigation_qmix_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_step = 1
    config.evaluator.stop_val = -float("inf")
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')
