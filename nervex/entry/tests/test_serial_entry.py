import pytest
import time
import os
from copy import deepcopy

from nervex.entry import serial_pipeline
from app_zoo.classic_control.bitflip.config import bitflip_dqn_default_config
from app_zoo.classic_control.cartpole.config import \
    cartpole_a2c_default_config, cartpole_dqn_default_config, cartpole_dqnvanilla_default_config, \
    cartpole_impala_default_config, cartpole_ppo_default_config, cartpole_ppovanilla_default_config, \
    cartpole_r2d2_default_config, cartpole_rainbowdqn_default_config, cartpole_rainbowdqn_iqn_config, \
    cartpole_ppg_default_config, cartpole_sqn_default_config
from app_zoo.classic_control.pendulum.config import pendulum_ddpg_default_config, pendulum_ppo_default_config, \
    pendulum_sac_auto_alpha_config, pendulum_sac_default_config, pendulum_td3_default_config
from app_zoo.multiagent_particle.config import cooperative_navigation_collaq_default_config, \
    cooperative_navigation_coma_default_config, cooperative_navigation_iql_default_config, \
    cooperative_navigation_qmix_default_config, cooperative_navigation_atoc_default_config


@pytest.mark.unittest
def test_dqn():
    config = deepcopy(cartpole_dqn_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_ddpg():
    config = deepcopy(pendulum_ddpg_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_td3():
    config = deepcopy(pendulum_td3_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c():
    config = deepcopy(cartpole_a2c_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_rainbow_dqn():
    config = deepcopy(cartpole_rainbowdqn_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_iqn():
    config = deepcopy(cartpole_rainbowdqn_iqn_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_dqn_vanilla():
    config = deepcopy(cartpole_dqnvanilla_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo():
    config = deepcopy(cartpole_ppo_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo_vanilla():
    config = deepcopy(cartpole_ppovanilla_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo_vanilla_continous():
    config = deepcopy(pendulum_ppo_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac():
    config = deepcopy(pendulum_sac_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac_auto_alpha():
    config = deepcopy(pendulum_sac_auto_alpha_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_r2d2():
    config = deepcopy(cartpole_r2d2_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
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
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_impala():
    config = deepcopy(cartpole_impala_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_her_dqn():
    config = deepcopy(bitflip_dqn_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_collaq_particle():
    config = deepcopy(cooperative_navigation_collaq_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_coma_particle():
    config = deepcopy(cooperative_navigation_coma_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_qmix_particle():
    config = deepcopy(cooperative_navigation_qmix_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_atoc_particle():
    config = deepcopy(cooperative_navigation_atoc_default_config)
    config.policy.use_cuda = False
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_ppg():
    config = deepcopy(cartpole_ppg_default_config)
    config.policy.use_cuda = False
    config.policy.learn.train_iteration = 10
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 10
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sqn():
    config = deepcopy(cartpole_sqn_default_config)
    config.policy.learn.train_iteration = 1
    config.evaluator.stop_value = -float("inf")
    config.evaluator.eval_freq = 1
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')
