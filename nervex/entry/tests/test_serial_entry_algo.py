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
from app_zoo.smac.entry import smac_collaQ_default_config, smac_coma_default_config, smac_qmix_default_config
from app_zoo.multiagent_particle.config import cooperative_navigation_collaq_default_config, \
    cooperative_navigation_coma_default_config, cooperative_navigation_iql_default_config, \
    cooperative_navigation_qmix_default_config

with open("./algo_record.log", "w+") as f:
    f.write("ALGO TEST STARTS\n")


@pytest.mark.algotest
def test_dqn():
    config = deepcopy(cartpole_dqn_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("1. dqn\n")


@pytest.mark.algotest
def test_ddpg():
    config = deepcopy(pendulum_ddpg_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("2. ddpg\n")


@pytest.mark.algotest
def test_td3():
    config = deepcopy(pendulum_td3_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("3. td3\n")


@pytest.mark.algotest
def test_a2c():
    config = deepcopy(cartpole_a2c_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("4. a2c\n")


@pytest.mark.algotest
def test_rainbow_dqn():
    config = deepcopy(cartpole_rainbowdqn_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("5. rainbow\n")


@pytest.mark.algotest
def test_dqn_vanilla():
    config = deepcopy(cartpole_dqnvanilla_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("6. dqn vanilla\n")


@pytest.mark.algotest
def test_ppo():
    config = deepcopy(cartpole_ppo_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("7. ppo\n")


@pytest.mark.algotest
def test_ppo_vanilla():
    config = deepcopy(cartpole_ppovanilla_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("8. ppo vanilla\n")


@pytest.mark.algotest
def test_sac():
    config = deepcopy(pendulum_sac_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("9. sac\n")


@pytest.mark.algotest
def test_sac_auto_alpha():
    config = deepcopy(pendulum_sac_auto_alpha_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("10. sac with auto alpha\n")


# @pytest.mark.algotest
def test_r2d2():
    config = deepcopy(cartpole_r2d2_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("11. r2d2\n")


# @pytest.mark.algotest
def test_qmix():
    config = deepcopy(smac_qmix_default_config)
    config.env.env_type = 'fake_smac'
    config.env.import_names = ['app_zoo.smac.envs.fake_smac_env']
    config.policy.use_cuda = False
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("12. qmix\n")


# @pytest.mark.algotest
def test_coma():
    config = deepcopy(smac_coma_default_config)
    config.env.env_type = 'fake_smac'
    config.env.import_names = ['app_zoo.smac.envs.fake_smac_env']
    config.policy.use_cuda = False
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("13. coma\n")


@pytest.mark.algotest
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
    with open("./algo_record.log", "a+") as f:
        f.write("14. a2c with nstep return\n")


# @pytest.mark.algotest
def test_ppo_vanilla_continous():
    config = deepcopy(pendulum_ppo_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("15. ppo vanilla continuous\n")


@pytest.mark.algotest
def test_impala():
    config = deepcopy(cartpole_impala_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("16. impala\n")


@pytest.mark.algotest
def test_iqn():
    config = deepcopy(cartpole_rainbowdqn_iqn_config)
    config.evaluator.stop_value = 30  # for save time
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("17. iqn\n")


@pytest.mark.algotest
def test_her_dqn():
    config = deepcopy(bitflip_dqn_default_config)
    try:
        serial_pipeline(config, seed=0)
        os.popen('rm -rf log ckpt*')
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("18. her dqn\n")


@pytest.mark.algotest
def test_ppg():
    config = deepcopy(cartpole_ppg_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("19. ppg\n")


@pytest.mark.algotest
def test_sqn():
    config = deepcopy(cartpole_sqn_default_config)
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("20. sqn\n")
