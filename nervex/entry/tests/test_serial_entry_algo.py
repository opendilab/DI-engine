import pytest
import time
import os
from copy import deepcopy

from nervex.entry import serial_pipeline
# from app_zoo.classic_control.bitflip.entry import bitflip_dqn_default_config
# from app_zoo.classic_control.cartpole.entry import \
#     cartpole_a2c_default_config, cartpole_dqn_default_config, cartpole_dqnvanilla_default_config, \
#     cartpole_impala_default_config, cartpole_ppo_default_config, cartpole_ppovanilla_default_config, \
#     cartpole_r2d2_default_config, cartpole_rainbowdqn_default_config, cartpole_rainbowdqn_iqn_config, \
#     cartpole_ppg_default_config, cartpole_sqn_default_config
from app_zoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from app_zoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from app_zoo.classic_control.cartpole.config.cartpole_a2c_config import cartpole_a2c_config, cartpole_a2c_create_config
from app_zoo.classic_control.cartpole.config.cartpole_impala_config import cartpole_impala_config, cartpole_impala_create_config  # noqa
from app_zoo.classic_control.cartpole.config.cartpole_rainbow_config import cartpole_rainbow_config, cartpole_rainbow_create_config  # noqa
from app_zoo.classic_control.cartpole.config.cartpole_iqn_config import cartpole_iqn_config, cartpole_iqn_create_config  # noqa
from app_zoo.classic_control.cartpole.config.cartpole_sqn_config import cartpole_sqn_config, cartpole_sqn_create_config  # noqa
from app_zoo.classic_control.pendulum.config import pendulum_ddpg_config, pendulum_ddpg_create_config
from app_zoo.classic_control.pendulum.config import pendulum_td3_config, pendulum_td3_create_config
from app_zoo.classic_control.bitflip.config import bitflip_dqn_config, bitflip_dqn_create_config
from app_zoo.multiagent_particle.config import cooperative_navigation_qmix_config, cooperative_navigation_qmix_create_config  # noqa
from app_zoo.multiagent_particle.config import cooperative_navigation_vdn_config, cooperative_navigation_vdn_create_config  # noqa
from app_zoo.multiagent_particle.config import cooperative_navigation_coma_config, cooperative_navigation_coma_create_config  # noqa
from app_zoo.multiagent_particle.config import cooperative_navigation_collaq_config, cooperative_navigation_collaq_create_config  # noqa

with open("./algo_record.log", "w+") as f:
    f.write("ALGO TEST STARTS\n")


@pytest.mark.algotest
def test_dqn():
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("1. dqn\n")


@pytest.mark.algotest
def test_ddpg():
    config = [deepcopy(pendulum_ddpg_config), deepcopy(pendulum_ddpg_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("2. ddpg\n")


@pytest.mark.algotest
def test_td3():
    config = [deepcopy(pendulum_td3_config), deepcopy(pendulum_td3_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("3. td3\n")


@pytest.mark.algotest
def test_a2c():
    config = [deepcopy(cartpole_a2c_config), deepcopy(cartpole_a2c_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("4. a2c\n")


@pytest.mark.algotest
def test_rainbow():
    config = [deepcopy(cartpole_rainbow_config), deepcopy(cartpole_rainbow_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("5. rainbow\n")


@pytest.mark.algotest
def test_ppo():
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("6. ppo\n")


# @pytest.mark.algotest
def test_collaq():
    config = [deepcopy(cooperative_navigation_collaq_config), deepcopy(cooperative_navigation_collaq_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("8. collaq\n")


# @pytest.mark.algotest
def test_coma():
    config = [deepcopy(cooperative_navigation_coma_config), deepcopy(cooperative_navigation_coma_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("8. coma\n")


@pytest.mark.algotest
def test_sac():
    config = deepcopy(pendulum_sac_default_config)  # noqa
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("9. sac\n")


@pytest.mark.algotest
def test_sac_auto_alpha():
    config = deepcopy(pendulum_sac_auto_alpha_config)  # noqa
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("10. sac with auto alpha\n")


# @pytest.mark.algotest
def test_r2d2():
    config = deepcopy(cartpole_r2d2_default_config)  # noqa
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("11. r2d2\n")


@pytest.mark.algotest
def test_a2c_with_nstep_return():
    config = [deepcopy(cartpole_a2c_config), deepcopy(cartpole_a2c_create_config)]
    config[0].policy.learn.nstep_return = config[0].policy.collect.nstep_return = True
    config[0].policy.collect.discount_factor = 0.9
    config[0].policy.collect.nstep = 3
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("12. a2c with nstep return\n")


# @pytest.mark.algotest
def test_vdn():
    config = [deepcopy(cooperative_navigation_vdn_config), deepcopy(cooperative_navigation_vdn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("14. vdn\n")


# @pytest.mark.algotest
def test_qmix():
    config = [deepcopy(cooperative_navigation_qmix_config), deepcopy(cooperative_navigation_qmix_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("15. qmix\n")


@pytest.mark.algotest
def test_impala():
    config = [deepcopy(cartpole_impala_config), deepcopy(cartpole_impala_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("16. impala\n")


@pytest.mark.algotest
def test_iqn():
    config = [deepcopy(cartpole_iqn_config), deepcopy(cartpole_iqn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("17. iqn\n")


@pytest.mark.algotest
def test_her_dqn():
    config = [deepcopy(bitflip_dqn_config), deepcopy(bitflip_dqn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("18. her dqn\n")


@pytest.mark.algotest
def test_ppg():
    config = deepcopy(cartpole_ppg_default_config)  # noqa
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("19. ppg\n")


@pytest.mark.algotest
@pytest.mark.sqn
def test_sqn():
    config = [deepcopy(cartpole_sqn_config), deepcopy(cartpole_sqn_create_config)]  # noqa
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("20. sqn\n")
