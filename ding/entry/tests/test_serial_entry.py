import pytest
import time
import os
from copy import deepcopy

from ding.entry import serial_pipeline, collect_demo_data, serial_pipeline_offline
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_a2c_config import cartpole_a2c_config, cartpole_a2c_create_config
from dizoo.classic_control.cartpole.config.cartpole_impala_config import cartpole_impala_config, cartpole_impala_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_rainbow_config import cartpole_rainbow_config, cartpole_rainbow_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_iqn_config import cartpole_iqn_config, cartpole_iqn_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_c51_config import cartpole_c51_config, cartpole_c51_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_qrdqn_config import cartpole_qrdqn_config, cartpole_qrdqn_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_sqn_config import cartpole_sqn_config, cartpole_sqn_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_ppg_config import cartpole_ppg_config, cartpole_ppg_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_acer_config import cartpole_acer_config, cartpole_acer_create_config  # noqa
from dizoo.classic_control.cartpole.entry.cartpole_ppg_main import main as ppg_main
from dizoo.classic_control.cartpole.entry.cartpole_ppo_main import main as ppo_main
from dizoo.classic_control.cartpole.config.cartpole_r2d2_config import cartpole_r2d2_config, cartpole_r2d2_create_config  # noqa
from dizoo.classic_control.pendulum.config import pendulum_ddpg_config, pendulum_ddpg_create_config
from dizoo.classic_control.pendulum.config import pendulum_td3_config, pendulum_td3_create_config
from dizoo.classic_control.pendulum.config import pendulum_sac_config, pendulum_sac_create_config
from dizoo.classic_control.pendulum.config import pendulum_d4pg_config, pendulum_d4pg_create_config
from dizoo.classic_control.bitflip.config import bitflip_her_dqn_config, bitflip_her_dqn_create_config
from dizoo.classic_control.bitflip.entry.bitflip_dqn_main import main as bitflip_dqn_main
from dizoo.multiagent_particle.config import cooperative_navigation_qmix_config, cooperative_navigation_qmix_create_config  # noqa
from dizoo.multiagent_particle.config import cooperative_navigation_wqmix_config, cooperative_navigation_wqmix_create_config  # noqa
from dizoo.multiagent_particle.config import cooperative_navigation_vdn_config, cooperative_navigation_vdn_create_config  # noqa
from dizoo.multiagent_particle.config import cooperative_navigation_coma_config, cooperative_navigation_coma_create_config  # noqa
from dizoo.multiagent_particle.config import cooperative_navigation_collaq_config, cooperative_navigation_collaq_create_config  # noqa
from dizoo.multiagent_particle.config import cooperative_navigation_qtran_config, cooperative_navigation_qtran_create_config  # noqa
from dizoo.multiagent_particle.config import cooperative_navigation_atoc_config, cooperative_navigation_atoc_create_config  # noqa
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config
from dizoo.league_demo.selfplay_demo_ppo_main import main as selfplay_main
from dizoo.league_demo.league_demo_ppo_main import main as league_main
from dizoo.classic_control.pendulum.config.pendulum_sac_data_generation_default_config import pendulum_sac_data_genearation_default_config, pendulum_sac_data_genearation_default_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_cql_config import pendulum_cql_default_config, pendulum_cql_default_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_qrdqn_generation_data_config import cartpole_qrdqn_generation_data_config, cartpole_qrdqn_generation_data_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_cql_config import cartpole_discrete_cql_config, cartpole_discrete_cql_create_config  # noqa


@pytest.mark.unittest
@pytest.mark.dqn
def test_dqn():
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_ddpg():
    config = [deepcopy(pendulum_ddpg_config), deepcopy(pendulum_ddpg_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_td3():
    config = [deepcopy(pendulum_td3_config), deepcopy(pendulum_td3_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c():
    config = [deepcopy(cartpole_a2c_config), deepcopy(cartpole_a2c_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_rainbow():
    config = [deepcopy(cartpole_rainbow_config), deepcopy(cartpole_rainbow_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_iqn():
    config = [deepcopy(cartpole_iqn_config), deepcopy(cartpole_iqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_c51():
    config = [deepcopy(cartpole_c51_config), deepcopy(cartpole_c51_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_qrdqn():
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_ppo():
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        ppo_main(config[0], seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac():
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sac_auto_alpha():
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.is_auto_alpha = True
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_r2d2():
    config = [deepcopy(cartpole_r2d2_config), deepcopy(cartpole_r2d2_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_a2c_with_nstep_return():
    config = [deepcopy(cartpole_a2c_config), deepcopy(cartpole_a2c_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.nstep_return = config[0].policy.collect.nstep_return = True
    config[0].policy.collect.discount_factor = 0.9
    config[0].policy.collect.nstep = 3
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_impala():
    config = [deepcopy(cartpole_impala_config), deepcopy(cartpole_impala_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_her_dqn():
    bitflip_her_dqn_config.policy.cuda = False
    try:
        bitflip_dqn_main(bitflip_her_dqn_config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_collaq():
    config = [deepcopy(cooperative_navigation_collaq_config), deepcopy(cooperative_navigation_collaq_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_coma():
    config = [deepcopy(cooperative_navigation_coma_config), deepcopy(cooperative_navigation_coma_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_qmix():
    config = [deepcopy(cooperative_navigation_qmix_config), deepcopy(cooperative_navigation_qmix_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_wqmix():
    config = [deepcopy(cooperative_navigation_wqmix_config), deepcopy(cooperative_navigation_wqmix_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_qtran():
    config = [deepcopy(cooperative_navigation_qtran_config), deepcopy(cooperative_navigation_qtran_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_atoc():
    config = [deepcopy(cooperative_navigation_atoc_config), deepcopy(cooperative_navigation_atoc_create_config)]
    config[0].policy.cuda = False
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_ppg():
    cartpole_ppg_config.policy.use_cuda = False
    try:
        ppg_main(cartpole_ppg_config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_sqn():
    config = [deepcopy(cartpole_sqn_config), deepcopy(cartpole_sqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.unittest
def test_selfplay():
    try:
        selfplay_main(deepcopy(league_demo_ppo_config), seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_league():
    try:
        league_main(deepcopy(league_demo_ppo_config), seed=0, max_iterations=1)
    except Exception as e:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_acer():
    config = [deepcopy(cartpole_acer_config), deepcopy(cartpole_acer_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.unittest
def test_cql():
    # train expert
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [
        deepcopy(pendulum_sac_data_genearation_default_config),
        deepcopy(pendulum_sac_data_genearation_default_create_config)
    ]
    collect_count = 1000
    expert_data_path = config[0].policy.collect.save_path
    state_dict = torch.load('./default_experiment/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(
            config, seed=0, collect_count=collect_count, expert_data_path=expert_data_path, state_dict=state_dict
        )
    except Exception:
        assert False, "pipeline fail"

    # test cql
    config = [deepcopy(pendulum_cql_default_config), deepcopy(pendulum_cql_default_create_config)]
    config[0].policy.learn.train_epoch = 1
    config[0].policy.eval.evaluator.eval_freq = 1
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf default_experiment')


@pytest.mark.unittest
def test_d4pg():
    config = [deepcopy(pendulum_d4pg_config), deepcopy(pendulum_d4pg_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_iterations=1)
    except Exception:
        assert False, "pipeline fail"


def test_discrete_cql():
    # train expert
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'cartpole'
    # collect expert data
    import torch
    config = [deepcopy(cartpole_qrdqn_generation_data_config), deepcopy(cartpole_qrdqn_generation_data_create_config)]
    collect_count = 1000
    expert_data_path = config[0].policy.collect.save_path
    state_dict = torch.load('./cartpole/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(
            config, seed=0, collect_count=collect_count, expert_data_path=expert_data_path, state_dict=state_dict
        )
    except Exception:
        assert False, "pipeline fail"

    # train cql
    config = [deepcopy(cartpole_discrete_cql_config), deepcopy(cartpole_discrete_cql_create_config)]
    config[0].policy.learn.train_epoch = 1
    config[0].policy.eval.evaluator.eval_freq = 1
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf cartpole cartpole_cql')
