import pytest
from itertools import product
import time
import os
from copy import deepcopy

from ding.entry import serial_pipeline, collect_demo_data, serial_pipeline_offline
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from dizoo.classic_control.cartpole.config.cartpole_dqn_stdim_config import cartpole_dqn_stdim_config, \
    cartpole_dqn_stdim_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config, \
    cartpole_ppo_offpolicy_create_config
from dizoo.classic_control.cartpole.config.cartpole_impala_config import cartpole_impala_config, cartpole_impala_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_rainbow_config import cartpole_rainbow_config, cartpole_rainbow_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_iqn_config import cartpole_iqn_config, cartpole_iqn_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_fqf_config import cartpole_fqf_config, cartpole_fqf_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_c51_config import cartpole_c51_config, cartpole_c51_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_qrdqn_config import cartpole_qrdqn_config, cartpole_qrdqn_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_sqn_config import cartpole_sqn_config, cartpole_sqn_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_ppg_config import cartpole_ppg_config, cartpole_ppg_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_acer_config import cartpole_acer_config, cartpole_acer_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_sac_config import cartpole_sac_config, cartpole_sac_create_config  # noqa
from dizoo.classic_control.cartpole.entry.cartpole_ppg_main import main as ppg_main
from dizoo.classic_control.cartpole.entry.cartpole_ppo_main import main as ppo_main
from dizoo.classic_control.cartpole.config.cartpole_r2d2_config import cartpole_r2d2_config, cartpole_r2d2_create_config  # noqa
from dizoo.classic_control.pendulum.config import pendulum_ddpg_config, pendulum_ddpg_create_config
from dizoo.classic_control.pendulum.config import pendulum_td3_config, pendulum_td3_create_config
from dizoo.classic_control.pendulum.config import pendulum_sac_config, pendulum_sac_create_config
from dizoo.classic_control.pendulum.config import pendulum_d4pg_config, pendulum_d4pg_create_config
from dizoo.bitflip.config import bitflip_her_dqn_config, bitflip_her_dqn_create_config
from dizoo.bitflip.entry.bitflip_dqn_main import main as bitflip_dqn_main
from dizoo.petting_zoo.config import ptz_simple_spread_atoc_config, ptz_simple_spread_atoc_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_collaq_config, ptz_simple_spread_collaq_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_coma_config, ptz_simple_spread_coma_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_qmix_config, ptz_simple_spread_qmix_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_qtran_config, ptz_simple_spread_qtran_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_vdn_config, ptz_simple_spread_vdn_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_wqmix_config, ptz_simple_spread_wqmix_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_madqn_config, ptz_simple_spread_madqn_create_config  # noqa
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config
from dizoo.league_demo.selfplay_demo_ppo_main import main as selfplay_main
from dizoo.league_demo.league_demo_ppo_main import main as league_main
from dizoo.classic_control.pendulum.config.pendulum_sac_data_generation_config import pendulum_sac_data_genearation_config, pendulum_sac_data_genearation_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_cql_config import pendulum_cql_config, pendulum_cql_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_qrdqn_generation_data_config import cartpole_qrdqn_generation_data_config, cartpole_qrdqn_generation_data_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_cql_config import cartpole_discrete_cql_config, cartpole_discrete_cql_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_dt_config import cartpole_discrete_dt_config, cartpole_discrete_dt_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_td3_data_generation_config import pendulum_td3_generation_config, pendulum_td3_generation_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_td3_bc_config import pendulum_td3_bc_config, pendulum_td3_bc_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_ibc_config import pendulum_ibc_config, pendulum_ibc_create_config
from dizoo.gym_hybrid.config.gym_hybrid_ddpg_config import gym_hybrid_ddpg_config, gym_hybrid_ddpg_create_config
from dizoo.gym_hybrid.config.gym_hybrid_pdqn_config import gym_hybrid_pdqn_config, gym_hybrid_pdqn_create_config
from dizoo.gym_hybrid.config.gym_hybrid_mpdqn_config import gym_hybrid_mpdqn_config, gym_hybrid_mpdqn_create_config
from dizoo.classic_control.pendulum.config.pendulum_bdq_config import pendulum_bdq_config, pendulum_bdq_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_mdqn_config import cartpole_mdqn_config, cartpole_mdqn_create_config


@pytest.mark.platformtest
@pytest.mark.unittest
def test_dqn():
    config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'cartpole_dqn_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf cartpole_dqn_unittest')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_mdqn():
    config = [deepcopy(cartpole_mdqn_config), deepcopy(cartpole_mdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'cartpole_mdqn_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1, dynamic_seed=False)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf cartpole_mdqn_unittest')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_bdq():
    config = [deepcopy(pendulum_bdq_config), deepcopy(pendulum_bdq_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'pendulum_bdq_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf pendulum_bdq_unittest')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_ddpg():
    config = [deepcopy(pendulum_ddpg_config), deepcopy(pendulum_ddpg_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


# @pytest.mark.platformtest
# @pytest.mark.unittest
def test_hybrid_ddpg():
    config = [deepcopy(gym_hybrid_ddpg_config), deepcopy(gym_hybrid_ddpg_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


# @pytest.mark.platformtest
# @pytest.mark.unittest
def test_hybrid_pdqn():
    config = [deepcopy(gym_hybrid_pdqn_config), deepcopy(gym_hybrid_pdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


# @pytest.mark.platformtest
# @pytest.mark.unittest
def test_hybrid_mpdqn():
    config = [deepcopy(gym_hybrid_mpdqn_config), deepcopy(gym_hybrid_mpdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_dqn_stdim():
    config = [deepcopy(cartpole_dqn_stdim_config), deepcopy(cartpole_dqn_stdim_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'cartpole_dqn_stdim_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf cartpole_dqn_stdim_unittest')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_td3():
    config = [deepcopy(pendulum_td3_config), deepcopy(pendulum_td3_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_rainbow():
    config = [deepcopy(cartpole_rainbow_config), deepcopy(cartpole_rainbow_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_iqn():
    config = [deepcopy(cartpole_iqn_config), deepcopy(cartpole_iqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_fqf():
    config = [deepcopy(cartpole_fqf_config), deepcopy(cartpole_fqf_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_c51():
    config = [deepcopy(cartpole_c51_config), deepcopy(cartpole_c51_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_qrdqn():
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_ppo():
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'ppo_offpolicy_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_ppo_nstep_return():
    config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.nstep_return = True
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_sac():
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.auto_alpha = False
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_sac_auto_alpha():
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.auto_alpha = True
    config[0].policy.learn.log_space = False
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_sac_log_space():
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.auto_alpha = True
    config[0].policy.learn.log_space = True
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_discrete_sac():
    auto_alpha, log_space = True, False
    config = [deepcopy(cartpole_sac_config), deepcopy(cartpole_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.auto_alpha = auto_alpha
    config[0].policy.learn.log_space = log_space
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_discrete_sac_twin_critic():
    config = [deepcopy(cartpole_sac_config), deepcopy(cartpole_sac_create_config)]
    config[0].cuda = True
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.learn.auto_alpha = True
    config[0].policy.learn.log_space = True
    config[0].policy.model.twin_critic = False
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_r2d2():
    config = [deepcopy(cartpole_r2d2_config), deepcopy(cartpole_r2d2_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=5)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_impala():
    config = [deepcopy(cartpole_impala_config), deepcopy(cartpole_impala_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_her_dqn():
    bitflip_her_dqn_config.policy.cuda = False
    try:
        bitflip_dqn_main(bitflip_her_dqn_config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_collaq():
    config = [deepcopy(ptz_simple_spread_collaq_config), deepcopy(ptz_simple_spread_collaq_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.collect.n_sample = 100
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_coma():
    config = [deepcopy(ptz_simple_spread_coma_config), deepcopy(ptz_simple_spread_coma_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.collect.n_sample = 100
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_qmix():
    config = [deepcopy(ptz_simple_spread_qmix_config), deepcopy(ptz_simple_spread_qmix_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.collect.n_sample = 100
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_wqmix():
    config = [deepcopy(ptz_simple_spread_wqmix_config), deepcopy(ptz_simple_spread_wqmix_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.collect.n_sample = 100
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_madqn():
    config = [deepcopy(ptz_simple_spread_madqn_config), deepcopy(ptz_simple_spread_madqn_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_qtran():
    config = [deepcopy(ptz_simple_spread_qtran_config), deepcopy(ptz_simple_spread_qtran_create_config)]
    config[0].policy.cuda = False
    config[0].policy.learn.update_per_collect = 1
    config[0].policy.collect.n_sample = 100
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_atoc():
    config = [deepcopy(ptz_simple_spread_atoc_config), deepcopy(ptz_simple_spread_atoc_create_config)]
    config[0].policy.cuda = False
    config[0].policy.collect.n_sample = 100
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_ppg():
    cartpole_ppg_config.policy.use_cuda = False
    try:
        ppg_main(cartpole_ppg_config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_sqn():
    config = [deepcopy(cartpole_sqn_config), deepcopy(cartpole_sqn_create_config)]
    config[0].policy.learn.update_per_collect = 8
    config[0].policy.learn.batch_size = 8
    try:
        serial_pipeline(config, seed=0, max_train_iter=2)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf log ckpt*')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_selfplay():
    try:
        selfplay_main(deepcopy(league_demo_ppo_config), seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_league():
    try:
        league_main(deepcopy(league_demo_ppo_config), seed=0, max_train_iter=1)
    except Exception as e:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_acer():
    config = [deepcopy(cartpole_acer_config), deepcopy(cartpole_acer_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_cql():
    # train expert
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'sac_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [deepcopy(pendulum_sac_data_genearation_config), deepcopy(pendulum_sac_data_genearation_create_config)]
    collect_count = 1000
    expert_data_path = config[0].policy.collect.save_path
    state_dict = torch.load('./sac_unittest/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(
            config, seed=0, collect_count=collect_count, expert_data_path=expert_data_path, state_dict=state_dict
        )
    except Exception:
        assert False, "pipeline fail"

    # test cql
    config = [deepcopy(pendulum_cql_config), deepcopy(pendulum_cql_create_config)]
    config[0].policy.learn.train_epoch = 1
    config[0].policy.eval.evaluator.eval_freq = 1
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_ibc():
    # train expert
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'sac_unittest'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [deepcopy(pendulum_sac_data_genearation_config), deepcopy(pendulum_sac_data_genearation_create_config)]
    collect_count = 1000
    expert_data_path = config[0].policy.collect.save_path
    state_dict = torch.load('./sac_unittest/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(
            config, seed=0, collect_count=collect_count, expert_data_path=expert_data_path, state_dict=state_dict
        )
    except Exception:
        assert False, "pipeline fail"

    # test cql
    config = [deepcopy(pendulum_ibc_config), deepcopy(pendulum_ibc_create_config)]
    config[0].policy.learn.train_epoch = 1
    config[0].policy.eval.evaluator.eval_freq = 1
    config[0].policy.model.stochastic_optim.iters = 2
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"


@pytest.mark.platformtest
@pytest.mark.unittest
def test_d4pg():
    config = [deepcopy(pendulum_d4pg_config), deepcopy(pendulum_d4pg_create_config)]
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception as e:
        assert False, "pipeline fail"
        print(repr(e))


@pytest.mark.platformtest
@pytest.mark.unittest
def test_discrete_cql():
    # train expert
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'cql_cartpole'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    # collect expert data
    import torch
    config = [deepcopy(cartpole_qrdqn_generation_data_config), deepcopy(cartpole_qrdqn_generation_data_create_config)]
    state_dict = torch.load('./cql_cartpole/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(config, seed=0, collect_count=1000, state_dict=state_dict)
    except Exception as e:
        assert False, "pipeline fail"
        print(repr(e))

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


@pytest.mark.platformtest
@pytest.mark.unittest
def test_discrete_dt():
    # train expert
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'dt_cartpole'
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    # collect expert data
    import torch
    config = [deepcopy(cartpole_qrdqn_generation_data_config), deepcopy(cartpole_qrdqn_generation_data_create_config)]
    state_dict = torch.load('./dt_cartpole/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(config, seed=0, collect_count=1000, state_dict=state_dict)
    except Exception as e:
        assert False, "pipeline fail"
        print(repr(e))

    # train dt
    config = [deepcopy(cartpole_discrete_dt_config), deepcopy(cartpole_discrete_dt_create_config)]
    config[0].policy.eval.evaluator.eval_freq = 5
    try:
        from ding.framework import task, ding_init
        from ding.framework.context import OfflineRLContext
        from ding.envs import SubprocessEnvManagerV2, BaseEnvManagerV2
        from ding.envs.env_wrappers.env_wrappers import AllinObsWrapper
        from dizoo.classic_control.cartpole.envs import CartPoleEnv
        from ding.utils import set_pkg_seed
        from ding.data import create_dataset
        from ding.config import compile_config
        from ding.model import DecisionTransformer
        from ding.policy import DTPolicy
        from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, \
            OfflineMemoryDataFetcher, offline_logger, termination_checker
        ding_init(config[0])
        config = compile_config(config[0], create_cfg=config[1], auto=True)
        with task.start(async_mode=False, ctx=OfflineRLContext()):
            evaluator_env = BaseEnvManagerV2(
                env_fn=[lambda: AllinObsWrapper(CartPoleEnv(config.env)) for _ in range(config.env.evaluator_env_num)],
                cfg=config.env.manager
            )

            set_pkg_seed(config.seed, use_cuda=config.policy.cuda)

            dataset = create_dataset(config)

            model = DecisionTransformer(**config.policy.model)
            policy = DTPolicy(config.policy, model=model)

            task.use(termination_checker(max_train_iter=1))
            task.use(interaction_evaluator(config, policy.eval_mode, evaluator_env))
            task.use(OfflineMemoryDataFetcher(config, dataset))
            task.use(trainer(config, policy.learn_mode))
            task.use(CkptSaver(policy, config.exp_name, train_freq=100))
            task.use(offline_logger())
            task.run()
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf cartpole cartpole_dt')


@pytest.mark.platformtest
@pytest.mark.unittest
def test_td3_bc():
    # train expert
    config = [deepcopy(pendulum_td3_config), deepcopy(pendulum_td3_create_config)]
    config[0].exp_name = 'td3'
    config[0].policy.learn.update_per_collect = 1
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [deepcopy(pendulum_td3_generation_config), deepcopy(pendulum_td3_generation_create_config)]
    state_dict = torch.load('./td3/ckpt/iteration_0.pth.tar', map_location='cpu')
    try:
        collect_demo_data(config, seed=0, collect_count=1000, state_dict=state_dict)
    except Exception:
        assert False, "pipeline fail"

    # train td3 bc
    config = [deepcopy(pendulum_td3_bc_config), deepcopy(pendulum_td3_bc_create_config)]
    config[0].exp_name = 'td3_bc'
    config[0].policy.learn.train_epoch = 1
    config[0].policy.eval.evaluator.eval_freq = 1
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.popen('rm -rf td3 td3_bc')
