import pytest
import time
import os
import torch
import subprocess
from copy import deepcopy

from ding.utils import K8sLauncher, OrchestratorLauncher
from ding.entry import serial_pipeline, serial_pipeline_offline, collect_demo_data, serial_pipeline_onpolicy
from ding.entry.serial_entry_sqil import serial_pipeline_sqil
from dizoo.classic_control.cartpole.config.cartpole_sql_config import cartpole_sql_config, cartpole_sql_create_config
from dizoo.classic_control.cartpole.config.cartpole_sqil_config import cartpole_sqil_config, cartpole_sqil_create_config
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
from dizoo.bitflip.config import bitflip_her_dqn_config, bitflip_her_dqn_create_config
from dizoo.bitflip.entry.bitflip_dqn_main import main as bitflip_dqn_main
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config
from dizoo.league_demo.selfplay_demo_ppo_main import main as selfplay_main
from dizoo.league_demo.league_demo_ppo_main import main as league_main
from dizoo.classic_control.pendulum.config.pendulum_sac_data_generation_config import pendulum_sac_data_genearation_config, pendulum_sac_data_genearation_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_cql_config import pendulum_cql_config, pendulum_cql_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_qrdqn_generation_data_config import cartpole_qrdqn_generation_data_config, cartpole_qrdqn_generation_data_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_cql_config import cartpole_discrete_cql_config, cartpole_discrete_cql_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_td3_data_generation_config import pendulum_td3_generation_config, pendulum_td3_generation_create_config  # noqa
from dizoo.classic_control.pendulum.config.pendulum_td3_bc_config import pendulum_td3_bc_config, pendulum_td3_bc_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_atoc_config, ptz_simple_spread_atoc_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_atoc_config, ptz_simple_spread_collaq_config, ptz_simple_spread_collaq_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_coma_config, ptz_simple_spread_coma_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_qmix_config, ptz_simple_spread_qmix_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_qtran_config, ptz_simple_spread_qtran_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_vdn_config, ptz_simple_spread_vdn_create_config  # noqa
from dizoo.petting_zoo.config import ptz_simple_spread_wqmix_config, ptz_simple_spread_wqmix_create_config  # noqa

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
        serial_pipeline_onpolicy(config, seed=0)
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
        ppo_main(config[0], seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("6. ppo\n")


# @pytest.mark.algotest
def test_collaq():
    config = [deepcopy(ptz_simple_spread_collaq_config), deepcopy(ptz_simple_spread_collaq_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("7. collaq\n")


# @pytest.mark.algotest
def test_coma():
    config = [deepcopy(ptz_simple_spread_coma_config), deepcopy(ptz_simple_spread_coma_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("8. coma\n")


@pytest.mark.algotest
def test_sac():
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("9. sac\n")


@pytest.mark.algotest
def test_c51():
    config = [deepcopy(cartpole_c51_config), deepcopy(cartpole_c51_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("10. c51\n")


@pytest.mark.algotest
def test_r2d2():
    config = [deepcopy(cartpole_r2d2_config), deepcopy(cartpole_r2d2_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("11. r2d2\n")


# @pytest.mark.algotest
def test_atoc():
    config = [deepcopy(ptz_simple_spread_atoc_config), deepcopy(ptz_simple_spread_atoc_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("13. atoc\n")


# @pytest.mark.algotest
def test_vdn():
    config = [deepcopy(ptz_simple_spread_vdn_config), deepcopy(ptz_simple_spread_vdn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("14. vdn\n")


# @pytest.mark.algotest
def test_qmix():
    config = [deepcopy(ptz_simple_spread_qmix_config), deepcopy(ptz_simple_spread_qmix_create_config)]
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
    try:
        bitflip_her_dqn_config.exp_name = 'bitflip5_dqn'
        bitflip_her_dqn_config.env.n_bits = 5
        bitflip_her_dqn_config.policy.model.obs_shape = 10
        bitflip_her_dqn_config.policy.model.action_shape = 5
        bitflip_dqn_main(bitflip_her_dqn_config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("18. her dqn\n")


@pytest.mark.algotest
def test_ppg():
    try:
        ppg_main(cartpole_ppg_config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("19. ppg\n")


@pytest.mark.algotest
def test_sqn():
    config = [deepcopy(cartpole_sqn_config), deepcopy(cartpole_sqn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("20. sqn\n")


@pytest.mark.algotest
def test_qrdqn():
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("21. qrdqn\n")


@pytest.mark.algotest
def test_acer():
    config = [deepcopy(cartpole_acer_config), deepcopy(cartpole_acer_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("22. acer\n")


@pytest.mark.algotest
def test_selfplay():
    try:
        selfplay_main(deepcopy(league_demo_ppo_config), seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("23. selfplay\n")


@pytest.mark.algotest
def test_league():
    try:
        league_main(deepcopy(league_demo_ppo_config), seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("24. league\n")


@pytest.mark.algotest
def test_sqil():
    expert_policy_state_dict_path = './expert_policy.pth'
    config = [deepcopy(cartpole_sql_config), deepcopy(cartpole_sql_create_config)]
    expert_policy = serial_pipeline(config, seed=0)
    torch.save(expert_policy.collect_mode.state_dict(), expert_policy_state_dict_path)

    config = [deepcopy(cartpole_sqil_config), deepcopy(cartpole_sqil_create_config)]
    config[0].policy.collect.model_path = expert_policy_state_dict_path
    try:
        serial_pipeline_sqil(config, [cartpole_sql_config, cartpole_sql_create_config], seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("25. sqil\n")


@pytest.mark.algotest
def test_cql():
    # train expert
    config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    config[0].exp_name = 'sac'
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [deepcopy(pendulum_sac_data_genearation_config), deepcopy(pendulum_sac_data_genearation_create_config)]
    collect_count = config[0].policy.collect.n_sample
    expert_data_path = config[0].policy.collect.save_path
    state_dict = torch.load('./sac/ckpt/ckpt_best.pth.tar', map_location='cpu')
    try:
        collect_demo_data(
            config, seed=0, collect_count=collect_count, expert_data_path=expert_data_path, state_dict=state_dict
        )
    except Exception:
        assert False, "pipeline fail"

    # train cql
    config = [deepcopy(pendulum_cql_config), deepcopy(pendulum_cql_create_config)]
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("26. cql\n")


@pytest.mark.algotest
def test_discrete_cql():
    # train expert
    config = [deepcopy(cartpole_qrdqn_config), deepcopy(cartpole_qrdqn_create_config)]
    config[0].exp_name = 'cartpole'
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [deepcopy(cartpole_qrdqn_generation_data_config), deepcopy(cartpole_qrdqn_generation_data_create_config)]
    collect_count = config[0].policy.collect.collect_count
    state_dict = torch.load('cartpole/ckpt/ckpt_best.pth.tar', map_location='cpu')
    try:
        collect_demo_data(config, seed=0, collect_count=collect_count, state_dict=state_dict)
    except Exception:
        assert False, "pipeline fail"

    # train cql
    config = [deepcopy(cartpole_discrete_cql_config), deepcopy(cartpole_discrete_cql_create_config)]
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("27. discrete cql\n")


# @pytest.mark.algotest
def test_wqmix():
    config = [deepcopy(ptz_simple_spread_wqmix_config), deepcopy(ptz_simple_spread_wqmix_create_config)]
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("28. wqmix\n")


@pytest.mark.algotest
def test_td3_bc():
    # train expert
    config = [deepcopy(pendulum_td3_config), deepcopy(pendulum_td3_create_config)]
    config[0].exp_name = 'td3'
    try:
        serial_pipeline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"

    # collect expert data
    import torch
    config = [deepcopy(pendulum_td3_generation_config), deepcopy(pendulum_td3_generation_create_config)]
    collect_count = config[0].policy.other.replay_buffer.replay_buffer_size
    expert_data_path = config[0].policy.collect.save_path
    state_dict = torch.load(config[0].policy.learn.learner.load_path, map_location='cpu')
    try:
        collect_demo_data(
            config, seed=0, collect_count=collect_count, expert_data_path=expert_data_path, state_dict=state_dict
        )
    except Exception:
        assert False, "pipeline fail"

    # train td3 bc
    config = [deepcopy(pendulum_td3_bc_config), deepcopy(pendulum_td3_bc_create_config)]
    try:
        serial_pipeline_offline(config, seed=0)
    except Exception:
        assert False, "pipeline fail"
    with open("./algo_record.log", "a+") as f:
        f.write("29. td3_bc\n")


# @pytest.mark.algotest
def test_running_on_orchestrator():
    from kubernetes import config, client, dynamic
    cluster_name = 'test-k8s-launcher'
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'k8s-config.yaml')
    # create cluster
    launcher = K8sLauncher(config_path)
    launcher.name = cluster_name
    launcher.create_cluster()

    # create orchestrator
    olauncher = OrchestratorLauncher('v0.2.0-rc.0', cluster=launcher)
    olauncher.create_orchestrator()

    # create dijob
    namespace = 'default'
    name = 'cartpole-dqn'
    timeout = 20 * 60
    file_path = os.path.dirname(__file__)
    agconfig_path = os.path.join(file_path, 'config', 'agconfig.yaml')
    dijob_path = os.path.join(file_path, 'config', 'dijob-cartpole.yaml')
    create_object_from_config(agconfig_path, 'di-system')
    create_object_from_config(dijob_path, namespace)

    # watch for dijob to converge
    config.load_kube_config()
    dyclient = dynamic.DynamicClient(client.ApiClient(configuration=config.load_kube_config()))
    dijobapi = dyclient.resources.get(api_version='diengine.opendilab.org/v1alpha1', kind='DIJob')

    wait_for_dijob_condition(dijobapi, name, namespace, 'Succeeded', timeout)

    v1 = client.CoreV1Api()
    logs = v1.read_namespaced_pod_log(f'{name}-coordinator', namespace, tail_lines=20)
    print(f'\ncoordinator logs:\n {logs} \n')

    # delete dijob
    dijobapi.delete(name=name, namespace=namespace, body={})
    # delete orchestrator
    olauncher.delete_orchestrator()
    # delete k8s cluster
    launcher.delete_cluster()


def create_object_from_config(config_path: str, namespace: str = 'default'):
    args = ['kubectl', 'apply', '-n', namespace, '-f', config_path]
    proc = subprocess.Popen(args, stderr=subprocess.PIPE)
    _, err = proc.communicate()
    err_str = err.decode('utf-8').strip()
    if err_str != '' and 'WARN' not in err_str and 'already exists' not in err_str:
        raise RuntimeError(f'Failed to create object: {err_str}')


def delete_object_from_config(config_path: str, namespace: str = 'default'):
    args = ['kubectl', 'delete', '-n', namespace, '-f', config_path]
    proc = subprocess.Popen(args, stderr=subprocess.PIPE)
    _, err = proc.communicate()
    err_str = err.decode('utf-8').strip()
    if err_str != '' and 'WARN' not in err_str and 'NotFound' not in err_str:
        raise RuntimeError(f'Failed to delete object: {err_str}')


def wait_for_dijob_condition(dijobapi, name: str, namespace: str, phase: str, timeout: int = 60, interval: int = 1):
    start = time.time()
    dijob = dijobapi.get(name=name, namespace=namespace)
    while (dijob.status is None or dijob.status.phase != phase) and time.time() - start < timeout:
        time.sleep(interval)
        dijob = dijobapi.get(name=name, namespace=namespace)

    if dijob.status.phase == phase:
        return
    raise TimeoutError(f'Timeout waiting for DIJob: {name} to be {phase}')
