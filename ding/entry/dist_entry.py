import os
import sys
import subprocess
import signal
import pickle
from ditk import logging
import time
from threading import Thread
from easydict import EasyDict
import numpy as np
from ding.worker import Coordinator, create_comm_collector, create_comm_learner, LearnerAggregator
from ding.config import read_config_with_system, compile_config_parallel
from ding.utils import set_pkg_seed, DEFAULT_K8S_AGGREGATOR_SLAVE_PORT, pod_exec_command


def dist_prepare_config(
        filename: str,
        seed: int,
        platform: str,
        coordinator_host: str,
        learner_host: str,
        collector_host: str,
        coordinator_port: int,
        learner_port: int,
        collector_port,
) -> str:
    set_pkg_seed(seed)
    main_cfg, create_cfg, system_cfg = read_config_with_system(filename)
    config = compile_config_parallel(
        main_cfg,
        create_cfg=create_cfg,
        system_cfg=system_cfg,
        seed=seed,
        platform=platform,
        coordinator_host=coordinator_host,
        learner_host=learner_host,
        collector_host=collector_host,
        coordinator_port=coordinator_port,
        learner_port=learner_port,
        collector_port=collector_port,
    )
    # Pickle dump config to disk for later use.
    real_filename = filename + '.pkl'
    with open(real_filename, 'wb') as f:
        pickle.dump(config, f)
    return real_filename


def dist_launch_coordinator(
        filename: str,
        seed: int,
        coordinator_port: int,
        disable_flask_log: bool,
        enable_total_log: bool = False
) -> None:
    set_pkg_seed(seed)
    # Disable some part of DI-engine log
    if not enable_total_log:
        coordinator_log = logging.getLogger('coordinator_logger')
        coordinator_log.disabled = True
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    with open(filename, 'rb') as f:
        config = pickle.load(f)
    # CLI > ENV VARIABLE > CONFIG
    if coordinator_port is not None:
        config.system.coordinator.port = coordinator_port
    elif os.environ.get('COORDINATOR_PORT', None):
        port = os.environ['COORDINATOR_PORT']
        if port.isdigit():
            config.system.coordinator.port = int(port)
    else:  # use config pre-defined value
        assert 'port' in config.system.coordinator and np.isscalar(config.system.coordinator.port)
    coordinator = Coordinator(config)
    coordinator.start()

    # Monitor thread: Coordinator will remain running until its ``system_shutdown_flag`` is set to False.
    def shutdown_monitor():
        while True:
            time.sleep(3)
            if coordinator.system_shutdown_flag:
                coordinator.close()
                break

    shutdown_monitor_thread = Thread(target=shutdown_monitor, args=(), daemon=True, name='shutdown_monitor')
    shutdown_monitor_thread.start()
    shutdown_monitor_thread.join()
    print("[DI-engine dist pipeline]Your RL agent is converged, you can refer to 'log' and 'tensorboard' for details")


def dist_launch_learner(
        filename: str, seed: int, learner_port: int, name: str = None, disable_flask_log: bool = True
) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    if name is None:
        name = 'learner'
    with open(filename, 'rb') as f:
        config = pickle.load(f).system[name]
    # CLI > ENV VARIABLE > CONFIG
    if learner_port is not None:
        config.port = learner_port
    elif os.environ.get('LEARNER_PORT', None):
        port = os.environ['LEARNER_PORT']
        if port.isdigit():
            config.port = int(port)
    else:  # use config pre-defined value
        assert 'port' in config and np.isscalar(config.port)
    learner = create_comm_learner(config)
    learner.start()


def dist_launch_collector(
        filename: str, seed: int, collector_port: int, name: str = None, disable_flask_log: bool = True
) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    if name is None:
        name = 'collector'
    with open(filename, 'rb') as f:
        config = pickle.load(f).system[name]
    # CLI > ENV VARIABLE > CONFIG
    if collector_port is not None:
        config.port = collector_port
    elif os.environ.get('COLLECTOR_PORT', None):
        port = os.environ['COLLECTOR_PORT']
        if port.isdigit():
            config.port = int(port)
    else:  # use config pre-defined value
        assert 'port' in config and np.isscalar(config.port)
    collector = create_comm_collector(config)
    collector.start()


def dist_launch_learner_aggregator(
        filename: str,
        seed: int,
        aggregator_host: str,
        aggregator_port: int,
        name: str = None,
        disable_flask_log: bool = True
) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    if filename is not None:
        if name is None:
            name = 'learner_aggregator'
        with open(filename, 'rb') as f:
            config = pickle.load(f).system[name]
    else:
        # start without config (create a fake one)
        host, port = aggregator_host, DEFAULT_K8S_AGGREGATOR_SLAVE_PORT
        if aggregator_port is not None:
            port = aggregator_port
        elif os.environ.get('AGGREGATOR_PORT', None):
            _port = os.environ['AGGREGATOR_PORT']
            if _port.isdigit():
                port = int(_port)
        config = dict(
            master=dict(host=host, port=port + 1),
            slave=dict(host=host, port=port + 0),
            learner={},
        )
        config = EasyDict(config)
    learner_aggregator = LearnerAggregator(config)
    learner_aggregator.start()


def dist_launch_spawn_learner(
        filename: str, seed: int, learner_port: int, name: str = None, disable_flask_log: bool = True
) -> None:
    current_env = os.environ.copy()
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    processes = []

    for local_rank in range(0, local_world_size):
        dist_rank = int(os.environ.get('START_RANK', 0)) + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        executable = subprocess.getoutput('which ding')
        assert len(executable) > 0, "cannot find executable \"ding\""

        cmd = [executable, '-m', 'dist', '--module', 'learner']
        if filename is not None:
            cmd += ['-c', f'{filename}']
        if seed is not None:
            cmd += ['-s', f'{seed}']
        if learner_port is not None:
            cmd += ['-lp', f'{learner_port}']
        if name is not None:
            cmd += ['--module-name', f'{name}']
        if disable_flask_log is not None:
            cmd += ['--disable-flask-log', f'{int(disable_flask_log)}']

        sig_names = {2: "SIGINT", 15: "SIGTERM"}
        last_return_code = None

        def sigkill_handler(signum, frame):
            for process in processes:
                print(f"Killing subprocess {process.pid}")
                try:
                    process.kill()
                except Exception:
                    pass
            if last_return_code is not None:
                raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
            if signum in sig_names:
                print(f"Main process received {sig_names[signum]}, exiting")
            sys.exit(1)

        # pass SIGINT/SIGTERM to children if the parent is being terminated
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        process = subprocess.Popen(cmd, env=current_env, stdout=None, stderr=None)
        processes.append(process)

    try:
        alive_processes = set(processes)
        while len(alive_processes):
            finished_processes = []
            for process in alive_processes:
                if process.poll() is None:
                    # the process is still running
                    continue
                else:
                    if process.returncode != 0:
                        last_return_code = process.returncode  # for sigkill_handler
                        sigkill_handler(signal.SIGTERM, None)  # not coming back
                    else:
                        # exited cleanly
                        finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)

            time.sleep(1)
    finally:
        # close open file descriptors
        pass


def dist_add_replicas(
        replicas_type: str,
        kubeconfig: str,
        replicas: int,
        coordinator_name: str,
        namespace: str,
        cpus: int,
        gpus: int,
        memory: str,
) -> None:
    assert coordinator_name and namespace, "Please provide --coordinator-name or --namespace"

    import json
    data = {
        "namespace": namespace,
        "coordinator": coordinator_name,
    }
    res = {"replicas": replicas}
    if cpus > 0:
        res['cpus'] = cpus
    if gpus > 0:
        res['gpus'] = gpus
    if memory:
        res['memory'] = memory
    if replicas_type == 'collector':
        data['collectors'] = res
    elif replicas_type == 'learner':
        data['learners'] = res
    cmd = 'curl -X POST $KUBERNETES_SERVER_URL/v1alpha1/replicas ' \
        '-H "content-type: application/json" ' \
        f'-d \'{json.dumps(data)}\''
    ret, msg = pod_exec_command(kubeconfig, coordinator_name, namespace, cmd)
    if ret == 0:
        print(f'{replicas_type} add successfully')
    else:
        print(f'Failed to add {replicas_type}, return code: {ret}, message: {msg}')


def dist_delete_replicas(
        replicas_type: str, kubeconfig: str, replicas: int, coordinator_name: str, namespace: str
) -> None:
    assert coordinator_name and namespace, "Please provide --coordinator-name or --namespace"

    import json
    data = {
        "namespace": namespace,
        "coordinator": coordinator_name,
    }
    if replicas_type == 'collector':
        data['collectors'] = {"replicas": replicas}
    elif replicas_type == 'learner':
        data['learners'] = {"replicas": replicas}
    cmd = 'curl -X DELETE $KUBERNETES_SERVER_URL/v1alpha1/replicas ' \
        '-H "content-type: application/json" ' \
        f'-d \'{json.dumps(data)}\''
    ret, msg = pod_exec_command(kubeconfig, coordinator_name, namespace, cmd)
    if ret == 0:
        print(f'{replicas_type} delete successfully')
    else:
        print(f'Failed to delete {replicas_type}, return code: {ret}, message: {msg}')


def dist_restart_replicas(
        replicas_type: str, kubeconfig: str, coordinator_name: str, namespace: str, restart_pod_name: str
) -> None:
    assert coordinator_name and namespace, "Please provide --coordinator-name or --namespace"

    import json
    data = {
        "namespace": namespace,
        "coordinator": coordinator_name,
    }
    assert restart_pod_name, "Please provide restart pod name with --restart-pod-name"
    if replicas_type == 'collector':
        data['collectors'] = [restart_pod_name]
    elif replicas_type == 'learner':
        data['learners'] = [restart_pod_name]
    cmd = 'curl -X POST $KUBERNETES_SERVER_URL/v1alpha1/replicas/failed ' \
        '-H "content-type: application/json" ' \
        f'-d \'{json.dumps(data)}\''
    ret, msg = pod_exec_command(kubeconfig, coordinator_name, namespace, cmd)
    if ret == 0:
        print(f'{replicas_type} restart successfully')
    else:
        print(f'Failed to restart {replicas_type}, return code: {ret}, message: {msg}')
