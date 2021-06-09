import os
import pickle
import logging
import time
from threading import Thread
from nervex.worker import Coordinator, create_comm_collector, create_comm_learner, LearnerAggregator
from nervex.config import read_config, compile_config_parallel
from nervex.utils import set_pkg_seed


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
    main_cfg, create_cfg, system_cfg = read_config(filename)
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
    # Disable some part nervex log
    if not enable_total_log:
        coordinator_log = logging.getLogger('coordinator_logger')
        # coordinator_log.disabled = True
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    with open(filename, 'rb') as f:
        config = pickle.load(f)
    if coordinator_port is not None:
        config.system.coordinator.port = coordinator_port
    elif os.environ.get('COORDINATOR_PORT', None):
        port = os.environ['COORDINATOR_PORT']
        if port.isdigit():
            config.system.coordinator.port = int(port)
    else:
        pass
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
    print("[nerveX dist pipeline]Your RL agent is converged, you can refer to 'log' and 'tensorboard' for details")


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
    if learner_port is not None:
        config.port = learner_port
    elif os.environ.get('LEARNER_PORT', None):
        port = os.environ['LEARNER_PORT']
        if port.isdigit():
            config.port = int(port)
    else:
        pass
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
    if collector_port is not None:
        config.port = collector_port
    elif os.environ.get('COLLECTOR_PORT', None):
        port = os.environ['COLLECTOR_PORT']
        if port.isdigit():
            config.port = int(port)
    else:
        pass
    collector = create_comm_collector(config)
    collector.start()


def dist_launch_learner_aggregator(filename: str, seed: int, name: str = None, disable_flask_log: bool = True) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    if name is None:
        name = 'learner_aggregator'
    with open(filename, 'rb') as f:
        config = pickle.load(f).system[name]
    learner_aggregator = LearnerAggregator(config)
    learner_aggregator.start()
