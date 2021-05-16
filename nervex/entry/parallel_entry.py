from typing import Optional, List, Union, Tuple
import subprocess
import time
import pickle
import logging
from threading import Thread, Event
from easydict import EasyDict

from nervex.worker import create_comm_learner, create_comm_collector, Coordinator, LearnerAggregator
from nervex.config import read_config, compile_config_parallel
from nervex.utils import set_pkg_seed


def parallel_pipeline(
        input_cfg: Union[str, Tuple[dict, dict, dict]],
        seed: int,
        enable_total_log: Optional[bool] = False,
        disable_flask_log: Optional[bool] = True,
) -> None:
    r"""
    Overview:
        Parallel pipeline entry.
    Arguments:
        - config (:obj:`Union[str, dict]`): Config file path.
        - seed (:obj:`int`): Random seed.
        - enable_total_log (:obj:`Optional[bool]`): whether enable total nervex system log
        - disable_flask_log (:obj:`Optional[bool]`): whether disable flask log
    """
    # Disable some part nervex log
    if not enable_total_log:
        coordinator_log = logging.getLogger('coordinator_logger')
        coordinator_log.disabled = True
    # Disable flask logger.
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    set_pkg_seed(seed)
    # Parallel job launch.
    if isinstance(input_cfg, str):
        main_cfg, create_cfg, system_cfg = read_config(input_cfg)
    elif isinstance(input_cfg, tuple) or isinstance(input_cfg, list):
        main_cfg, create_cfg, system_cfg = input_cfg
    else:
        raise TypeError("invalid config type: {}".format(input_cfg))
    config = compile_config_parallel(main_cfg, create_cfg=create_cfg, system_cfg=system_cfg)
    learner_handle = []
    collector_handle = []
    for k, v in config.system.items():
        if 'learner' in k:
            learner_handle.append(launch_learner(seed, v))
        elif 'collector' in k:
            collector_handle.append(launch_collector(seed, v))
    launch_coordinator(seed, config, learner_handle=learner_handle, collector_handle=collector_handle)


# Following functions are used to launch different components(learner, learner aggregator, collector, coordinator).
# Argument ``config`` is the dict type config. If it is None, then ``filename`` and ``name`` must be passed,
# for they can be used to read correponding config from file.
def launch_learner(
        seed: int, config: Optional[dict] = None, filename: Optional[str] = None, name: Optional[str] = None
) -> list:
    set_pkg_seed(seed)
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f)[name]
    start_learner_event = Event()
    close_learner_event = Event()

    def run_learner():
        learner = create_comm_learner(config)
        learner.start()
        start_learner_event.set()
        close_learner_event.wait()
        learner.close()

    learner_thread = Thread(target=run_learner, args=(), name='learner_entry_thread')
    learner_thread.start()
    return learner_thread, start_learner_event, close_learner_event


def launch_collector(
        seed: int, config: Optional[dict] = None, filename: Optional[str] = None, name: Optional[str] = None
) -> list:
    set_pkg_seed(seed)
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f)[name]
    start_collector_event = Event()
    close_collector_event = Event()

    def run_collector():
        collector = create_comm_collector(config)
        collector.start()
        start_collector_event.set()
        close_collector_event.wait()
        collector.close()

    collector_thread = Thread(target=run_collector, args=(), name='collector_entry_thread')
    collector_thread.start()
    return collector_thread, start_collector_event, close_collector_event


def launch_coordinator(
        seed: int,
        config: Optional[EasyDict] = None,
        filename: Optional[str] = None,
        learner_handle: Optional[list] = None,
        collector_handle: Optional[list] = None
) -> None:
    set_pkg_seed(seed)
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f)
    coordinator = Coordinator(config)
    for _, start_event, _ in learner_handle:
        start_event.wait()
    for _, start_event, _ in collector_handle:
        start_event.wait()
    coordinator.start()
    system_shutdown_event = Event()

    # Monitor thread: Coordinator will remain running until its ``system_shutdown_flag`` is set to False.
    def shutdown_monitor():
        while True:
            time.sleep(3)
            if coordinator.system_shutdown_flag:
                coordinator.close()
                for _, _, close_event in learner_handle:
                    close_event.set()
                for _, _, close_event in collector_handle:
                    close_event.set()
                system_shutdown_event.set()
                break

    shutdown_monitor_thread = Thread(target=shutdown_monitor, args=(), daemon=True, name='shutdown_monitor')
    shutdown_monitor_thread.start()
    system_shutdown_event.wait()
    print("[nerveX parallel pipeline]Your RL agent is converged, you can refer to 'log' and 'tensorboard' for details")


def launch_learner_aggregator(seed: int, filename: Optional[str] = None, name: Optional[str] = None) -> None:
    set_pkg_seed(seed)
    with open(filename, 'rb') as f:
        config = pickle.load(f)[name]
    aggregator = LearnerAggregator(config)
    aggregator.start()
