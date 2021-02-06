from typing import Optional, List
import subprocess
import time
import pickle
import logging
from threading import Thread, Event
from easydict import EasyDict
from nervex.worker import create_comm_learner, create_comm_actor, Coordinator, LearnerAggregator
from nervex.config import Config, parallel_transform, parallel_transform_slurm


def parallel_pipeline(
        filename: str,
        seed: int,
        enable_total_log: Optional[bool] = False,
        disable_flask_log: Optional[bool] = True,
) -> None:
    r"""
    Overview:
        Parallel pipeline entry.
    Arguments:
        - filename (:obj:`str`): Config file path.
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
    # Parallel job launch.
    cfg = Config.file_to_dict(filename).cfg_dict
    config = cfg['main_config']
    config = parallel_transform(config)
    learner_handle = []
    actor_handle = []
    for k, v in config.items():
        if 'learner' in k:
            learner_handle.append(launch_learner(v))
        elif 'actor' in k:
            actor_handle.append(launch_actor(v))
    launch_coordinator(config.coordinator, learner_handle=learner_handle, actor_handle=actor_handle)


# Following functions are used to launch different components(learner, learner aggregator, actor, coordinator).
# Argument ``config`` is the dict type config. If it is None, then ``filename`` and ``name`` must be passed,
# for they can be used to read correponding config from file.
def launch_learner(config: Optional[dict] = None, filename: Optional[str] = None, name: Optional[str] = None) -> list:
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


def launch_actor(config: Optional[dict] = None, filename: Optional[str] = None, name: Optional[str] = None) -> list:
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f)[name]
    start_actor_event = Event()
    close_actor_event = Event()

    def run_actor():
        actor = create_comm_actor(config)
        actor.start()
        start_actor_event.set()
        close_actor_event.wait()
        actor.close()

    actor_thread = Thread(target=run_actor, args=(), name='actor_entry_thread')
    actor_thread.start()
    return actor_thread, start_actor_event, close_actor_event


def launch_coordinator(
        config: Optional[EasyDict] = None,
        filename: Optional[str] = None,
        learner_handle: Optional[list] = None,
        actor_handle: Optional[list] = None
) -> None:
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f).coordinator
    coordinator = Coordinator(config)
    for _, start_event, _ in learner_handle:
        start_event.wait()
    for _, start_event, _ in actor_handle:
        start_event.wait()
    coordinator.start()

    # Monitor thread: Coordinator will remain running until its ``system_shutdown_flag`` is set to False.
    def shutdown_monitor():
        while True:
            time.sleep(3)
            if coordinator.system_shutdown_flag:
                coordinator.close()
                for _, _, close_event in learner_handle:
                    close_event.set()
                for _, _, close_event in actor_handle:
                    close_event.set()
                break

    shutdown_monitor_thread = Thread(target=shutdown_monitor, args=(), daemon=True, name='shutdown_monitor')
    shutdown_monitor_thread.start()


def launch_learner_aggregator(filename: Optional[str] = None, name: Optional[str] = None) -> None:
    with open(filename, 'rb') as f:
        config = pickle.load(f)[name]
    aggregator = LearnerAggregator(config)
    aggregator.start()


def dist_pipeline(
        filename: str,
        seed: int,
        platform: str,
        coordinator_host: Optional[str] = None,
        learner_host: Optional[List[str]] = None,
        actor_host: Optional[List[str]] = None,
        enable_total_log: Optional[bool] = False,
        disable_flask_log: Optional[bool] = True,
) -> None:
    r"""
    Overview:
        Distributed parallel pipeline entry.
    Arguments:
        - filename (:obj:`str`): Config file path.
        - seed (:obj:`int`): Random seed.
        - platform (:obj:`str`): The Platform to launch job. Now supports ["local", "slurm"].
        - coordinator_host (:obj:`Optional[str]`): Coordinator host.
        - learner_host (:obj:`Optional[str]`): Learner host.
        - actor_host (:obj:`Optional[str]`): Actor host.
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
    if platform == 'local':
        raise NotImplementedError
    elif platform == 'slurm':
        cfg = Config.file_to_dict(filename).cfg_dict
        config = cfg['main_config']
        config = parallel_transform_slurm(config, coordinator_host, learner_host, actor_host)
        # Pickle dump config to disk for later use.
        real_filename = filename + '.pkl'
        with open(real_filename, 'wb') as f:
            pickle.dump(config, f)
        for k, v in config.items():
            if 'learner' in k:
                # Learner launch will be different whether to use learner aggregator or not.
                use_aggregator = v.get('use_aggregator', False)
                num = v.get('repeat_num', 1)
                # Learner launch.
                if use_aggregator:
                    srun_args = "srun --mpi=pmi2 -p {} -w {} -n{} --gres=gpu:{}".format(v.partition, v.node, num, num)
                else:
                    srun_args = "srun -p {} -w {} --gres=gpu:1".format(v.partition, v.node)
                subprocess.Popen(
                    "{} --job-name=learner python -c \
                    \"import nervex.entry.parallel_entry as pe; pe.launch_learner(filename='{}', name='{}')\"".format(
                        srun_args, real_filename, k
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                )
                # Additional learner aggregator launch.
                if use_aggregator:
                    aggregator_k = 'aggregator' + k[7:]
                    aggregator_cfg = config[aggregator_k]
                    subprocess.Popen(
                        "srun -p {} -w {} --job-name=learner_aggregator python -c \
                        \"import nervex.entry.parallel_entry as pe; pe.launch_learner_aggregator(filename='{}', \
                        name='{}')\"".format(
                            aggregator_cfg.partition, aggregator_cfg.node, real_filename, aggregator_k
                        ),
                        stderr=subprocess.STDOUT,
                        shell=True,
                    )
            elif 'actor' in k:
                subprocess.Popen(
                    "srun -p {} -w {} --gres=gpu:1 --job-name=actor python -c \
                    \"import nervex.entry.parallel_entry as pe; pe.launch_actor(filename='{}', name='{}')\"".format(
                        v.partition, v.node, real_filename, k
                    ),
                    stderr=subprocess.STDOUT,
                    shell=True,
                )
        # Coordinator run in manager node.
        subprocess.run(
            "python -c \"import nervex.entry.parallel_entry as pe; pe.launch_coordinator(filename='{}')\"".
            format(real_filename),
            stderr=subprocess.STDOUT,
            shell=True,
        )
    elif platform == 'k8s':
        raise NotImplementedError
