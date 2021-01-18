from typing import Optional, List
import subprocess
import time
import pickle
import logging
from threading import Thread
from easydict import EasyDict
from nervex.worker import create_comm_learner, create_comm_actor, Coordinator, LearnerAggregator
from nervex.config import Config, parallel_transform, parallel_transform_slurm


def parallel_pipeline(
        filename: str,
        seed: int,
        platform: str,
        coordinator_host: Optional[str] = None,
        learner_host: Optional[List[str]] = None,
        actor_host: Optional[List[str]] = None
) -> None:
    # disable flask logger
    log = logging.getLogger('werkzeug')
    log.disabled = True
    if platform == 'local':
        cfg = Config.file_to_dict(filename).cfg_dict
        config = cfg['main_config']
        config = parallel_transform(config, coordinator_host, learner_host, actor_host)
        for k, v in config.items():
            if 'learner' in k:
                launch_learner(v)
            elif 'actor' in k:
                launch_actor(v)
        launch_coordinator(config.coordinator)
    elif platform == 'slurm':
        cfg = Config.file_to_dict(filename).cfg_dict
        config = cfg['main_config']
        config = parallel_transform_slurm(config, coordinator_host, learner_host, actor_host)
        real_filename = filename + '.pkl'
        with open(real_filename, 'wb') as f:
            pickle.dump(config, f)
        for k, v in config.items():
            if 'learner' in k:
                use_aggregator = v.get('use_aggregator', False)
                num = v.get('repeat_num', 1)
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
        # coordinator run in manager node
        subprocess.run(
            "python -c \"import nervex.entry.parallel_entry as pe; pe.launch_coordinator(filename='{}')\"".
            format(real_filename),
            stderr=subprocess.STDOUT,
            shell=True,
        )
    elif platform == 'k8s':
        raise NotImplementedError


def launch_learner(config: Optional[dict] = None, filename: Optional[str] = None, name: Optional[str] = None) -> None:
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f)[name]
    learner = create_comm_learner(config)
    learner.start()


def launch_learner_aggregator(filename: Optional[str] = None, name: Optional[str] = None) -> None:
    with open(filename, 'rb') as f:
        config = pickle.load(f)[name]
    aggregator = LearnerAggregator(config)
    aggregator.start()


def launch_actor(config: Optional[dict] = None, filename: Optional[str] = None, name: Optional[str] = None) -> None:
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f)[name]
    actor = create_comm_actor(config)
    actor.start()


def launch_coordinator(config: Optional[EasyDict] = None, filename: Optional[str] = None) -> None:
    if config is None:
        with open(filename, 'rb') as f:
            config = pickle.load(f).coordinator
    coordinator = Coordinator(config)
    coordinator.start()

    # monitor thread
    def shutdown_monitor():
        while True:
            time.sleep(3)
            if coordinator.system_shutdown_flag:
                coordinator.close()
                break

    shutdown_monitor_thread = Thread(target=shutdown_monitor, args=(), daemon=True)
    shutdown_monitor_thread.start()
