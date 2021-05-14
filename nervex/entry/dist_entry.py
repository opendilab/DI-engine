import pickle
import logging
from nervex.worker import Coordinator, create_comm_collector, create_comm_learner
from nervex.config import read_config, parallel_transform, parallel_transform_slurm
from nervex.utils import set_pkg_seed


def dist_prepare_config(
        filename: str, seed: int, platform: str, coordinator_host: str, learner_host: str, collector_host: str
) -> str:
    set_pkg_seed(seed)
    config = read_config(filename)
    if platform == 'local':
        config = parallel_transform(config, coordinator_host, learner_host, collector_host)
    elif platform == 'slurm':
        config = parallel_transform_slurm(config, coordinator_host, learner_host, collector_host)
    elif platform == 'k8s':
        raise NotImplementedError
    # Pickle dump config to disk for later use.
    real_filename = filename + '.pkl'
    with open(real_filename, 'wb') as f:
        pickle.dump(config, f)
    return real_filename


def dist_launch_coordinator(filename: str, seed: int, disable_flask_log: bool) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    with open(filename, 'rb') as f:
        config = pickle.load(f).coordinator
    coordinator = Coordinator(config)
    coordinator.start()


def dist_launch_learner(filename: str, seed: int, name: str = None, disable_flask_log: bool = True) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    if name is None:
        name = 'learner'
    with open(filename, 'rb') as f:
        config = pickle.load(f)[name]
    learner = create_comm_learner(config)
    learner.start()


def dist_launch_collector(filename: str, seed: int, name: str = None, disable_flask_log: bool = True) -> None:
    set_pkg_seed(seed)
    if disable_flask_log:
        log = logging.getLogger('werkzeug')
        log.disabled = True
    if name is None:
        name = 'collector'
    with open(filename, 'rb') as f:
        config = pickle.load(f)[name]
    collector = create_comm_collector(config)
    collector.start()
