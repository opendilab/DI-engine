from easydict import EasyDict
from nervex.worker import create_comm_learner, create_comm_actor, Coordinator
from nervex.config import Config


def parallel_pipeline(filename: str, seed: int):
    cfg = Config.file_to_dict(filename).cfg_dict
    main_config = cfg['main_config']
    for k, v in main_config.items():
        if 'learner' in k:
            launch_learner(v)
        elif 'actor' in k:
            launch_actor(v)
    launch_coordinator(main_config['coordinator'])


def launch_learner(config: dict) -> None:
    config = EasyDict(config)
    learner = create_comm_learner(config)
    learner.start()


def launch_actor(config: dict) -> None:
    config = EasyDict(config)
    actor = create_comm_actor(config)
    actor.start()


def launch_coordinator(config: dict) -> None:
    config = EasyDict(config)
    coordinator = Coordinator(config)
    coordinator.start()
