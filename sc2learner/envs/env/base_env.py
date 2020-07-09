from collections import namedtuple
from typing import Any


class BaseEnv:
    """
    Overview: basic environment class
    Interface: __init__
    Property: timestep
    """
    timestep = namedtuple('BaseEnvTimestep', ['obs', 'act', 'reward', 'done', 'info'])
    info_template = namedtuple('BaseEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        raise NotImplementedError

    def reset(self) -> 'BaseEnv.timestep':
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def step(self, action: Any) -> 'BaseEnv.timestep':
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def info(self) -> 'BaseEnv.info':
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
