import gym
import copy
import numpy as np
from typing import Any, Dict, Optional
from easydict import EasyDict
from itertools import product
from ding.envs.env.base_env import BaseEnvTimestep
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils.data_helper import to_ndarray
import matplotlib.pyplot as plt
from typing import NoReturn, Optional, List
from gym import utils
from abc import ABC, abstractmethod

class AAA():
    def __init__(self) -> None:
        self.x = 0


def deep_update(
    original: dict,
    new_dict: dict,
    new_keys_allowed: bool = False,
    whitelist: Optional[List[str]] = None,
    override_all_if_type_changes: Optional[List[str]] = None
):
    """
    Overview:
        Updates original dict with values from new_dict recursively.

    .. note::

        If new key is introduced in new_dict, then if new_keys_allowed is not
        True, an error will be thrown. Further, for sub-dicts, if the key is
        in the whitelist, then new subkeys can be introduced.

    Arguments:
        - original (:obj:`dict`): Dictionary with default values.
        - new_dict (:obj:`dict`): Dictionary with values to be updated
        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.
        - whitelist (Optional[List[str]]): List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original

def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        merge two dict using deep_update
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])

    return EasyDict(merged)


class BaseDriveEnv(gym.Env, utils.EzPickle):
    """
    Base class for environments. It is inherited from `gym.Env` and uses the same interfaces.
    All Drive Env class is supposed to inherit from this class.

    Note:
        To run Reinforcement Learning on DI-engine platform, the environment should be wrapped with `DingEnvWrapper`.

    :Arguments:
        - cfg (Dict): Config Dict.

    :Interfaces: reset, step, close, seed
    """
    config = dict()

    @abstractmethod
    def __init__(self, cfg: Dict, **kwargs) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        utils.EzPickle.__init__(self)

    @abstractmethod
    def step(self, action: Any) -> Any:
        """
        Run one step of the environment and return the observation dict.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> Any:
        """
        Reset current environment.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release all resources in environment and close.
        """
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Set random seed.
        """
        raise NotImplementedError

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
