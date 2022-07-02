from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Any, Optional, Union
from collections import namedtuple
from easydict import EasyDict
import copy

from ding.envs import BaseEnvManager
from ding.utils import SERIAL_COLLECTOR_REGISTRY, import_module
from ding.torch_utils import to_tensor

INF = float("inf")


class ISerialCollector(ABC):
    """
    Overview:
        Abstract baseclass for serial collector.
    Interfaces:
        default_config, reset_env, reset_policy, reset, collect
    Property:
        envstep
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Get collector's default config. We merge collector's default config with other default configs\
                and user's config to get the final config.
        Return:
            cfg: (:obj:`EasyDict`): collector's default config
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset collector's environment. In some case, we need collector use the same policy to collect \
                data in different environments. We can use reset_env to reset the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset collector's policy. In some case, we need collector work in this same environment but use\
                different policy to collect data. We can use reset_policy to reset the policy.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset collector's policy and environment. Use new policy and environment to collect data.
        """
        raise NotImplementedError

    @abstractmethod
    def collect(self, per_collect_target: Any) -> List[Any]:
        """
        Overview:
            Collect the corresponding data according to the specified target and return. \
                There are different definitions in episode and sample mode.
        """
        raise NotImplementedError

    @abstractproperty
    def envstep(self) -> int:
        """
        Overview:
            Get the total envstep num.
        """
        raise NotImplementedError


def create_serial_collector(cfg: EasyDict, **kwargs) -> ISerialCollector:
    """
    Overview:
        Create a specific collector instance based on the config.
    """
    import_module(cfg.get('import_names', []))
    return SERIAL_COLLECTOR_REGISTRY.build(cfg.type, cfg=cfg, **kwargs)


def get_serial_collector_cls(cfg: EasyDict) -> type:
    """
    Overview:
        Get the specific collector class according to the config.
    """
    assert hasattr(cfg, 'type'), "{}-{}-{}".format(type(cfg), cfg.keys(), cfg['type'])
    import_module(cfg.get('import_names', []))
    return SERIAL_COLLECTOR_REGISTRY.get(cfg.type)


class CachePool(object):
    """
    Overview:
       CachePool is the repository of cache items.
    Interfaces:
        __init__, update, __getitem__, reset
    """

    def __init__(self, name: str, env_num: int, deepcopy: bool = False) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - name (:obj:`str`): name of cache
            - env_num (:obj:`int`): number of environments
            - deepcopy (:obj:`bool`): whether to deepcopy data
        """
        self._pool = [None for _ in range(env_num)]
        # TODO(nyz) whether must use deepcopy
        self._deepcopy = deepcopy

    def update(self, data: Union[Dict[int, Any], list]) -> None:
        """
        Overview:
            Update elements in cache pool.
        Arguments:
            - data (:obj:`Dict[int, Any]`): A dict containing update index-value pairs. Key is index in cache pool, \
                and value is the new element.
        """
        if isinstance(data, dict):
            data = [data]
        for index in range(len(data)):
            for i in data[index].keys():
                d = data[index][i]
                if self._deepcopy:
                    copy_d = copy.deepcopy(d)
                else:
                    copy_d = d
                if index == 0:
                    self._pool[i] = [copy_d]
                else:
                    self._pool[i].append(copy_d)

    def __getitem__(self, idx: int) -> Any:
        """
        Overview:
            Get item in cache pool.
        Arguments:
            - idx (:obj:`int`): The index of the item we need to get.
        Return:
            - item (:obj:`Any`): The item we get.
        """
        data = self._pool[idx]
        if data is not None and len(data) == 1:
            data = data[0]
        return data

    def reset(self, idx: int) -> None:
        """
        Overview:
            Reset the cache pool.
        Arguments:
            - idx (:obj:`int`): The index of the position we need to reset.
        """
        self._pool[idx] = None


class TrajBuffer(list):
    """
    Overview:
       TrajBuffer is used to store traj_len pieces of transitions.
    Interfaces:
        __init__, append
    """

    def __init__(self, maxlen: int, *args, **kwargs) -> None:
        """
        Overview:
            Initialization trajBuffer.
        Arguments:
            - maxlen (:obj:`int`): The maximum length of trajectory buffer.
        """
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def append(self, data: Any) -> None:
        """
        Overview:
            Append data to trajBuffer.
        """
        if self._maxlen is not None:
            while len(self) >= self._maxlen:
                del self[0]
        super().append(data)


def to_tensor_transitions(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Overview:
        transitions collected data to tensor.
    Argument:
        - data (:obj:`List[Dict[str, Any]]`): the data that will be transited to tensor.
    Return:
        - data (:obj:`List[Dict[str, Any]]`): the data that can be transited to tensor.

    .. tip::
        In order to save memory, If there are next_obs in the passed data, we do special \
            treatment on next_obs so that the next_obs of each state in the data fragment is \
            the next state's obs and the next_obs of the last state is its own next_obs, \
            and we make transform_scalar is False.
    """
    if 'next_obs' not in data[0]:
        return to_tensor(data, transform_scalar=False)
    else:
        # for save memory
        data = to_tensor(data, ignore_keys=['next_obs'], transform_scalar=False)
        for i in range(len(data) - 1):
            data[i]['next_obs'] = data[i + 1]['obs']
        data[-1]['next_obs'] = to_tensor(data[-1]['next_obs'], transform_scalar=False)
        return data
