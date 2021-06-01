from abc import ABC, abstractmethod, abstractproperty
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from collections import namedtuple, deque
from easydict import EasyDict
import copy

from nervex.envs import BaseEnvManager
from nervex.utils import SERIAL_COLLECTOR_REGISTRY, import_module
from nervex.torch_utils import to_tensor

INF = float("inf")


class ISerialCollector(ABC):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def collect(self, per_collect_target: Any) -> List[Any]:
        raise NotImplementedError

    @abstractproperty
    def envstep(self) -> int:
        raise NotImplementedError


def create_serial_collector(cfg: EasyDict) -> ISerialCollector:
    import_module(cfg.get('import_names', []))
    return SERIAL_COLLECTOR_REGISTRY.build(cfg.type, cfg=cfg)


def get_serial_collector_cls(cfg: EasyDict) -> type:
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
        data = self._pool[idx]
        if len(data) == 1:
            data = data[0]
        return data

    def reset(self, idx: int) -> None:
        self._pool[idx] = None


class TrajBuffer(list):

    def __init__(self, maxlen: int, *args, **kwargs) -> None:
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def append(self, data: Any) -> None:
        while len(self) >= self._maxlen:
            del self[0]
        super().append(data)


def to_tensor_transitions(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if 'next_obs' not in data[0]:
        return to_tensor(data, transform_scalar=False)
    else:
        # for save memory
        data = to_tensor(data, ignore_keys=['next_obs'], transform_scalar=False)
        for i in range(len(data) - 1):
            data[i]['next_obs'] = data[i + 1]['obs']
        data[-1]['next_obs'] = to_tensor(data[-1]['next_obs'])
        return data
