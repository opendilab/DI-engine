from typing import Union, Dict, Any, List
from abc import ABC, abstractmethod
import copy
from easydict import EasyDict

from ding.utils import import_module, BUFFER_REGISTRY


class IBuffer(ABC):
    r"""
    Overview:
        Buffer interface
    Interfaces:
        default_config, push, update, sample, clear, count, state_dict, load_state_dict
    """

    @classmethod
    def default_config(cls) -> EasyDict:
        r"""
        Overview:
            Default config of this buffer class.
        Returns:
            - default_config (:obj:`EasyDict`)
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def push(self, data: Union[List[Any], Any], cur_collector_envstep: int) -> None:
        r"""
        Overview:
            Push a data into buffer.
        Arguments:
            - data (:obj:`Union[List[Any], Any]`): The data which will be pushed into buffer. Can be one \
                (in `Any` type), or many(int `List[Any]` type).
            - cur_collector_envstep (:obj:`int`): Collector's current env step.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, info: Dict[str, list]) -> None:
        r"""
        Overview:
            Update data info, e.g. priority.
        Arguments:
            - info (:obj:`Dict[str, list]`): Info dict. Keys depends on the specific buffer type.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, cur_learner_iter: int) -> list:
        r"""
        Overview:
            Sample data with length ``batch_size``.
        Arguments:
            - size (:obj:`int`): The number of the data that will be sampled.
            - cur_learner_iter (:obj:`int`): Learner's current iteration.
        Returns:
            - sampled_data (:obj:`list`): A list of data with length `batch_size`.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Overview:
            Clear all the data and reset the related variables.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Overview:
            Count how many valid datas there are in the buffer.
        Returns:
            - count (:obj:`int`): Number of valid data.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        Overview:
            Provide a state dict to keep a record of current buffer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer. \
                With the dict, one can easily reproduce the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, _state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load state dict to reproduce the buffer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.
        """
        raise NotImplementedError


def create_buffer(cfg: EasyDict, *args, **kwargs) -> IBuffer:
    r"""
    Overview:
        Create a buffer according to cfg and other arguments.
    Arguments:
        - cfg (:obj:`EasyDict`): Buffer config.
    ArgumentsKeys:
        - necessary: `type`
    """
    import_module(cfg.get('import_names', []))
    if cfg.type == 'naive':
        kwargs.pop('tb_logger', None)
    return BUFFER_REGISTRY.build(cfg.type, cfg, *args, **kwargs)


def get_buffer_cls(cfg: EasyDict) -> type:
    r"""
    Overview:
        Get a buffer class according to cfg.
    Arguments:
        - cfg (:obj:`EasyDict`): Buffer config.
    ArgumentsKeys:
        - necessary: `type`
    """
    import_module(cfg.get('import_names', []))
    return BUFFER_REGISTRY.get(cfg.type)
