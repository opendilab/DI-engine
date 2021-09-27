from abc import ABC, abstractmethod
from typing import Any
from easydict import EasyDict

from ding.utils import get_task_uid, import_module, COMM_COLLECTOR_REGISTRY
from ..base_parallel_collector import create_parallel_collector, BaseParallelCollector


class BaseCommCollector(ABC):
    """
    Overview:
        Abstract baseclass for common collector.
    Interfaces:
        __init__, get_policy_update_info, send_metadata, send_stepdata
        start, close, _create_collector
    Property:
        collector_uid
    """

    def __init__(self, cfg):
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
        """
        self._cfg = cfg
        self._end_flag = True
        self._collector_uid = get_task_uid()

    @abstractmethod
    def get_policy_update_info(self, path: str) -> Any:
        """
        Overview:
            Get policy information in corresponding path.
            Will be registered in base collector.
        Arguments:
            - path (:obj:`str`): path to policy update information.
        """
        raise NotImplementedError

    @abstractmethod
    def send_metadata(self, metadata: Any) -> None:
        """
        Overview:
            Store meta data in queue, which will be retrieved by callback function "deal_with_collector_data"
            in collector slave, then will be sent to coordinator.
            Will be registered in base collector.
        Arguments:
            - metadata (:obj:`Any`): meta data.
        """
        raise NotImplementedError

    @abstractmethod
    def send_stepdata(self, stepdata: Any) -> None:
        """
        Overview:
            Save step data in corresponding path.
            Will be registered in base collector.
        Arguments:
            - stepdata (:obj:`Any`): step data.
        """
        raise NotImplementedError

    def start(self) -> None:
        """
        Overview:
            Start comm collector.
        """
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close comm collector.
        """
        self._end_flag = True

    @property
    def collector_uid(self) -> str:
        return self._collector_uid

    def _create_collector(self, task_info: dict) -> BaseParallelCollector:
        """
        Overview:
            Receive ``task_info`` passed from coordinator and create a collector.
        Arguments:
            - task_info (:obj:`dict`): Task info dict from coordinator. Should be like \
        Returns:
            - collector (:obj:`BaseParallelCollector`): Created base collector.
        Note:
            Four methods('send_metadata', 'send_stepdata', 'get_policy_update_info'), and policy are set.
            The reason why they are set here rather than base collector is, they highly depend on the specific task.
            Only after task info is passed from coordinator to comm collector through learner slave, can they be
            clarified and initialized.
        """
        collector_cfg = EasyDict(task_info['collector_cfg'])
        collector = create_parallel_collector(collector_cfg)
        for item in ['send_metadata', 'send_stepdata', 'get_policy_update_info']:
            setattr(collector, item, getattr(self, item))
        return collector


def create_comm_collector(cfg: EasyDict) -> BaseCommCollector:
    """
    Overview:
        Given the key(comm_collector_name), create a new comm collector instance if in comm_map's values,
        or raise an KeyError. In other words, a derived comm collector must first register,
        then can call ``create_comm_collector`` to get the instance.
    Arguments:
        - cfg (:obj:`EasyDict`): Collector config. Necessary keys: [import_names, comm_collector_type].
    Returns:
        - collector (:obj:`BaseCommCollector`): The created new comm collector, should be an instance of one of \
        comm_map's values.
    """
    import_module(cfg.get('import_names', []))
    return COMM_COLLECTOR_REGISTRY.build(cfg.type, cfg=cfg)
