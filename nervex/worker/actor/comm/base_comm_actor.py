import threading
from abc import ABC, abstractmethod
from typing import Any
from easydict import EasyDict

from nervex.policy import create_policy
from nervex.utils import get_task_uid, import_module
from ..base_parallel_actor import create_actor, BaseActor


class BaseCommActor(ABC):
    """
    Overview:
        Abstract baseclass for common actor.
    Interfaces:
        __init__, get_policy_update_info, send_metadata, send_stepdata, send_finish_info,
        start, close, _create_actor
    Property:
        actor_uid
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
        self._actor_uid = get_task_uid()

    @abstractmethod
    def get_policy_update_info(self, path: str) -> Any:
        """
        Overview:
            Get policy information in corresponding path.
            Will be registered in base actor.
        Arguments:
            - path (:obj:`str`): path to policy update information.
        """
        raise NotImplementedError

    @abstractmethod
    def send_metadata(self, metadata: Any) -> None:
        """
        Overview:
            Store meta data in queue, which will be retrieved by callback function "deal_with_actor_data"
            in actor slave, then will be sent to coordinator.
            Will be registered in base actor.
        Arguments:
            - metadata (:obj:`Any`): meta data.
        """
        raise NotImplementedError

    @abstractmethod
    def send_stepdata(self, stepdata: Any) -> None:
        """
        Overview:
            Save step data in corresponding path.
            Will be registered in base actor.
        Arguments:
            - stepdata (:obj:`Any`): step data.
        """
        raise NotImplementedError

    @abstractmethod
    def send_finish_info(self, path: str, finish_info: Any) -> None:
        """
        Overview:
            Store finish info dict in queue, which will be retrieved by callback function "deal_with_actor_data"
            in actor slave, then will be sent to coordinator.
            Will be registered in base actor.
        Arguments:
            - finish_info (:obj:`dict`): Finish info in `dict` type. Keys are like 'finished_task', 'finish_time' \
            'duration', etc.
        """
        raise NotImplementedError

    def start(self) -> None:
        """
        Overview:
            Start comm actor.
        """
        self._end_flag = False

    def close(self) -> None:
        """
        Overview:
            Close comm actor.
        """
        self._end_flag = True

    @property
    def actor_uid(self) -> str:
        return self._actor_uid

    def _create_actor(self, task_info: dict) -> BaseActor:
        """
        Overview:
            Receive ``task_info`` passed from coordinator and create a actor.
        Arguments:
            - task_info (:obj:`dict`): Task info dict from coordinator. Should be like \
        Returns:
            - actor (:obj:`BaseActor`): Created base actor.
        Note:
            Four methods('send_metadata', 'send_stepdata', 'get_policy_update_info', 'send_finish_info'),
            and policy are set.
            The reason why they are set here rather than base actor is that, they highly depend on the specific task.
            Only after task info is passed from coordinator to comm actor through learner slave, can they be
            clarified and initialized.
        """
        actor_cfg = EasyDict(task_info['actor_cfg'])
        actor = create_actor(actor_cfg)
        for item in ['send_metadata', 'send_stepdata', 'get_policy_update_info', 'send_finish_info']:
            setattr(actor, item, getattr(self, item))
        eval_flag = actor_cfg.eval_flag
        if eval_flag:
            actor.policy = create_policy(task_info['policy'], enable_field=['eval']).eval_mode
        else:
            actor.policy = create_policy(task_info['policy'], enable_field=['collect']).collect_mode
        return actor


comm_map = {}


def register_comm_actor(name: str, actor_type: type) -> None:
    """
    Overview:
        Register a new CommActor class with its name to dict ``comm_map``
    Arguments:
        - name (:obj:`str`): Name of the new CommActor
        - actor_type (:obj:`type`): The new CommActor class, should be subclass of ``BaseCommActor``.
    """
    assert isinstance(name, str)
    assert issubclass(actor_type, BaseCommActor)
    comm_map[name] = actor_type


def create_comm_actor(cfg: dict) -> BaseCommActor:
    """
    Overview:
        Given the key(comm_actor_name), create a new comm actor instance if in comm_map's values,
        or raise an KeyError. In other words, a derived comm actor must first register,
        then can call ``create_comm_actor`` to get the instance.
    Arguments:
        - cfg (:obj:`dict`): Actor config. Necessary keys: [import_names, comm_actor_type].
    Returns:
        - actor (:obj:`BaseCommActor`): The created new comm actor, should be an instance of one of \
        comm_map's values.
    """
    cfg = EasyDict(cfg)
    import_module(cfg.import_names)
    comm_actor_type = cfg.comm_actor_type
    if comm_actor_type not in comm_map.keys():
        raise KeyError("not support comm actor type: {}".format(comm_actor_type))
    else:
        return comm_map[comm_actor_type](cfg)
