from abc import ABC, abstractmethod
import threading
from typing import Any


class BaseCommActor(ABC):
    def __init__(self, cfg):
        self._cfg = cfg
        self._active_flag = False
        # Note: the following variable will be set by the outside caller
        self._logger = None
        self._actor_uid = None

    @abstractmethod
    def register_actor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_job(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_agent_update_info(self, path: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def send_traj_metadata(self, metadata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_traj_stepdata(self, stepdata: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_finish_job(self, finish_info: Any) -> None:
        raise NotImplementedError

    def start_heartbeats_thread(self) -> None:
        # start sending heartbeats thread
        check_send_actor_heartbeats_thread = threading.Thread(target=self._send_actor_heartbeat)
        check_send_actor_heartbeats_thread.daemon = True
        check_send_actor_heartbeats_thread.start()
        self._logger.info("Actor({}) send heartbeat thread start...".format(self._comm_actor_id))

    def init_service(self) -> None:
        self.register_actor()
        self._active_flag = True
        self.start_heartbeats_thread()

    def close(self) -> None:
        self._active_flag = False

    @property
    def actor_uid(self) -> str:
        return self._actor_uid

    @actor_uid.setter
    def actor_uid(self, _actor_uid: str) -> None:
        self._actor_uid = _actor_uid

    @property
    def logger(self) -> Any:
        return self._logger

    @logger.setter
    def logger(self, _logger: Any) -> None:
        self._logger = _logger

    # ************************** thread *********************************
    @abstractmethod
    def _send_actor_heartbeats(self) -> None:
        raise NotImplementedError


class SingleMachineActor(ABC):
    # TODO single matchine actor for some micro envs
   pass
