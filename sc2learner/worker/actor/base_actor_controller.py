from abc import ABC, abstractmethod
import os
from collections import namedtuple
from typing import Union, Any
from sc2learner.data import BaseContainer
from sc2learner.utils import build_logger_naive, EasyTimer, get_actor_uid


class BaseActor(ABC):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._actor_uid = get_actor_uid()
        self._timer = EasyTimer()
        self._setup_logger()

        assert hasattr(self, 'get_agent_update_info')
        assert hasattr(self, 'send_traj_metadata')
        assert hasattr(self, 'send_traj_stepdata')
        assert hasattr(self, 'send_finish_job')

    @abstractmethod
    def _init_with_job(self, job: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def episode_reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_logger(self) -> None:
        path = os.path.join(self.cfg.common.save_path, 'actor-log')
        name = 'actor.{}.log'.format(self.actor_uid)
        self.logger, self.variable_record = build_logger_naive(path, name)
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    def run(self, job: dict) -> None:
        self._init_with_job(job)
        for episode_idx in range(job['episode_num']):
            obs = self.episode_reset()
            while True:
                action = self._agent_inference(obs)
                timestep = self._env_step(action)
                self._accumulate_data(obs, action, timestep)
                obs = timestep.obs
                if timestep.all_done:
                    break
            self._finish_episode(timestep)
        self._finish_job()

    @abstractmethod
    def _agent_inference(self, obs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _env_step(self, action: Any) -> namedtuple:
        raise NotImplementedError

    @abstractmethod
    def _accumulate_timestep(self, obs: Any, action: Any, timestep: namedtuple) -> None:
        raise NotImplementedError

    def _finish_episode(self, timestep: namedtuple) -> None:
        raise NotImplementedError

    def _finish_job(self) -> None:
        raise NotImplementedError

    def _pack_trajectory(self) -> None:
        raise NotImplementedError

    def _update_agent(self) -> None:
        raise NotImplementedError
