from abc import ABC, abstractmethod, abstractproperty
import os
import sys
from collections import namedtuple
from typing import Union, Any
from nervex.utils import build_logger_naive, EasyTimer, get_task_uid, VariableRecord, import_module
from .comm.actor_comm_helper import ActorCommHelper


class BaseActor(ABC):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init()
        if self._cfg.actor.communication.type == 'single_machine':
            self._logger.info('[WARNING]: use default single machine communication strategy')
            # TODO single machine actor
            raise NotImplementedError
        else:
            comm_cfg = self._cfg.actor.communication
            ActorCommHelper.enable_comm_helper(self, comm_cfg)

    def _init(self) -> None:
        self._actor_uid = get_task_uid()
        self._setup_logger()
        self._end_flag = False
        self._setup_timer()

    def _setup_timer(self):
        self._timer = EasyTimer()

        def agent_wrapper(fn):
            def wrapper(*args, **kwargs):
                with self._timer:
                    ret = fn(*args, **kwargs)
                self._variable_record.update_var({'agent_time': self._timer.value})
                return ret

            return wrapper

        def env_wrapper(fn):
            def wrapper(*args, **kwargs):
                with self._timer:
                    ret = fn(*args, **kwargs)
                size = sys.getsizeof(ret) / (1024 * 1024)  # MB
                self._variable_record.update_var(
                    {
                        'env_time': self._timer.value,
                        'timestep_size': size,
                        'norm_env_time': self._timer.value / size
                    }
                )
                return ret

            return wrapper

        self._agent_inference = agent_wrapper(self._agent_inference)
        self._env_step = env_wrapper(self._env_step)

    def _check(self) -> None:
        assert hasattr(self, 'init_service')
        assert hasattr(self, 'close_service')
        assert hasattr(self, 'get_job')
        assert hasattr(self, 'get_agent_update_info')
        assert hasattr(self, 'send_traj_metadata')
        assert hasattr(self, 'send_traj_stepdata')
        assert hasattr(self, 'send_finish_job')

    def _init_with_job(self, job: dict) -> None:
        # update iter_count and varibale_record for each job
        self._iter_count = 0
        self._variable_record = VariableRecord(self._cfg.actor.print_freq)
        self._variable_record.register_var('agent_time')
        self._variable_record.register_var('env_time')
        self._variable_record.register_var('timestep_size')
        self._variable_record.register_var('norm_env_time')
        # other parts need to be implemented by subclass

    @abstractmethod
    def episode_reset(self) -> None:
        raise NotImplementedError

    def _setup_logger(self) -> None:
        path = os.path.join(self._cfg.common.save_path, 'actor-log')
        name = 'actor.{}.log'.format(self._actor_uid)
        self._logger, self._variable_record = build_logger_naive(path, name)

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    def run(self) -> None:
        self.init_service()
        while not self._end_flag:
            job = self.get_job()
            self._init_with_job(job)
            for episode_idx in range(job['episode_num']):
                obs = self.episode_reset()
                while True:
                    action = self._agent_inference(obs)
                    timestep = self._env_step(action)
                    self._accumulate_timestep(obs, action, timestep)
                    obs = self._get_next_obs(timestep)
                    self._iter_after_hook()
                    if self.all_done:
                        break
                self._finish_episode(timestep)
            self._finish_job()

    def close(self) -> None:
        self.close_service()
        self._end_flag = True

    def _iter_after_hook(self):
        # print info
        if self._iter_count % self._cfg.actor.print_freq == 0:
            self._logger.info(
                'actor({}):\n{}TimeStep{}{} {}'.format(
                    self._actor_uid, '=' * 35, self._iter_count, '=' * 35, self._variable_record.get_vars_text()
                )
            )
        self._iter_count += 1

    def _get_next_obs(self, timestep: namedtuple) -> Any:
        # some special actor will do additional operation on next obs, thus we design this interface
        return timestep.obs

    @abstractmethod
    def _agent_inference(self, obs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _env_step(self, action: Any) -> namedtuple:
        raise NotImplementedError

    @abstractmethod
    def _accumulate_timestep(self, obs: Any, action: Any, timestep: namedtuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def _finish_episode(self, timestep: namedtuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def _finish_job(self) -> None:
        raise NotImplementedError

    def _pack_trajectory(self) -> None:
        raise NotImplementedError

    def _update_agent(self) -> None:
        raise NotImplementedError

    @abstractproperty
    def all_done(self) -> bool:
        raise NotImplementedError


actor_mapping = {}


def register_actor(name: str, actor: BaseActor) -> None:
    assert isinstance(name, str)
    assert issubclass(actor, BaseActor)
    actor_mapping[name] = actor


def create_actor(cfg: dict) -> BaseActor:
    import_module(cfg.actor.import_names)
    if cfg.actor.actor_type not in actor_mapping.keys():
        raise KeyError("not support actor type: {}".format(cfg.actor.actor_type))
    else:
        return actor_mapping[cfg.actor.actor_type](cfg)
