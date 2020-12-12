import os
import sys
from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from typing import Any, Union
from functools import partial

from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from nervex.utils import build_logger, EasyTimer, get_task_uid, import_module
from nervex.torch_utils import build_log_buffer
from .comm.actor_comm_helper import ActorCommHelper


class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info of one training iteration.
        Info include: cur_lr, time(data, train, forward, backward), loss(total,...)
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    agent_time = LoggedValue(float)
    env_time = LoggedValue(float)
    timestep_size = LoggedValue(int)
    norm_env_time = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list)

        self.register_attribute_value('avg', 'agent_time', partial(__avg_func, prop_name='agent_time'))
        self.register_attribute_value('avg', 'env_time', partial(__avg_func, prop_name='env_time'))
        self.register_attribute_value('avg', 'timestep_size', partial(__avg_func, prop_name='timestep_size'))
        self.register_attribute_value('avg', 'norm_env_time', partial(__avg_func, prop_name='norm_env_time'))


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
                self._log_buffer['agent_time'] = self._timer.value
                return ret

            return wrapper

        def env_wrapper(fn):

            def wrapper(*args, **kwargs):
                with self._timer:
                    ret = fn(*args, **kwargs)
                size = sys.getsizeof(ret) / (1024 * 1024)  # MB
                self._log_buffer['env_time'] = self._timer.value
                self._log_buffer['timestep_size'] = size
                self._log_buffer['norm_env_time'] = self._timer.value / size
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
        self._logger.info("ACTOR({}): JOB INFO:\n{}".format(self._actor_uid, job))

        # other parts need to be implemented by subclass

    def _setup_logger(self) -> None:
        path = os.path.join(self._cfg.common.save_path, 'actor')
        name = 'actor.{}.log'.format(self._actor_uid)
        self._logger, _ = build_logger(path, name, False)
        self._monitor = TickMonitor(TickTime(), expire=self._cfg.actor.print_freq * 2)
        self._log_buffer = build_log_buffer()

    def run(self) -> None:
        self.init_service()
        while not self._end_flag:
            job = self.get_job()
            self._init_with_job(job)
            while True:
                obs = self._env_manager.next_obs
                action = self._agent_inference(obs)
                timestep = self._env_step(action)
                self._process_timestep(timestep)
                self._iter_after_hook()
                if self._env_manager.done:
                    break
            self._finish_job()

    def close(self) -> None:
        self.close_service()
        self._end_flag = True

    def _iter_after_hook(self):
        # log_buffer -> tick_monitor -> monitor.step
        for k, v in self._log_buffer.items():
            setattr(self._monitor, k, v)
        self._monitor.time.step()

        # print info
        if self._iter_count % self._cfg.actor.print_freq == 0:
            self._logger.info(
                'ACTOR({}):\n{}TimeStep{}{}'.format(self._actor_uid, '=' * 35, self._iter_count, '=' * 35)
            )
            # tick_monitor -> var_dict
            var_dict = {}
            for k in self._log_buffer:
                for attr in self._monitor.get_property_attribute(k):
                    k_attr = k + '_' + attr
                    var_dict[k_attr] = getattr(self._monitor, attr)[k]()
            self._logger.print_vars(var_dict)
        self._log_buffer.clear()
        self._iter_count += 1

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _agent_inference(self, obs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _env_step(self, action: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _process_timestep(self, timestep: namedtuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def _finish_job(self) -> None:
        raise NotImplementedError

    def _pack_trajectory(self) -> None:
        raise NotImplementedError

    def _update_agent(self) -> None:
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
