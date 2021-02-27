import os
import sys
import logging
from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
from typing import Any, Union, Tuple
from functools import partial

from nervex.policy import Policy
from nervex.utils.autolog import LoggedValue, LoggedModel, NaturalTime, TickTime, TimeMode
from nervex.utils import build_logger, EasyTimer, get_task_uid, import_module, pretty_print
from nervex.torch_utils import build_log_buffer


class TickMonitor(LoggedModel):
    """
    Overview:
        TickMonitor is to monitor related info of one interation with env.
        Info include: policy_time, env_time, norm_env_time, timestep_size...
        These info variables would first be recorded in ``log_buffer``, then in ``self._iter_after_hook`` will vars in
        in this monitor be updated by``log_buffer``, then printed to ``TextLogger`` and ``TensorBoradLogger``.
    Interface:
        __init__, fixed_time, current_time, freeze, unfreeze, register_attribute_value, __getattr__
    Property:
        time, expire
    """
    policy_time = LoggedValue(float)
    env_time = LoggedValue(float)
    timestep_size = LoggedValue(float)
    norm_env_time = LoggedValue(float)

    def __init__(self, time_: 'BaseTime', expire: Union[int, float]):  # noqa
        LoggedModel.__init__(self, time_, expire)
        self.__register()

    def __register(self):

        def __avg_func(prop_name: str) -> float:
            records = self.range_values[prop_name]()
            _list = [_value for (_begin_time, _end_time), _value in records]
            return sum(_list) / len(_list)

        self.register_attribute_value('avg', 'policy_time', partial(__avg_func, prop_name='policy_time'))
        self.register_attribute_value('avg', 'env_time', partial(__avg_func, prop_name='env_time'))
        self.register_attribute_value('avg', 'timestep_size', partial(__avg_func, prop_name='timestep_size'))
        self.register_attribute_value('avg', 'norm_env_time', partial(__avg_func, prop_name='norm_env_time'))


class BaseActor(ABC):
    """
    Overview:
        Abstract baseclass for actor.
    Interfaces:
        __init__, start, close
    Property:
        policy
    """

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
        """
        self._cfg = cfg
        self._eval_flag = cfg.eval_flag
        self._prefix = 'EVALUATOR' if self._eval_flag else 'ACTOR'
        self._actor_uid = get_task_uid()
        self._logger, self._monitor, self._log_buffer = self._setup_logger()
        self._end_flag = False
        self._setup_timer()
        self._iter_count = 0
        self.info("\nCFG INFO:\n{}".format(pretty_print(cfg, direct_print=False)))

    def info(self, s: str) -> None:
        self._logger.info("[{}({})]: {}".format(self._prefix, self._actor_uid, s))

    def debug(self, s: str) -> None:
        self._logger.debug("[{}({})]: {}".format(self._prefix, self._actor_uid, s))

    def error(self, s: str) -> None:
        self._logger.error("[{}({})]: {}".format(self._prefix, self._actor_uid, s))

    def _setup_timer(self) -> None:
        """
        Overview:
            Setup TimeWrapper for base_actor. TimeWrapper is a decent timer wrapper that can be used easily.
            You can refer to ``nervex/utils/time_helper.py``.

        Note:
            - _policy_inference (:obj:`Callable`): The wrapper to acquire a policy's time.
            - _env_step (:obj:`Callable`): The wrapper to acquire a environment's time.
        """
        self._timer = EasyTimer()

        def policy_wrapper(fn):

            def wrapper(*args, **kwargs):
                with self._timer:
                    ret = fn(*args, **kwargs)
                self._log_buffer['policy_time'] = self._timer.value
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

        self._policy_inference = policy_wrapper(self._policy_inference)
        self._env_step = env_wrapper(self._env_step)

    def _setup_logger(self) -> Tuple:
        """
        Overview:
            Setup logger for base_actor. Logger includes logger, monitor and log buffer dict.
        Returns:
            - logger (:obj:`TextLogger`): logger that displays terminal output
            - monitor (:obj:`TickMonitor`): monitor that is related info of one interation with env
            - log_buffer (:obj:`LogDict`): log buffer dict
        """
        path = './log/{}'.format(self._prefix.lower())
        name = '{}'.format(self._actor_uid)
        logger, _ = build_logger(path, name, need_tb=False)
        monitor = TickMonitor(TickTime(), expire=self._cfg.print_freq * 2)
        log_buffer = build_log_buffer()
        return logger, monitor, log_buffer

    def start(self) -> None:
        self._end_flag = False
        self._update_policy()
        self._start_thread()
        while not self._end_flag:
            obs = self._env_manager.next_obs
            action = self._policy_inference(obs)
            timestep = self._env_step(action)
            self._process_timestep(timestep)
            self._iter_after_hook()
            if self._env_manager.done:
                self._finish_task()
                break

    def close(self) -> None:
        if self._end_flag:
            return
        self._end_flag = True

    def _iter_after_hook(self):
        # log_buffer -> tick_monitor -> monitor.step
        for k, v in self._log_buffer.items():
            setattr(self._monitor, k, v)
        self._monitor.time.step()
        # Print info
        if self._iter_count % self._cfg.print_freq == 0:
            self.debug('{}TimeStep{}{}'.format('=' * 35, self._iter_count, '=' * 35))
            # tick_monitor -> var_dict
            var_dict = {}
            for k in self._log_buffer:
                for attr in self._monitor.get_property_attribute(k):
                    k_attr = k + '_' + attr
                    var_dict[k_attr] = getattr(self._monitor, attr)[k]()
            self._logger.print_vars(var_dict, level=logging.DEBUG)
        self._log_buffer.clear()
        self._iter_count += 1

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _policy_inference(self, obs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _env_step(self, action: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _process_timestep(self, timestep: namedtuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def _finish_task(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _update_policy(self) -> None:
        raise NotImplementedError

    def _start_thread(self) -> None:
        pass

    @property
    def policy(self) -> Policy:
        return self._policy

    @policy.setter
    def policy(self, _policy: Policy) -> None:
        self._policy = _policy
        if not self._eval_flag:
            self._policy.set_setting('collect', self._cfg.collect_setting)


actor_mapping = {}


def register_actor(name: str, actor: BaseActor) -> None:
    assert isinstance(name, str)
    assert issubclass(actor, BaseActor)
    actor_mapping[name] = actor


def create_actor(cfg: dict) -> BaseActor:
    import_module(cfg.import_names)
    actor_type = cfg.actor_type
    if actor_type not in actor_mapping.keys():
        raise KeyError("not support actor type: {}".format(actor_type))
    else:
        return actor_mapping[actor_type](cfg)
