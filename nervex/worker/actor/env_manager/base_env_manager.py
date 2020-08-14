from abc import ABC, abstractmethod
from typing import Union, Any, List, Callable


class BaseEnvManager(ABC):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._init()
        assert hasattr(self, '_envs')

    @abstractmethod
    def _init(self) -> None:
        raise NotImplementedError

    def __getattr__(self, key: str) -> Any:
        """
        Note: if a python object doesn't have the attribute named key, it will call this method
        """
        return [getattr(env, key) if hasattr(env, key) else None for env in self._envs]

    @property
    def env_num(self) -> int:
        return self._env_num

    def reset(self,
              reset_param: Union[None, List[dict]] = None,
              env_id: Union[None, List[int]] = None) -> Union[list, dict]:
        for i in (env_id if env_id is not None else range(self.env_num)):
            self._env_done[i] = False
        return self._execute_by_envid('reset', param=reset_param, env_id=env_id)

    def step(self, action: List[Any], env_id: Union[None, List[int]] = None) -> Union[list, dict]:
        param = [{'action': act} for act in action]
        ret = self._execute_by_envid('step', param=param, env_id=env_id)
        if isinstance(ret, list):
            self._env_done = [t.done for t in ret]
        elif isinstance(ret, dict):
            for k, v in ret.items():
                self._env_done[k] = v.done
        return ret

    def seed(self, seed: List[int], env_id: Union[None, List[int]] = None) -> None:
        param = [{'seed': s} for s in seed]
        return self._execute_by_envid('seed', param=param, env_id=env_id)

    def _execute_by_envid(
            self,
            fn_name: str,
            param: Union[None, List[dict]] = None,
            env_id: Union[None, List[int]] = None
    ) -> Union[list, dict]:
        real_env_id = list(range(self.env_num)) if env_id is None else env_id
        if param is None:
            ret = {real_env_id[i]: getattr(self._envs[real_env_id[i]], fn_name)() for i in range(len(real_env_id))}
        else:
            ret = {
                real_env_id[i]: getattr(self._envs[real_env_id[i]], fn_name)(**param[i])
                for i in range(len(real_env_id))
            }
        ret = list(ret.values()) if env_id is None else ret
        return ret

    def close(self) -> None:
        for env in self._envs:
            env.close()

    @property
    def all_done(self) -> bool:
        return all(self._env_done)

    @property
    def env_done(self) -> List[bool]:
        return self._env_done
