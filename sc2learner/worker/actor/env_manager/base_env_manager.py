from abc import ABC, abstractmethod
from typing import Union, Any, List, Callable


class BaseEnvManager(ABC):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._init()
        hasattr(self, '_envs')

    @abstractmethod
    def _init(self) -> None:
        raise NotImplementedError

    def __getattr__(self, key: str) -> Any:
        return [getattr(env, key) if hasattr(env, key) else None for env in self._envs]

    @property
    def env_num(self) -> int:
        return self._env_num

    def reset(self, reset_param: Union[None, List[dict]], env_id: Union[None, List[int]]) -> Union[list, dict]:
        return self._execute_by_envid('reset', param=reset_param, env_id=env_id)

    def step(self, action: List[Any], env_id: Union[None, List[int]]) -> Union[list, dict]:
        param = [{'action': act} for act in action]
        return self._execute_by_envid('step', param=param, env_id=env_id)

    def seed(self, seed: List[int], env_id: Union[None, List[int]]) -> None:
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
