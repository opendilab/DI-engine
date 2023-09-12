from . import BaseEnvManagerV2, SubprocessEnvManagerV2
from ..env import DingEnvWrapper
from typing import Optional
from functools import partial


def setup_ding_env_manager(
        env: DingEnvWrapper,
        env_num: int,
        context: Optional[str] = None,
        debug: bool = False,
        caller: str = 'collector'
) -> BaseEnvManagerV2:
    assert caller in ['evaluator', 'collector']
    if debug:
        env_cls = BaseEnvManagerV2
        manager_cfg = env_cls.default_config()
    else:
        env_cls = SubprocessEnvManagerV2
        manager_cfg = env_cls.default_config()
        if context is not None:
            manager_cfg.context = context
    return env_cls([partial(env.clone, caller) for _ in range(env_num)], manager_cfg)
