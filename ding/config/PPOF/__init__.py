from easydict import EasyDict
from . import gym_lunarlander_v2
from . import gym_lunarlandercontinuous_v2

supported_env_cfg = {
    gym_lunarlander_v2.cfg.env_id: gym_lunarlander_v2.cfg,
    gym_lunarlandercontinuous_v2.cfg.env_id: gym_lunarlandercontinuous_v2.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)

supported_env = {
    gym_lunarlander_v2.cfg.env_id: gym_lunarlander_v2.env,
    gym_lunarlandercontinuous_v2.cfg.env_id: gym_lunarlandercontinuous_v2.env,
}

supported_env = EasyDict(supported_env)
