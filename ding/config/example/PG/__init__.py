from easydict import EasyDict
from . import gym_pendulum_v1

supported_env_cfg = {
    gym_pendulum_v1.cfg.env.env_id: gym_pendulum_v1.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)

supported_env = {
    gym_pendulum_v1.cfg.env.env_id: gym_pendulum_v1.env,
}

supported_env = EasyDict(supported_env)
