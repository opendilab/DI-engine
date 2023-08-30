from easydict import EasyDict
from . import gym_bipedalwalker_v3
from . import gym_halfcheetah_v3
from . import gym_hopper_v3
from . import gym_lunarlander_v2
from . import gym_walker2d_v3

supported_env_cfg = {
    gym_bipedalwalker_v3.cfg.env.env_id: gym_bipedalwalker_v3.cfg,
    gym_halfcheetah_v3.cfg.env.env_id: gym_halfcheetah_v3.cfg,
    gym_hopper_v3.cfg.env.env_id: gym_hopper_v3.cfg,
    gym_lunarlander_v2.cfg.env.env_id: gym_lunarlander_v2.cfg,
    gym_walker2d_v3.cfg.env.env_id: gym_walker2d_v3.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)

supported_env = {
    gym_bipedalwalker_v3.cfg.env.env_id: gym_bipedalwalker_v3.env,
    gym_halfcheetah_v3.cfg.env.env_id: gym_halfcheetah_v3.env,
    gym_hopper_v3.cfg.env.env_id: gym_hopper_v3.env,
    gym_lunarlander_v2.cfg.env.env_id: gym_lunarlander_v2.env,
    gym_walker2d_v3.cfg.env.env_id: gym_walker2d_v3.env,
}

supported_env = EasyDict(supported_env)
