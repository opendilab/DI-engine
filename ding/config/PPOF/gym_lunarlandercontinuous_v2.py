from easydict import EasyDict

cfg = dict(
    exp_name='LunarLanderContinuous-V2-PPO',
    env_id='LunarLanderContinuous-v2',
    action_space='continuous',
    n_sample=400,
    act_scale=True,
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
from functools import partial
env = partial(ding.envs.gym_env.env, continuous=True)
