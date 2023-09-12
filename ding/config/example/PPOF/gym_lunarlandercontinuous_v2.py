from easydict import EasyDict
from functools import partial
import ding.envs.gym_env

cfg = dict(
    exp_name='LunarLanderContinuous-V2-PPO',
    env_id='LunarLanderContinuous-v2',
    action_space='continuous',
    n_sample=400,
    act_scale=True,
)

cfg = EasyDict(cfg)

env = partial(ding.envs.gym_env.env, continuous=True)
