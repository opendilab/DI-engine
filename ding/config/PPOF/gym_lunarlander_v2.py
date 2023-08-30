from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='LunarLander-v2-PPO',
    env_id='LunarLander-v2',
    n_sample=400,
    value_norm='popart',
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
