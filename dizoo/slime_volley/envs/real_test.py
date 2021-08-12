import numpy as np
from easydict import EasyDict

from dizoo.slime_volley.envs.slime_volley_env import SlimeVolleyEnv


num_agent = 2
env = SlimeVolleyEnv(EasyDict({'env_id': 'SlimeVolley-v0'}))
obs = env.reset()
done = False
print(env._env.observation_space)
