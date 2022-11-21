import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register("luxai2021")
class LuxEnvironment(BaseEnv):
    pass