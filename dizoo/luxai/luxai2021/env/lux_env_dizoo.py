import numpy as np
from easydict import EasyDict
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list

from dizoo.luxai.luxai2021.env.lux_env import LuxEnvironment
from dizoo.luxai.luxai2021.env.agent import Agent, AgentWithModel
from dizoo.luxai.luxai2021.env.agent_policy import AgentPolicy

@ENV_REGISTRY.register("luxai2021")
class Lux2021Env(BaseEnv):
    """
    Created & adapted by i-am-tc for use in DI-engine
    Credts to  glmcdona's LuxPythonEnvGym
    """

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg # TODO: Is my config compatible with DI-engine & original lux env?
        self._init_flag = False
        self._replay_path = None

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> np.ndarray:
        # Instantiate environment & update flag
        if not self._init_flag:
            self._env = self._make_env
            self._init_flag = True
        # Set dynamic seed 
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed) # TODO: does env have a seed method?
        obs = self._env.reset()
        obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        pass

    def _make_env(self):
        # Read config and prepare agents
        opponent = Agent()
        learner = AgentPolicy(mode="train")
        env = LuxEnvironment(configs=self._cfg, learning_agent=learner, opponent_agent=opponent)
        return env

if __name__ == "__main__":
    from dizoo.luxai.luxai2021.config.lux_ppo_config import main_config, create_config
    from ding.config import compile_config
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    test = Lux2021Env(cfg)