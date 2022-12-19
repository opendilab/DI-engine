import numpy as np
from dizoo.beergame.envs.beergame_core import BeerGame
from typing import Union, List, Optional

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray
import copy


@ENV_REGISTRY.register('beergame')
class BeerGameEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = BeerGame(self._cfg.role, self._cfg.agent_type, self._cfg.demandDistribution)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = self._env.reward_space
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._eval_episode_return = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        if isinstance(action, np.ndarray) and action.shape == (1, ):
            action = action.squeeze()  # 0-dim array
        obs, rew, done, info = self._env.step(action)
        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)  # wrapped to be transfered to a array with shape (1,)
        return BaseEnvTimestep(obs, rew, done, info)

    def reward_shaping(self, transitions: List[dict]) -> List[dict]:
        new_transitions = copy.deepcopy(transitions)
        for trans in new_transitions:
            trans['reward'] = self._env.reward_shaping(trans['reward'])
        return new_transitions

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        if isinstance(random_action, int):
            random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def enable_save_figure(self, figure_path: Optional[str] = None) -> None:
        self._env.enable_save_figure(figure_path)

    @property
    def observation_space(self) -> int:
        return self._observation_space

    @property
    def action_space(self) -> int:
        return self._action_space

    @property
    def reward_space(self) -> int:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Beergame Env"
