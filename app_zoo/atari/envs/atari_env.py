from typing import Any
import torch
from nervex.envs import BaseEnv
from nervex.torch_utils import to_tensor
from .atari_wrappers import wrap_deepmind


class AtariEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env = wrap_deepmind(
            cfg.env_id, frame_stack=cfg.frame_stack, episode_life=cfg.is_train, clip_rewards=cfg.is_train
        )

    def reset(self) -> torch.FloatTensor:
        obs = self._env.reset()
        obs = to_tensor(obs, torch.float)
        return obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def step(self, action: Any) -> BaseEnv.timestep:
        action = action.numpy()
        obs, rew, done, info = self._env.step(action)
        obs = to_tensor(obs, torch.float)
        rew = to_tensor(rew, torch.float)
        return BaseEnv.timestep(obs, rew, done, info)

    def info(self) -> BaseEnv.info_template:
        rew_range = self._env.reward_range
        return BaseEnv.info_template(
            agent_num=1,
            obs_space={'shape': self._env.observation_space.shape},
            act_space={'shape': self._env.action_space.n},
            rew_space={
                'shape': 1,
                'value': {
                    'min': rew_range[0],
                    'max': rew_range[1]
                }
            },
        )

    def __repr__(self) -> str:
        return "nerveX Atari Env({})".format(self._cfg.env_id)
