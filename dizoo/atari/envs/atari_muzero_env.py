"""
Adapt Atari to BaseGameEnv interface
"""

import sys
from typing import Any, List, Union, Sequence
import copy
import numpy as np
import gym
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray
from dizoo.atari.envs.atari_wrappers import wrap_muzero, wrap_muzero_dqn_expert_data


@ENV_REGISTRY.register('atari-muzero')
class AtariMuZeroEnv(BaseEnv):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False

    def _make_env(self):
        if self.cfg.dqn_expert_data:
            return wrap_muzero_dqn_expert_data(self.cfg)
        else:
            return wrap_muzero(self.cfg)

    def reset(self):
        if not self._init_flag:
            self._env = self._make_env()
            self._observation_space = self._env.env.observation_space
            self._action_space = self._env.env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.env.reward_range[0], high=self._env.env.reward_range[1], shape=(1,), dtype=np.float32
            )

            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.env.seed(self._seed)

        obs = self._env.reset()
        self.obs = to_ndarray(obs)
        self._final_eval_reward = 0.
        self.has_reset = True
        obs = self.observe()
        return obs

    def observe(self):
        """
        Overview:
            add action_mask to obs to adapt with MCTS alg..
        """
        observation = self.obs
        action_mask = np.ones(self._action_space.n, 'int8')
        return {'observation': observation, 'action_mask': action_mask, 'to_play': None}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # self._env.render()
        self.obs = to_ndarray(obs)
        self.reward = np.array(reward).astype(np.float32)
        self._final_eval_reward += self.reward
        observation = self.observe()
        if done:
            info['final_eval_reward'] = self._final_eval_reward

        return BaseEnvTimestep(observation, self.reward, done, info)

    @property
    def legal_actions(self):
        return np.arange(self._action_space.n)

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def render(self, mode='human'):
        self._env.render()

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print(f"Current available actions for the player are:{self.legal_actions}")
                choice = int(
                    input(
                        f"Enter the index of next action: "
                    )
                )
                if choice in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Atari MuZero Env({})".format(self.cfg.env_name)

    @staticmethod
    def create_collector_envcfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_envcfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]
