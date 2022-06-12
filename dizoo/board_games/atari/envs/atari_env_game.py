"""
Adapt Atari to BaseGameEnv interface
"""

from pettingzoo.utils.agent_selector import agent_selector
import sys
from dizoo.board_games.base_game_env import BaseGameEnv
from typing import Any, List, Union, Sequence
import copy
import numpy as np
import gym
from ding.envs import BaseEnv, BaseEnvTimestep, update_shape
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list
from dizoo.atari.envs.atari_wrappers import wrap_deepmind, wrap_deepmind_mr
from ding.rl_utils.efficientzero.utils import make_atari, WarpFrame, EpisodicLifeEnv
from ding.rl_utils.efficientzero.atari_env_wrapper import AtariWrapper


@ENV_REGISTRY.register('atari-game')
class AtariGameEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False
        self.current_player_index = 0
        self.agents = [f"player_{i + 1}" for i in range(2)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}
        self.agent_str = None

    @property
    def current_player(self):
        return self.current_player_index

    def to_play(self):
        return self.agents.index(self._agent_selector.next())

    def _make_env(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False,
                  final_test=False):
        """DI-engine wrapper"""
        # return wrap_deepmind(
        #     self.cfg.env_id,
        #     frame_stack=self.cfg.frame_stack,
        #     episode_life=self.cfg.is_train,
        #     clip_rewards=self.cfg.is_train
        # )
        """EfficientZero wrapper"""
        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip
            else:
                max_moves = self.cfg.test_max_moves
            env = make_atari(self.env_name, skip=self.frame_skip, max_episode_steps=max_moves)
        else:
            env = make_atari(self.cfg.env_name, skip=self.cfg.frame_skip, max_episode_steps=self.cfg.max_moves)

        if self.cfg.episode_life and not test:
            env = EpisodicLifeEnv(env)
        env = WarpFrame(env, width=self.cfg.obs_shape[1], height=self.cfg.obs_shape[2], grayscale=self.cfg.gray_scale)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers import Monitor
            env = Monitor(env, directory=save_path, force=True, video_callable=video_callable, uid=uid)
        return AtariWrapper(env, discount=self.cfg.discount, cvt_string=self.cfg.cvt_string)

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
        self.agents = self.possible_agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_str = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        agent = self.agent_str
        self.current_player_index = self.agents.index(agent)
        obs = self.observe()
        return BaseEnvTimestep(obs, None, None, None)

    def observe(self):
        observation = self.obs
        action_mask = np.ones(self._action_space.n, 'int8')
        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        current_agent = self.agent_str
        self.current_player_index = self.agents.index(current_agent)
        self.agent_str = self._agent_selector.next()  # next player to play
        action = int(action)
        obs, reward, done, info = self._env.step(action)
        # self._env.render()
        self._final_eval_reward += reward
        self.obs = to_ndarray(obs)
        self.reward = to_ndarray([reward])
        observation = self.observe()
        info = {'next player to play': self.agent_str}

        if done:
            info['final_eval_reward'] = self._final_eval_reward

        return BaseEnvTimestep(observation, self.reward, done, info)

    def legal_actions(self):
        return np.arange(self._action_space.n)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def random_action(self):
        action_list = self.legal_actions()
        return np.random.choice(action_list)

    def render(self, mode='human'):
        self._env.render()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print(f"Current available actions for the player {self.to_play()} are:{self.legal_actions()}")
                choice = int(
                    input(
                        f"Enter the index of next move for the player {self.to_play()}: "
                    )
                )
                if choice in self.legal_actions():
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return str(action_number)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return "DI-engine Atari Env({})".format(self.cfg.env_name)

    @staticmethod
    def create_collector_envcfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_envcfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]


if __name__ == '__main__':
    from easydict import EasyDict

    cfg = EasyDict(env_name='PongNoFrameskip-v4',
                   frame_skip=4,
                   frame_stack=4,
                   max_moves=1e6,
                   episode_life=True,
                   obs_shape=(12, 96, 96),
                   gray_scale=False,
                   discount=0.997,
                   cvt_string=True,
                   is_train=True)
    env = AtariGameEnv(cfg)
    obs, reward, done, info = env.reset()
    env.render()
    print('=' * 20)
    print('In atari, player 1 = player 2')
    print('=' * 20)
    while True:
        action = env.random_action()
        # action = env.human_to_action()
        print('player 1: ' + env.action_to_string(action))
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print(info)
            print('=' * 20)
            print('In atari, player 1 = player 2')
            print('=' * 20)
            break

        action = env.random_action()
        # action = env.human_to_action()
        print('player 2: ' + env.action_to_string(action))
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print(info)
            print('=' * 20)
            print('In atari, player 1 = player 2')
            print('=' * 20)
            break
