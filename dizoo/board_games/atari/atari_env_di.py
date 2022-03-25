from pettingzoo.utils.agent_selector import agent_selector
import sys
from dizoo.board_games.base_game_env import BaseGameEnv
from typing import Any, List, Union, Sequence
import copy
import numpy as np
import gym
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list
from dizoo.atari.envs.atari_wrappers import wrap_deepmind, wrap_deepmind_mr

"""
Adapte Atari to BaseGameEnv interface
"""


@ENV_REGISTRY.register('AtariDI')
class AtariDIEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._init_flag = False
        self.current_player_index = 0

        self.agents = [f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}
        self.agent_selection = None

    @property
    def current_player(self):
        return self.current_player_index

    def to_play(self):
        return self.agents.index(self._agent_selector.next())

    def _make_env(self):
        return wrap_deepmind(
            self.cfg.env_id,
            frame_stack=self.cfg.frame_stack,
            episode_life=self.cfg.is_train,
            clip_rewards=self.cfg.is_train
        )

    def reset(self):
        if not self._init_flag:
            self._env = self._make_env()
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )

            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        obs = self._env.reset()
        self.obs = to_ndarray(obs)
        self._final_eval_reward = 0.

        self.has_reset = True
        self.agents = self.possible_agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}


        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        agent = self.agent_selection
        self.current_player_index = self.agents.index(agent)
        # self.current_player_index = 0  # next player is 0
        observation = self.observe(agent)
        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    def observe(self, agent):
        observation = self.obs
        action_mask = np.ones(self._action_space.n, 'int8')
        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        # assert isinstance(action, np.ndarray), type(action)
        current_agent = self.agent_selection  # 'player_0'
        self.current_player_index = self.agents.index(current_agent)  # 0
        self.agent_selection = self._agent_selector.next()

        action = action.item()
        obs, rew, done, info = self._env.step(action)
        # self._env.render()
        self._final_eval_reward += rew
        self.obs = to_ndarray(obs)
        self.rew = to_ndarray([rew])  # wrapped to be transfered to a Tensor with shape (1,)
        agent = current_agent
        observation = self.observe(agent)

        if done:
            info['final_eval_reward'] = self._final_eval_reward
            self.infos[current_agent] = info['final_eval_reward']
            self.dones[current_agent] = True
        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    def legal_actions(self):
        return np.arange(self._action_space.n)

    def legal_moves(self):
        return np.arange(self._action_space.n)

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def random_action(self):
        action_list = self.legal_moves()
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
                print(f"Current available actions for the player {self.to_play()} are:{self.legal_moves()}")
                choice = int(
                    input(
                        f"Enter the index of next move for the player {self.to_play()}: "
                    )
                )
                if choice in self.legal_moves():
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

    def game_end(self):
        """Check whether the game is ended or not"""
        pass

    def do_action(self, action):
        pass

    def __repr__(self) -> str:
        return "DI-engine Atari Env({})".format(self.cfg.env_id)

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

    cfg = EasyDict(env_id='PongNoFrameskip-v4',
                   frame_stack=4,
                   is_train=True)
    env = AtariDIEnv(cfg)
    obs, reward, done, info = env.reset()
    env.render()
    while True:
        # action = env.human_to_action()
        action = env.random_action()

        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print(info)
            break

        action = env.random_action()
        # print('computer player ' + env.action_to_string(action))
        print('computer player (player1) take action: ' + f'{action}')

        obs, reward, done, info = env.step(action)
        print(f'After the computer player (player1) took action: {action}, the current board state is:')
        env.render()
        if done:
            print(info)
            break
