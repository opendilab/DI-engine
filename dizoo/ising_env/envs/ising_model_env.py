from typing import Any, Optional
import copy
import os
import sys
import numpy as np
from numpy import dtype
import gym
import matplotlib.pyplot as plt
import imageio
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from dizoo.ising_env.envs.ising_model.multiagent.environment import IsingMultiAgentEnv
import dizoo.ising_env.envs.ising_model as ising_model_


@ENV_REGISTRY.register('ising_model')
class IsingModelEnv(BaseEnv):
    """
    Overview:
        Ising Model Environment for Multi-Agent Reinforcement Learning according to the paper: \
        [Mean Field Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1802.05438). \
        The environment is a grid of agents, each of which can be in one of two states: \
        spin up or spin down. The agents interact with their neighbors according to the Ising model, \
        and the goal is to maximize the global order parameter, which is the average spin of all agents. \
        Details of the environment can be found in the \
        [DI-engine-Doc](https://di-engine-docs.readthedocs.io/zh-cn/latest/13_envs/index.html).
    Interface:
        `__init__`, `reset`, `close`, `seed`, `step`, `random_action`, `num_agents`, \
        `observation_space`, `action_space`, `reward_space`.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._action_space = gym.spaces.Discrete(cfg.dim_spin)  # default 2
        self._observation_space = gym.spaces.MultiBinary(4 * cfg.agent_view_sight)
        self._reward_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)
        self._replay_path = None

    def calculate_action_prob(self, actions):
        num_action = self._action_space.n
        N = actions.shape[0]  # agent_num
        # Convert actions to one_hot encoding
        one_hot_actions = np.eye(num_action)[actions.flatten()]
        action_prob = np.zeros((N, num_action))

        for i in range(N):
            # Select only the one_hot actions of agents visible to agent i
            visible_actions = one_hot_actions[self._env.agents[i].spin_mask == 1]
            if visible_actions.size > 0:
                # Calculate the average of the one_hot encoding for visible agents only
                action_prob[i] = visible_actions.mean(axis=0)
            else:
                # If no visible agents, action_prob remains zero for agent i
                action_prob[i] = np.zeros(num_action)

        return action_prob

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._cfg.seed = self._seed + np_seed
        elif hasattr(self, '_seed'):
            self._cfg.seed = self._seed
        if not self._init_flag:
            # self._env = MujocoMulti(env_args=self._cfg)
            ising_model = ising_model_.load('Ising.py').Scenario()
            self._env = IsingMultiAgentEnv(
                world=ising_model.make_world(num_agents=self._cfg.num_agents, agent_view=1),
                reset_callback=ising_model.reset_world,
                reward_callback=ising_model.reward,
                observation_callback=ising_model.observation,
                done_callback=ising_model.done
            )
            self._init_flag = True
        obs = self._env._reset()
        obs = np.stack(obs)
        self.pre_action = np.zeros(self._cfg.num_agents, dtype=np.int32)
        # consider the last global state as pre action prob
        pre_action_prob = self.calculate_action_prob(self._env.world.global_state.flatten().astype(int))
        obs = np.concatenate([obs, pre_action_prob], axis=1)
        obs = to_ndarray(obs).astype(np.float32)
        self._eval_episode_return = 0
        self.cur_step = 0
        if self._replay_path is not None:
            self._frames = []
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        action = to_ndarray(action)
        if len(action.shape) == 1:
            action = np.expand_dims(action, axis=1)
        obs, rew, done, order_param, ups, downs = self._env._step(action)
        info = {"order_param": order_param, "ups": ups, "downs": downs, 'pre_action': self.pre_action}
        pre_action_prob = self.calculate_action_prob(self.pre_action)
        self.pre_action = action
        obs = np.stack(obs)
        obs = np.concatenate([obs, pre_action_prob], axis=1)
        obs = to_ndarray(obs).astype(np.float32)
        rew = np.stack(rew)
        self._eval_episode_return += np.sum(rew)
        self.cur_step += 1

        if self._replay_path is not None:
            # transform the action to a 2D grid. e.g. (100,) -> (10, 10)
            action_matrix = action.reshape((int(np.sqrt(self._cfg.num_agents)), -1))
            self._frames.append(self.render(action_matrix, info))

        done = done[0]  # dones are the same for all agents
        if done:
            info['eval_episode_return'] = self._eval_episode_return / self.cur_step
            if self._replay_path is not None:
                path = os.path.join(self._replay_path, '{}_episode.gif'.format(self._save_replay_count))
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        if not os.path.exists(replay_path):
            os.makedirs(replay_path)
        self._save_replay_count = 0

    def render(self, action_matrix, info) -> None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(action_matrix, cmap='gray', vmin=0, vmax=1, interpolation='none')
        ax.set_title(f"Step {self.cur_step}: Order={info['order_param']}, Up={info['ups']}, Down={info['downs']}")
        # save the figure to buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)
        return image

    @staticmethod
    def display_frames_as_gif(frames: list, output_path: str) -> None:
        imageio.mimsave(output_path, frames, duration=50)

    @property
    def num_agents(self) -> Any:
        return self._env.n

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space[0]

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space[0]

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Ising Model Env({})".format(self._cfg.env_id)
