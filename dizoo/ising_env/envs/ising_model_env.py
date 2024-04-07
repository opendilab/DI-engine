from typing import Any, Union, List
import copy
import os
import sys
import numpy as np
from numpy import dtype
import gym
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY

# ising_model_dir = os.path.join(os.path.dirname(__file__), 'ising_model')
# if ising_model_dir not in sys.path:
#     sys.path.append(ising_model_dir)
# from ising_model.multiagent.environment import IsingMultiAgentEnv
# import ising_model
from dizoo.ising_env.envs.ising_model.multiagent.environment import IsingMultiAgentEnv
import dizoo.ising_env.envs.ising_model as ising_model_


@ENV_REGISTRY.register('ising_model')
class IsingModelEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._action_space = gym.spaces.Discrete(cfg.dim_spin)  # default 2
        self._observation_space = gym.spaces.MultiBinary(4 * cfg.agent_view_sight)
        self._reward_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)

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
        rew = np.concatenate(rew)
        self._eval_episode_return += np.sum(rew)
        self.cur_step += 1

        done = done[0]  # dones are the same for all agents
        if done:
            info['eval_episode_return'] = self._eval_episode_return / self.cur_step
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

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
