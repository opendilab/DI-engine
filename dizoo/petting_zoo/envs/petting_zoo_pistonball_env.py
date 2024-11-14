from functools import reduce
from typing import List, Optional, Dict

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from dizoo.petting_zoo.envs.petting_zoo_simple_spread_env import PTZRecordVideo
from pettingzoo.butterfly import pistonball_v6


@ENV_REGISTRY.register('petting_zoo_pistonball')
class PettingZooPistonballEnv(BaseEnv):
    """
    DI-engine PettingZoo environment adapter for the Pistonball environment.
    This class integrates the `pistonball_v6` environment into the DI-engine
    framework, supporting both continuous and discrete actions.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._num_pistons = self._cfg.get('n_pistons', 20)
        self._continuous_actions = self._cfg.get('continuous_actions', False)
        self._max_cycles = self._cfg.get('max_cycles', 125)
        self._act_scale = self._cfg.get('act_scale', False)
        self._agent_specific_global_state = self._cfg.get('agent_specific_global_state', False)
        if self._act_scale:
            assert self._continuous_actions, 'Action scaling only applies to continuous action spaces.'
        self._channel_first = self._cfg.get('channel_first', True)

    def reset(self) -> np.ndarray:
        """
        Resets the environment and returns the initial observations.
        """
        if not self._init_flag:
            # Initialize the pistonball environment
            parallel_env = pistonball_v6.parallel_env
            self._env = parallel_env(
                n_pistons=self._num_pistons,
                continuous=self._continuous_actions,
                max_cycles=self._max_cycles
            )
            self._env.reset()
            self._agents = self._env.agents

            # Define action and observation spaces
            self._action_space = gym.spaces.Dict({agent: self._env.action_space(agent) for agent in self._agents})
            single_agent_obs_space = self._env.observation_space(self._agents[0])
            single_agent_action_space = self._env.action_space(self._agents[0])

            if isinstance(single_agent_action_space, gym.spaces.Box):
                self._action_dim = single_agent_action_space.shape
            elif isinstance(single_agent_action_space, gym.spaces.Discrete):
                self._action_dim = (single_agent_action_space.n, )
            else:
                raise Exception('Only support `Box` or `Discrete` obs space for single agent.')

            if isinstance(single_agent_obs_space, gym.spaces.Box):
                self._obs_shape = single_agent_obs_space.shape
            else:
                raise ValueError("Only support `Box` observation space for each agent.")

            self._observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self._num_pistons, *self._obs_shape), dtype=np.uint8
            )

            self._reward_space = gym.spaces.Dict(
                {
                    agent: gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=np.float32)
                    for agent in self._agents
                }
            )

            if self._replay_path is not None:
                self._env.render_mode = 'rgb_array'
                self._env = PTZRecordVideo(self._env, self._replay_path, name_prefix=f'rl-video-{id(self)}', disable_logger=True)
            self._init_flag = True

        if hasattr(self, '_seed'):
            obs = self._env.reset(seed=self._seed)
        else:
            obs = self._env.reset()

        self._eval_episode_return = 0.0
        self._step_count = 0
        obs_n = self._process_obs(obs)
        return obs_n

    def close(self) -> None:
        """
        Closes the environment.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        """
        Renders the environment.
        """
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Sets the seed for the environment.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        """
        Steps through the environment using the provided action.
        """
        self._step_count += 1
        assert isinstance(action, np.ndarray), type(action)
        action = self._process_action(action)
        if self._act_scale:
            for agent in self._agents:
                action[agent] = affine_transform(action[agent], min_val=self.action_space[agent].low, max_val=self.action_space[agent].high)

        obs, rew, done, trunc, info = self._env.step(action)
        obs_n = self._process_obs(obs)
        rew_n = np.array([sum([rew[agent] for agent in self._agents])])
        rew_n = rew_n.astype(np.float32)
        self._eval_episode_return += rew_n.item()

        done_n = reduce(lambda x, y: x and y, done.values()) or self._step_count >= self._max_cycles
        if done_n:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(obs_n, rew_n, done_n, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enables video recording during the episode.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def _process_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Processes the observations into the required format.
        """
        # Process agent observations, transpose if channel_first is True
        obs = np.array(
            [np.transpose(obs[agent], (2, 0, 1)) if self._channel_first else obs[agent]
             for agent in self._agents],
            dtype=np.uint8
        )

        # Return only agent observations if configured to do so
        if self._cfg.get('agent_obs_only', False):
            return obs

        # Initialize return dictionary
        ret = {
            'agent_state': (obs / 255.0).astype(np.float32)
        }

        # Obtain global state, transpose if channel_first is True
        global_state = self._env.state()
        if self._channel_first:
            global_state = global_state.transpose(2, 0, 1)
        ret['global_state'] = (global_state / 255.0).astype(np.float32)

        # Handle agent-specific global states by repeating the global state for each agent
        if self._agent_specific_global_state:
            ret['global_state'] = np.tile(
                np.expand_dims(ret['global_state'], axis=0),
                (self._num_pistons, 1, 1, 1)
            )

        # Set action mask for each agent
        ret['action_mask'] = np.ones((self._num_pistons, *self._action_dim), dtype=np.float32)

        return ret

    def _process_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Processes the action array into a dictionary format for each agent.
        """
        dict_action = {}
        for i, agent in enumerate(self._agents):
            dict_action[agent] = action[i]
        return dict_action

    def random_action(self) -> np.ndarray:
        """
        Generates a random action for each agent.
        """
        random_action = self.action_space.sample()
        for k in random_action:
            if isinstance(random_action[k], np.ndarray):
                pass
            elif isinstance(random_action[k], int):
                random_action[k] = to_ndarray([random_action[k]], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        return "DI-engine PettingZoo Pistonball Env"

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space