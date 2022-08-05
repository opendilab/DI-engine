from typing import Any, List, Union, Optional, Dict
import gym
import logging
import numpy as np
from easydict import EasyDict
from functools import reduce

from ding.envs import BaseEnv, BaseEnvTimestep, FrameStackWrapper
from ding.torch_utils import to_ndarray, to_list
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY, import_module

from .environment import MultiAgentEnv
from .scenarios import load

logging.basicConfig(level=logging.WARNING)


@ENV_REGISTRY.register('mpe')
class MPEEnv(BaseEnv):
    # Now only supports simple_spread
    
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False # whether the environment begins
        self._replay_path = None
        self._scenario_name = self._cfg.env_id
        self._num_agents = self._cfg.n_agent
        self._num_landmarks = self._cfg.n_landmark
        self._max_cycles = self._cfg.get('max_cycles', 25)
        self._agent_specific_global_state = self._cfg.get('agent_specific_global_state', False)
        logging.debug('env.init')


    def reset(self) -> np.ndarray:
        if not self._init_flag:
            create_args = EasyDict(
                scenario_name = self._scenario_name,
                num_agents = self._num_agents,
                num_landmarks = self._num_landmarks,
                episode_length = self._max_cycles,
            )
            # load scenario from args
            scenario = load(create_args.scenario_name + ".py").Scenario()
            # create world
            world = scenario.make_world(create_args)
            # create multiagent environment
            self._env = MultiAgentEnv(world, scenario.reset_world,
                                scenario.reward, scenario.observation, scenario.info)
        # dynamic seed, different seed in each training env    
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.Monitor(
                self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )
        obs = self._env.reset()
        if not self._init_flag:
            # set _action_space
            self._agents = self._env.agents
            self._action_space = gym.spaces.Dict({'agent' + str(i): self._env.action_space[i] for i in range(self._num_agents)})
            single_agent_obs_space = self._env.action_space[0]
            if isinstance(single_agent_obs_space, gym.spaces.Discrete):
                self._action_dim = (single_agent_obs_space.n,)
            else:
                raise Exception('Only support `Discrete` obs space for single agent.')
            
            # set _observation_space
            if not self._cfg.agent_obs_only:
                self._observation_space = gym.spaces.Dict({
                    'agent_state':
                    gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(self._num_agents, self._env.observation_space[0].shape[0]),
                        dtype=np.float32
                    ),
                    'global_state':
                    gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(self._num_agents * 4 + self._num_landmarks * 2, self._num_agents * (self._num_agents - 1) * 2,),
                        dtype=np.float32,
                    ),
                    'agent_alone_state':
                    gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(self._num_agents, 4 + self._num_landmarks * 2 + (self._num_agents - 1) * 2),
                        dtype=np.float32,
                    ),
                    'agent_alone_padding_state':
                    gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(self._num_agents, self._env.observation_space[0].shape[0]),
                        dtype=np.float32,
                    ),
                    'action_mask':
                    gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(self._num_agents, self._action_dim[0]),
                        dtype=np.float32,
                    )
                })
                # whether use agent specific global state:
                if self._agent_specific_global_state:
                    agent_specific_global_state = gym.spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(self._num_agents, self._env.observation_space[0].shape[0] +
                        self._num_agents * 4 + self._num_landmarks * 2, self._num_agents * (self._num_agents - 1) * 2,),
                        dtype=np.float32,
                    )
                    self._observation_space['global_state'] = agent_specific_global_state
            else:
                # for case when env.agent_obs_only = True
                self._observation_space = gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=(self._num_agents, self._env.observation_space[0].shape[0]), 
                    dtype=np.float32,
                )
            # set reward_space
            self._reward_space = gym.spaces.Dict({
                'agent' + str(i): gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=(1,),
                    dtype=np.float32,
                ) for i in range(self._num_agents)
            })
            self._init_flag = True
        self._final_eval_reward = 0.
        self._step_count = 0    # env step counter
        obs_n = self._process_obs(obs)
        logging.debug('env reset')
        return obs_n

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False
        logging.debug('env close')
            
    def render(self) -> None:
        self._env.render()
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        logging.debug('env seed')
    
    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        logging.debug('env step start')
        logging.debug(f'origin action: {action}')
        self._step_count += 1
        assert isinstance(action, np.ndarray), type(action)
        action = self._process_action(action)

        obs, rew, done, info = self._env.step(action)
        info = {
            'individual_reward'+str(i): info[i]['individual_reward'] for i in range(self._num_agents)
        }
        obs_n = self._process_obs(obs)
        # all agents' reward is originally the sum of the reward
        rew_n = np.array(rew[0])
        self._final_eval_reward += rew_n
        done_n = reduce(lambda x, y: x and y, done) or self._step_count >= self._max_cycles
        if done_n:  # or reduce(lambda x, y: x and y, done.values())
            info['final_eval_reward'] = self._final_eval_reward
        logging.debug('env step end')
        return BaseEnvTimestep(obs_n, rew_n, done_n, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def _process_obs(self, obs: List[np.ndarray]) -> np.ndarray:  # noqa
        logging.debug('env process obs')
        obs = np.array(obs).astype(np.float32)
        if self._cfg.get('agent_obs_only', False):
            return obs
        ret = {}
        # Raw agent observation structure is --
        # [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
        # where `communication` are signals from other agents (two for each agent in `simple_spread_v2`` env)

        # agent_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2).
        #              Stacked observation. Contains
        #              - agent itself's state(velocity + position)
        #              - position of items that the agent can observe(e.g. other agents, landmarks)
        #              - communication
        ret['agent_state'] = obs
        # global_state: Shape (n_agent * (2 + 2) + n_landmark * 2 + n_agent * (n_agent - 1) * 2, ).
        #               1-dim vector. Contains
        #               - all agents' state(velocity + position) +
        #               - all landmarks' position +
        #               - all agents' communication
        ret['global_state'] = np.concatenate(
            [
                obs[0, 2:-(self._num_agents - 1) * 2],  # all agents' position + all landmarks' position
                obs[:, 0:2].flatten(),  # all agents' velocity
                obs[:, -(self._num_agents - 1) * 2:].flatten()  # all agents' communication
            ]
        )
        # agent_specific_global_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) + n_landmark * 2 + n_agent * (n_agent - 1) * 2).
        #               2-dim vector. contains
        #               - agent_state info
        #               - global_state info
        if self._agent_specific_global_state:
            ret['global_state'] = np.concatenate(
                [
                    ret['agent_state'],
                    np.expand_dims(ret['global_state'], axis=0).repeat(self._num_agents, axis=0)
                ],
                axis=1
            )
        # agent_alone_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2).
        #                    Stacked observation. Exclude other agents' positions from agent_state. Contains
        #                    - agent itself's state(velocity + position) +
        #                    - landmarks' positions (do not include other agents' positions)
        #                    - communication
        ret['agent_alone_state'] = np.concatenate(
            [
                obs[:, 0:(4 + self._num_landmarks * 2)],  # agent itself's state + landmarks' position
                obs[:, -(self._num_agents - 1) * 2:],  # communication
            ],
            1
        )
        # agent_alone_padding_state: Shape (n_agent, 2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2).
        #                            Contains the same information as agent_alone_state;
        #                            But 0-padding other agents' positions.
        ret['agent_alone_padding_state'] = np.concatenate(
            [
                obs[:, 0:(4 + self._num_agents * 2)],  # agent itself's state + landmarks' position
                np.zeros((self._num_agents,
                          (self._num_agents - 1) * 2), np.float32),  # Other agents' position(0-padding)
                obs[:, -(self._num_agents - 1) * 2:]  # communication
            ],
            1
        )
        # action_mask: All actions are of use(either 1 for discrete or 5 for continuous). Thus all 1.
        ret['action_mask'] = np.ones((self._num_agents, *self._action_dim))
        return ret

    def _process_action(self, action: np.ndarray) -> List[List[int]]:
        logging.debug('env process action')
        action_array = np.zeros((self._num_agents, self._action_dim[0]), dtype=np.int64)
        action_array[np.arange(self._num_agents), action] = 1
        action_list = action_array.tolist()
        return action_list

    def random_action(self) -> np.ndarray:
        random_action_array = np.array([self._action_space['agent'+str(i)].sample() for i in range(self._num_agents)], dtype=np.int64)
        return random_action_array

    def __repr__(self) -> str:
        return "DI-engine MPE Env"
    
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


# def MPEEnv(args):
#     '''
#     Creates a MultiAgentEnv object as env. This can be used similar to a gym
#     environment by calling env.reset() and env.step().
#     Use env.render() to view the environment on the screen.

#     Input:
#         scenario_name   :   name of the scenario from ./scenarios/ to be Returns
#                             (without the .py extension)
#         benchmark       :   whether you want to produce benchmarking data
#                             (usually only done during evaluation)

#     Some useful env properties (see environment.py):
#         .observation_space  :   Returns the observation space for each agent
#         .action_space       :   Returns the action space for each agent
#         .n                  :   Returns the number of Agents
#     '''

#     # load scenario from script
#     scenario = load(args.scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world(args)
#     # create multiagent environment
#     env = MultiAgentEnv(world, scenario.reset_world,
#                         scenario.reward, scenario.observation, scenario.info)

#     return env
