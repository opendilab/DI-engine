from typing import Any, Union, List
import copy
import numpy as np
import gym
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray, to_list
from .mujoco_multi import MujocoMulti
from ding.utils import ENV_REGISTRY
from namedlist import namedlist
from numpy import dtype


@ENV_REGISTRY.register('mujoco_multi')
class MujocoEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._cfg.seed = self._seed + np_seed
        elif hasattr(self, '_seed'):
            self._cfg.seed = self._seed
        if not self._init_flag:
            self._env = MujocoMulti(env_args=self._cfg)
            self._init_flag = True
        obs = self._env.reset()
        self._final_eval_reward = 0.

        # TODO: 
        # self.env_info for scenario='Ant-v2', agent_conf="2x4d", 
        # {'state_shape': 2, 'obs_shape': 54,...}
        # 'state_shape' is wrong, it should be 111
        self.env_info = self._env.get_env_info()
        # self._env.observation_space[agent].shape equals above 'state_shape'

        self._num_agents = self.env_info['n_agents']
        self._agents = [i for i in range(self._num_agents)]
        self._observation_space = [gym.spaces.Dict({
            'agent_state':
                gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=obs['agent_state'].shape[1:],
                    dtype=np.float32
                ), 
            'global_state':
                gym.spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=obs['global_state'].shape[1:],
                    dtype=np.float32
                ),
        }) for agent in self._agents]
        self._action_space = gym.spaces.Dict({agent: self._env.action_space[agent] for agent in self._agents})
        single_agent_obs_space = self._env.action_space[self._agents[0]]
        if isinstance(single_agent_obs_space, gym.spaces.Box):
            self._action_dim = single_agent_obs_space.shape
        elif isinstance(single_agent_obs_space, gym.spaces.Discrete):
            self._action_dim = (single_agent_obs_space.n, )
        else:
            raise Exception('Only support `Box` or `Discrte` obs space for single agent.')
        self._reward_space = gym.spaces.Dict(
            {
                agent: gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)
                for agent in self._agents
            }
        )

        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        self._final_eval_reward += rew
        rew = to_ndarray([rew])  # wrapped to be transfered to a array with shape (1,)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def num_agents(self) -> Any:
        return self._num_agents

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "DI-engine Multi-agent Mujoco Env({})".format(self._cfg.env_id)
