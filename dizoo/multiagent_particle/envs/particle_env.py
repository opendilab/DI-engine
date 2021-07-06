from collections import namedtuple
from typing import Any, Optional
from easydict import EasyDict
import copy
import numpy as np
import torch

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray, to_list
from dizoo.multiagent_particle.envs.make_env import make_env
from dizoo.multiagent_particle.envs.multiagent.multi_discrete import MultiDiscrete
import gym
from gym import wrappers


class ParticleEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_name = cfg.get("env_name", "simple")
        self._num_agents = cfg.get("num_agents")
        self._num_landmarks = cfg.get("num_landmarks")
        self._env = make_env(self._env_name, self._num_agents, self._num_landmarks)
        self._env.discrete_action_input = cfg.get('discrete_action', True)
        self._max_step = cfg.get('max_step', 100)
        # self._env.discrete_action_input = True
        self._env.force_discrete_action = True
        self.agent_num = self._env.n

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        if hasattr(self, '_seed'):
            # Note: the real env instance only has a empty seed method, only pass
            self._env.seed(self._seed)
        obs_n = self._env.reset()
        obs_n = to_ndarray(obs_n, np.float32)
        return obs_n

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pass
        self._env.close()

    def seed(self, seed: int, dynamic_seed: bool = False) -> None:
        if dynamic_seed:
            raise NotImplementedError
        self._seed = seed

    def _process_action(self, action: list):
        return to_list(action)

    def step(self, action: list) -> BaseEnvTimestep:
        action = self._process_action(action)
        obs_n, rew_n, done_n, info_n = self._env.step(action)
        obs_n = [to_ndarray(obs, np.float32) for obs in obs_n]
        rew_n = [to_ndarray(rew, np.float32) for rew in rew_n]
        if self._step_count >= self._max_step:
            done_n = True
        return BaseEnvTimestep(obs_n, rew_n, done_n, info_n)

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        act_space = {}
        obs_space = {}
        rew_space = {}
        for i in range(self._env.n):
            obs_space['agent' + str(i)] = T(
                self._env.observation_space[i].shape,
                {
                    'min': -np.inf,
                    'max': +np.inf,
                    'dtype': np.float32
                },
            )
            rew_space['agent' + str(i)] = T(
                (1, ),
                {
                    'min': -np.inf,
                    'max': +np.inf,
                    'dtype': np.float32
                },
            )
            act = self._env.action_space[i]
            if isinstance(act, MultiDiscrete):
                act_space['agent' + str(i)] = T(
                    (act.shape, ),
                    {
                        'min': [int(l) for l in list(act.low)],
                        'max': [int(h) for h in list(act.high)]
                    },
                )
            elif isinstance(act, gym.spaces.Tuple):
                #are not used in our environment yet
                act_space['agent' + str(i)] = T(
                    (len(act.gym.spaces), ),
                    {'space': act.gym.spaces},
                )
            elif isinstance(act, gym.spaces.Discrete):
                act_space['agent' + str(i)] = T(
                    (1, ),
                    {
                        'min': 0,
                        'max': act.n - 1,
                        'dtype': int
                    },
                )
            elif isinstance(act, gym.spaces.Box):
                act_space['agent' + str(i)] = T(
                    act.shape,
                    {
                        'min': act.low,
                        'max': act.high,
                        'dtype': act.dtype
                    },
                )
        return BaseEnvInfo(
            agent_num=self.agent_num, obs_space=obs_space, act_space=act_space, rew_space=rew_space, use_wrappers=None
        )

    def __repr__(self) -> str:
        return "DI-engine wrapped Multiagent particle Env({})".format(self._cfg.env_name)


CNEnvTimestep = namedtuple('CNEnvTimestep', ['obs', 'reward', 'done', 'info'])
CNEnvInfo = namedtuple('CNEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])


# same structure as smac env
@ENV_REGISTRY.register('cooperative_navigation')
class CooperativeNavigation(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_name = 'simple_spread'
        self._n_agent = cfg.n_agent
        self._num_landmarks = cfg.get("num_landmarks", 3)
        self._env = make_env(self._env_name, self._n_agent, self._num_landmarks, True)
        self._env.discrete_action_input = cfg.get('discrete_action', True)
        self._max_step = cfg.get('max_step', 100)
        self._collide_penalty = cfg.get('collide_penal', self._n_agent)
        self._agent_obs_only = cfg.get('agent_obs_only', False)
        self._env.force_discrete_action = cfg.get('force_discrete_action', False)
        self.action_dim = 5
        # obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos)
        self.obs_dim = 2 + 2 + (self._n_agent - 1) * 2 + self._num_landmarks * 2
        self.global_obs_dim = self._n_agent * 2 + self._num_landmarks * 2 + self._n_agent * 2
        self.obs_alone_dim = 2 + 2 + (self._num_landmarks) * 2

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        self._sum_reward = 0
        if hasattr(self, '_seed'):
            # Note: the real env instance only has a empty seed method, only pass
            self._env.seed = self._seed
        obs_n = self._env.reset()
        obs_n = self.process_obs(obs_n)
        return obs_n

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pass
        self._env.close()

    def seed(self, seed: int, dynamic_seed: bool = False) -> None:
        self._seed = seed
        if dynamic_seed:
            raise NotImplementedError
        if hasattr(self, '_seed'):
            # Note: the real env instance only has a empty seed method, only pass
            self._env.seed = self._seed

    def _process_action(self, action: list):
        return to_list(action)

    def process_obs(self, obs: list):
        ret = {}
        obs = np.array(obs).astype(np.float32)
        if self._agent_obs_only:
            return obs
        ret['agent_state'] = obs
        ret['global_state'] = np.concatenate((obs[0, 2:], obs[:, 0:2].flatten()))
        ret['agent_alone_state'] = np.concatenate([obs[:, 0:4], obs[:, -self._num_landmarks * 2:]], 1)
        ret['agent_alone_padding_state'] = np.concatenate(
            [
                obs[:, 0:4],
                np.zeros((self._n_agent, (self._n_agent - 1) * 2), np.float32), obs[:, -self._num_landmarks * 2:]
            ], 1
        )
        ret['action_mask'] = np.ones((self._n_agent, self.action_dim))
        return ret

    # note: the reward is shared between all the agents
    # (see dizoo/multiagent_particle/envs/multiagent/scenarios/simple_spread.py)
    # If you need to make the reward different to each agent, change the code there
    def step(self, action: list) -> BaseEnvTimestep:
        self._step_count += 1
        action = self._process_action(action)
        obs_n, rew_n, _, info_n = self._env.step(action)
        obs_n = self.process_obs(obs_n)
        rew_n = np.array([sum(rew_n)])
        info = info_n

        collide_sum = 0
        for i in range(self._n_agent):
            collide_sum += info['n'][i][1]
        rew_n += collide_sum * (1.0 - self._collide_penalty)
        rew_n = rew_n / (self._max_step * self._n_agent)
        self._sum_reward += rew_n
        occupied_landmarks = info['n'][0][3]
        if self._step_count >= self._max_step or occupied_landmarks >= self._n_agent or occupied_landmarks >= self._num_landmarks:
            done_n = True
        else:
            done_n = False
        if done_n:
            info['final_eval_reward'] = self._sum_reward
        return CNEnvTimestep(obs_n, rew_n, done_n, info)

    def info(self):
        T = EnvElementInfo
        if self._agent_obs_only:
            return CNEnvInfo(
                agent_num=self._n_agent,
                obs_space=T(
                    (self._n_agent, self.obs_dim),
                    None,
                ),
                act_space=T(
                    (self._n_agent, self.action_dim),
                    {
                        'min': 0,
                        'max': self.action_dim,
                        'dtype': int
                    },
                ),
                rew_space=T(
                    (1, ),
                    None,
                )
            )
        return CNEnvInfo(
            agent_num=self._n_agent,
            obs_space=T(
                {
                    'agent_state': (self._n_agent, self.obs_dim),
                    'agent_alone_state': (self._n_agent, self.obs_alone_dim),
                    'agent_alone_padding_state': (self._n_agent, self.obs_dim),
                    'global_state': (self.global_obs_dim, ),
                    'action_mask': (self._n_agent, self.action_dim)
                },
                None,
            ),
            act_space=T(
                (self._n_agent, self.action_dim),
                {
                    'min': 0,
                    'max': self.action_dim,
                    'dtype': int
                },
            ),
            rew_space=T(
                (1, ),
                None,
            )
        )

    def __repr__(self) -> str:
        return "DI-engine wrapped Multiagent particle Env: CooperativeNavigation({})".format(self._env_name)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._env = wrappers.Monitor(self._env, self._replay_path, video_callable=lambda episode_id: True, force=True)
