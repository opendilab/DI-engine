from collections import namedtuple
import gym
import torch
import numpy as np
from gym import spaces
from typing import Any
from nervex.envs import BaseEnv, register_env, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from app_zoo.multiagent_particle.envs.make_env import make_env
from app_zoo.multiagent_particle.envs.multiagent.multi_discrete import MultiDiscrete


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
        obs_n = to_ndarray(obs_n, float)
        return obs_n

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pass
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def _process_action(self, action: list):
        return to_list(action)

    def step(self, action: list) -> BaseEnvTimestep:
        action = self._process_action(action)
        obs_n, rew_n, done_n, info_n = self._env.step(action)
        obs_n = [to_ndarray(obs, float) for obs in obs_n]
        rew_n = [to_ndarray(rew, float) for rew in rew_n]
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
                self._env.observation_space[i].shape, {
                    'min': -np.inf,
                    'max': +np.inf,
                    'dtype': float
                }, None, None
            )
            rew_space['agent' + str(i)] = T((1, ), {'min': -np.inf, 'max': +np.inf, 'dtype': float}, None, None)
            act = self._env.action_space[i]
            if isinstance(act, MultiDiscrete):
                act_space['agent' + str(i)] = T(
                    (act.shape, ), {
                        'min': [int(l) for l in list(act.low)],
                        'max': [int(h) for h in list(act.high)]
                    }, None, None
                )
            elif isinstance(act, spaces.Tuple):
                #are not used in our environment yet
                act_space['agent' + str(i)] = T((len(act.spaces), ), {'space': act.spaces}, None, None)
            elif isinstance(act, spaces.Discrete):
                act_space['agent' + str(i)] = T((1, ), {'min': 0, 'max': act.n - 1, 'dtype': int}, None, None)
            elif isinstance(act, spaces.Box):
                act_space['agent' +
                          str(i)] = T(act.shape, {
                              'min': act.low,
                              'max': act.high,
                              'dtype': act.dtype
                          }, None, None)
        return BaseEnvInfo(
            agent_num=self.agent_num,
            obs_space=obs_space,
            act_space=act_space,
            rew_space=rew_space,
        )

    def __repr__(self) -> str:
        return "nervex wrapped Multiagent particle Env({})".format(self._cfg.env_id)


CNEnvTimestep = namedtuple('CNEnvTimestep', ['obs', 'reward', 'done', 'info'])
CNEnvInfo = namedtuple('CNEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])


# same structure as smac env
class CooperativeNavigation(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._env_name = 'simple_spread'
        self.agent_num = cfg.get("num_agents", 3)
        self._num_landmarks = cfg.get("num_landmarks", 3)
        self._env = make_env(self._env_name, self.agent_num, self._num_landmarks, True)
        self._env.discrete_action_input = cfg.get('discrete_action', True)
        self._max_step = cfg.get('max_step', 100)
        self._collide_penalty = cfg.get('collide_penal', self.agent_num)
        self._env.force_discrete_action = True
        self.action_dim = 5
        # obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos)
        self.obs_dim = 2 + 2 + (self.agent_num - 1) * 2 + self._num_landmarks * 2
        self.global_obs_dim = self.agent_num * 2 + self._num_landmarks * 2 + self.agent_num * 2
        self.obs_alone_dim = 2 + 2 + (self._num_landmarks) * 2

    def reset(self) -> torch.Tensor:
        self._step_count = 0
        self._sum_reward = 0
        if hasattr(self, '_seed'):
            # Note: the real env instance only has a empty seed method, only pass
            self._env.seed(self._seed)
        obs_n = self._env.reset()
        obs_n = self.process_obs(obs_n)
        return obs_n

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pass
        self._env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed

    def _process_action(self, action: list):
        return to_list(action)

    def process_obs(self, obs: list):
        ret = {}
        obs = np.array(obs)
        ret['agent_state'] = obs
        ret['global_state'] = np.concatenate((obs[0, 2:], obs[:, 0:2].flatten()))
        ret['agent_alone_state'] = np.concatenate([obs[:, 0:4], obs[:, -self._num_landmarks * 2:]], 1)
        ret['agent_alone_padding_state'] = np.concatenate(
            [
                obs[:, 0:4],
                np.zeros((self.agent_num, (self.agent_num - 1) * 2), float), obs[:, -self._num_landmarks * 2:]
            ], 1
        )
        ret['action_mask'] = np.ones((self.agent_num, self.action_dim))
        return ret

    # note: the reward is shared between all the agents
    # (see app_zoo/multiagent_particle/envs/multiagent/scenarios/simple_spread.py)
    # If you need to make the reward different to each agent, change the code there
    def step(self, action: list) -> BaseEnvTimestep:
        self._step_count += 1
        action = self._process_action(action)
        obs_n, rew_n, _, info_n = self._env.step(action)
        obs_n = self.process_obs(obs_n)
        rew_n = np.array([sum(rew_n)])
        info = info_n

        collide_sum = 0
        for i in range(self.agent_num):
            collide_sum += info['n'][i][1]
        rew_n += collide_sum * (1.0 - self._collide_penalty)
        self._sum_reward += rew_n
        occupied_landmarks = info['n'][0][3]
        if self._step_count >= self._max_step or occupied_landmarks >= self.agent_num or occupied_landmarks >= self._num_landmarks:
            done_n = True
        else:
            done_n = False
        if done_n:
            info['final_eval_reward'] = self._sum_reward
        return CNEnvTimestep(obs_n, rew_n, done_n, info)

    def info(self):
        T = EnvElementInfo
        return CNEnvInfo(
            agent_num=self.agent_num,
            obs_space=T(
                {
                    'agent_state': (self.agent_num, self.obs_dim),
                    'agent_alone_state': (self.agent_num, self.obs_alone_dim),
                    'agent_alone_padding_state': (self.agent_num, self.obs_dim),
                    'global_state': (self.global_obs_dim, ),
                    'action_mask': (self.agent_num, self.action_dim)
                }, None, None, None
            ),
            act_space=T((self.agent_num, self.action_dim), None, None, None),
            rew_space=T((1, ), None, None, None)
        )

    def __repr__(self) -> str:
        return "nervex wrapped Multiagent particle Env: CooperativeNavigation({})".format(self._cfg.env_id)


register_env('cooperative_navigation', CooperativeNavigation)
