from collections import namedtuple
import numpy as np

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY

FakeSMACEnvTimestep = namedtuple('FakeSMACEnvTimestep', ['obs', 'reward', 'done', 'info'])
FakeSMACEnvInfo = namedtuple('FakeSMACEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])


@ENV_REGISTRY.register('fake_smac')
class FakeSMACEnv(BaseEnv):

    def __init__(self, cfg=None):
        self.agent_num = 8
        self.action_dim = 6 + self.agent_num
        self.obs_dim = 248
        self.obs_alone_dim = 216
        self.global_obs_dim = 216

    def reset(self):
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            'agent_state': np.random.random((self.agent_num, self.obs_dim)),
            'agent_alone_state': np.random.random((self.agent_num, self.obs_alone_dim)),
            'agent_alone_padding_state': np.random.random((self.agent_num, self.obs_dim)),
            'global_state': np.random.random((self.global_obs_dim)),
            'action_mask': np.random.randint(0, 2, size=(self.agent_num, self.action_dim)),
        }

    def step(self, action):
        assert action.shape == (self.agent_num, ), action.shape
        obs = self._get_obs()
        reward = np.random.randint(0, 10, size=(1, ))
        done = self.step_count >= 314
        info = {}
        if done:
            info['final_eval_reward'] = 0.71
        self.step_count += 1
        return FakeSMACEnvTimestep(obs, reward, done, info)

    def info(self):
        T = EnvElementInfo
        return FakeSMACEnvInfo(
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

    def close(self):
        pass

    def seed(self, _seed):
        pass

    def __repr__(self):
        return 'FakeSMACEnv'
