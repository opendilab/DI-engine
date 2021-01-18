from collections import namedtuple
import torch
from nervex.envs import BaseEnv, register_env, BaseEnvTimestep, BaseEnvInfo 

FakeSMACEnvTimestep = namedtuple('FakeSMACEnvTimestep', ['obs', 'reward', 'done', 'info'])
FakeSMACEnvInfo = namedtuple('FakeSMACEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

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
            'agent_state': torch.randn(self.agent_num, self.obs_dim),
            'agent_alone_state': torch.randn(self.agent_num, self.obs_alone_dim),
            'agent_alone_padding_state': torch.randn(self.agent_num, self.obs_dim),
            'global_state': torch.randn(self.global_obs_dim),
            'action_mask': torch.randint(0, 2, size=(self.agent_num, self.action_dim)),
        }

    def step(self, action):
        assert action.shape == (self.agent_num, ), action.shape
        obs = self._get_obs()
        reward = torch.randint(0, 10, size=(1, ))
        done = self.step_count >= 314
        info = {}
        if done:
            info['final_eval_reward'] = 0.71
        self.step_count += 1
        return FakeSMACEnvTimestep(obs, reward, done, info)

    def info(self):
        T = FakeSMACEnvInfo
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


register_env('fake_smac', FakeSMACEnv)
