import torch

from nervex.envs.common import EnvElement


class PendulumObs(EnvElement):
    _name = "pendulumObs"

    def _init(self):
        self._default_val = None
        self._shape = (3,)
        self._value = {
            'min': [-1.0, -1.0, -8.0],
            'max': [1.0, 1.0, 8.0],
            'dtype': torch.FloatTensor,
            'dinfo': 'float value tensor of (cos_theta, sin_theta, theta_dot)',
        }

    def _to_agent_processor(self, obs):
        return torch.from_numpy(obs).float()

    def _from_agent_processor(self, obs):
        return obs

    def _details(self):
        return '\t'.join(self._name)
