from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym.spaces.discrete import Discrete


class MaskDiscrete(Discrete):
    def sample(self, availables):
        x = np.random.choice(availables).item()
        assert self.contains(x, availables)
        return x

    def contains(self, x, availables):
        return super(MaskDiscrete, self).contains(x) and x in availables

    def __repr__(self):
        return "MaskDiscrete(%d)" % self.n
