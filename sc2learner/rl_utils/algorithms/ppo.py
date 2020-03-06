from .base import BaseRLAlgorithm


class PPO(BaseRLAlgorithm):
    # overwrite
    def __init__(self, cfg):
        raise NotImplementedError

    # overwrite
    def __call__(self, inputs):
        raise NotImplementedError

    # overwrite
    def __repr__(self):
        raise NotImplementedError
