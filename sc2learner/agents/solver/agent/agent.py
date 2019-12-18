import torch
from sc2learner.utils import build_checkpoint_helper


class BaseAgent(object):

    def __init__(self, env, model, cfg):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        assert(cfg.common.load_path != "")
        self.checkpoint_helper.load(cfg.common.load_path, self.model)

    def reset(self):
        raise NotImplementedError

    def act(self, obs):
        raise NotImplementedError
