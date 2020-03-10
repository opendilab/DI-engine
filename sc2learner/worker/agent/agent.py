import torch
from sc2learner.torch_utils import build_checkpoint_helper


class BaseAgent(object):

    def __init__(self, env, model, tb_logger, cfg):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.tb_logger = tb_logger
        self.viz = cfg.logger.viz
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        assert(cfg.common.load_path != "")
        self.checkpoint_helper.load(cfg.common.load_path, self.model)

    def reset(self):
        raise NotImplementedError

    def act(self, obs):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError
