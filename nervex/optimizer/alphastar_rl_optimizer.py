"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import os.path as osp
from nervex.utils import override, merge_dicts, pretty_print, read_config
from nervex.model import alphastar_model_default_config
from nervex.optimizer.alphastar_rl_loss import AlphaStarRLLoss
from nervex.optimizer.base_optimizer import BaseOptimizer

default_config = read_config(osp.join(osp.dirname(__file__), "alphastar_rl_optimizer_default_config.yaml"))


def build_config(user_config):
    """Aggregate a general config at the highest level class: Learner"""
    default_config_with_model = merge_dicts(default_config, alphastar_model_default_config)
    return merge_dicts(default_config_with_model, user_config)


class AlphaStarRLOptimizer(BaseOptimizer):
    def __init__(self, agent, cfg):
        cfg = build_config(cfg)
        train_config = cfg.train
        model_config = cfg.model
        loss = AlphaStarRLLoss(agent, train_config, model_config)
        super(AlphaStarRLOptimizer, self).__init__(agent, loss, train_config)
