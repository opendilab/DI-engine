"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
from sc2learner.optimizer.alphastar_rl_loss import AlphaStarRLLoss
from sc2learner.optimizer.base_optimizer import BaseOptimizer


class AlphaStarRLOptimizer(BaseOptimizer):
    def __init__(self, agent, train_config, model_config):
        loss = AlphaStarRLLoss(agent, train_config, model_config)
        super(AlphaStarRLOptimizer, self).__init__(agent, loss, train_config)
