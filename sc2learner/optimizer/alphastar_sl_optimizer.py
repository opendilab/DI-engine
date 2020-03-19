"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
from sc2learner.optimizer.alphastar_sl_loss import AlphaStarSupervisedLoss
from sc2learner.optimizer.base_optimizer import BaseOptimizer


class AlphaStarSupervisedOptimizer(BaseOptimizer):
    def __init__(self, agent, train_config, model_config):
        loss = AlphaStarSupervisedLoss(agent, train_config, model_config)
        super(AlphaStarSupervisedOptimizer, self).__init__(agent, loss, train_config)
