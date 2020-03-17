"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""

from sc2learner.optimizer.alphastar_sl_loss import AlphaStarSupervisedLoss
from sc2learner.optimizer.base_optimizer import BaseOptimizer
from sc2learner.utils import override


class AlphaStarSupervisedOptimizer(BaseOptimizer):
    def __init__(self, cfg, agent, use_distributed, world_size):
        loss = AlphaStarSupervisedLoss(cfg=cfg, agent=agent)
        super(AlphaStarSupervisedOptimizer, self).__init__(loss, cfg, agent, use_distributed, world_size)

    @override(BaseOptimizer)
    def process_loss(self, loss, data=None, var_items=None):
        avg_loss = loss / self.world_size
        return avg_loss
