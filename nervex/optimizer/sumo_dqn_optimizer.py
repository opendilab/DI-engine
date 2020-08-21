from nervex.optimizer.sumo_dqn_loss import SumoDqnLoss
from nervex.optimizer.base_optimizer import BaseOptimizer


class SumoDqnOptimizer(BaseOptimizer):
    def __init__(self, agent, train_config):
        loss = SumoDqnLoss(agent, train_config)
        super(SumoDqnOptimizer, self).__init__(agent, loss, train_config)
