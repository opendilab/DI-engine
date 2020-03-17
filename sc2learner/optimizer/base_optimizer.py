"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""

import torch

from sc2learner.optimizer.base_loss import BaseLoss
from sc2learner.utils import EasyTimer


class BaseOptimizer:
    def __init__(self, loss, cfg, agent, use_distributed, world_size):
        assert isinstance(loss, BaseLoss)
        self.loss = loss

        self.model = agent.get_model()

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=float(cfg.train.learning_rate),
            weight_decay=float(cfg.train.weight_decay)
        )

        self.use_distributed = use_distributed
        self.world_size = world_size
        self.forward_timer = EasyTimer()
        self.backward_timer = EasyTimer()
        self.sync_gradients_timer = EasyTimer(cuda=False)  # Don't let timer wait for GPU works

    def register_stats(self, variable_record, tb_logger):
        variable_record.register_var('total_loss')
        variable_record.register_var('backward_time')
        variable_record.register_var('forward_time')
        variable_record.register_var('sync_gradients_time')
        tb_logger.register_var('total_loss')
        tb_logger.register_var('backward_time')
        tb_logger.register_var('sync_gradients_time')
        self.loss.register_log(variable_record, tb_logger)

    def state_dict(self):
        # FIXME(pzh, nyz) should we add optimizer state to the agent / learner state??
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def learn(self, data):
        # forward pass
        with self.forward_timer:
            var_items = self.loss.compute_loss(data)
            loss = var_items["total_loss"]

        # backward pass
        with self.backward_timer:
            self.optimizer.zero_grad()
            loss = self.process_loss(loss, data, var_items)
            loss.backward()
            with self.sync_gradients_timer:
                if self.use_distributed:
                    self.model.sync_gradients()
            self.optimizer.step()

        stats = dict(
            forward_time=self.forward_timer.value,
            backward_time=self.backward_timer.value,
            sync_gradients_time=self.sync_gradients_timer.value
        )
        return var_items, stats

    def process_loss(self, loss, data=None, var_items=None):
        return loss
