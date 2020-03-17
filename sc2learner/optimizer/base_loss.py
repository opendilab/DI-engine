"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""


class BaseLoss:
    def compute_loss(self, data):
        raise NotImplementedError()

    def register_log(self, variable_record, tb_logger):
        """Input variable record and tensorboard logger. Return nothing."""
        raise NotImplementedError()
