from .base import BaseRLAlgorithm
from .rl_losses import td_lambda_loss, pg_loss


class PPO(BaseRLAlgorithm):
    # overwrite
    def __init__(self, cfg):
        self.upgo_weight = cfg.rl.upgo_weight
        self.ent_weight = cfg.rl.ent_weight
        self.vtrace_gamma = cfg.rl.vtrace_gamma
        self.vtrace_lambda = cfg.rl.vtrace_lambda
        self.value_gamma = cfg.rl.value_gamma  # default to 1
        self.value_lambda = cfg.rl.value_lambda

    # overwrite
    def __call__(self, inputs):
        return_val = {}
        return_val['value_loss'] = td_lambda_loss(
            inputs['rewards'], inputs['values'], gamma=self.value_gamma, lambda_=self.value_lambda
        )
        return_val['pg_loss'] = pg_loss(
            inputs['action_logits'],
            inputs['current_logits'],
            inputs['action'],
            inputs['rewards'],
            inputs['values'],
            upgo_weight=self.upgo_weight,
            ent_weight=self.ent_weight,
            vtrace_gamma=self.vtrace_gamma,
            vtrace_lambda=self.vtrace_lambda
        )
        return return_val

    # overwrite
    def __repr__(self):
        raise NotImplementedError
