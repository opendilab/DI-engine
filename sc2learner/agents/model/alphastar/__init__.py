from .policy_network import Policy


def build_model(cfg):
    return Policy(cfg.model.policy)
