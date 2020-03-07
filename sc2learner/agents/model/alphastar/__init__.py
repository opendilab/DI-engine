from .actor_critic import AlphaStarActorCritic


def build_model(cfg):
    return AlphaStarActorCritic(cfg.model)
