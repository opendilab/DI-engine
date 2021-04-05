from .registry import Registry

POLICY_REGISTRY = Registry()
ENV_REGISTRY = Registry()
LEARNER_REGISTRY = Registry()
COMM_LEARNER_REGISTRY = Registry()
ACTOR_REGISTRY = Registry()
COMM_ACTOR_REGISTRY = Registry()
COMMANDER_REGISTRY = Registry()
LEAGUE_REGISTRY = Registry()
PLAYER_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
ENV_MANAGER_REGISTRY = Registry()

registries = {
    'policy': POLICY_REGISTRY,
    'env': ENV_REGISTRY,
    'model': MODEL_REGISTRY,
    'learner': LEARNER_REGISTRY,
    'actor': ACTOR_REGISTRY,
    'env_manager': ENV_MANAGER_REGISTRY,
    'comm_learner': COMM_LEARNER_REGISTRY,
    'comm_actor': COMM_ACTOR_REGISTRY,
    'commander': COMMANDER_REGISTRY,
    'league': LEAGUE_REGISTRY,
    'player': PLAYER_REGISTRY,
}
