from .registry import Registry

POLICY_REGISTRY = Registry()
ENV_REGISTRY = Registry()
ENV_WRAPPER_REGISTRY = Registry()
LEARNER_REGISTRY = Registry()
COMM_LEARNER_REGISTRY = Registry()
SERIAL_COLLECTOR_REGISTRY = Registry()
PARALLEL_COLLECTOR_REGISTRY = Registry()
COMM_COLLECTOR_REGISTRY = Registry()
BUFFER_REGISTRY = Registry()
COMMANDER_REGISTRY = Registry()
LEAGUE_REGISTRY = Registry()
PLAYER_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
ENV_MANAGER_REGISTRY = Registry()
REWARD_MODEL_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
SERIAL_EVALUATOR_REGISTRY = Registry()
MQ_REGISTRY = Registry()
WORLD_MODEL_REGISTRY = Registry()

registries = {
    'policy': POLICY_REGISTRY,
    'env': ENV_REGISTRY,
    'env_wrapper': ENV_WRAPPER_REGISTRY,
    'model': MODEL_REGISTRY,
    'reward_model': REWARD_MODEL_REGISTRY,
    'learner': LEARNER_REGISTRY,
    'serial_collector': SERIAL_COLLECTOR_REGISTRY,
    'parallel_collector': PARALLEL_COLLECTOR_REGISTRY,
    'env_manager': ENV_MANAGER_REGISTRY,
    'comm_learner': COMM_LEARNER_REGISTRY,
    'comm_collector': COMM_COLLECTOR_REGISTRY,
    'commander': COMMANDER_REGISTRY,
    'league': LEAGUE_REGISTRY,
    'player': PLAYER_REGISTRY,
    'buffer': BUFFER_REGISTRY,
    'dataset': DATASET_REGISTRY,
    'serial_evaluator': SERIAL_EVALUATOR_REGISTRY,
    'message_queue': MQ_REGISTRY,
    'world_model': WORLD_MODEL_REGISTRY,
}
