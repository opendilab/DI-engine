from typing import Any, List
from dataclasses import dataclass, field

#TODO(zms): simplify fields


@dataclass
class ActorDataMeta:
    player_total_env_step: int = 0
    actor_id: int = 0
    send_wall_time: float = 0.0


@dataclass
class ActorEnvTrajectories:
    env_id: int = 0
    trajectories: List = field(default_factory=[])


@dataclass
class ActorData:
    meta: ActorDataMeta
    train_data: List[ActorEnvTrajectories] = field(default_factory=[])
