from typing import Any
from dataclasses import dataclass


@dataclass
class ActorDataMeta:
    palyer_total_env_step: int = 0
    actor_id: int = 0
    env_id: int = 0
    send_wall_time: float = 0.0


@dataclass
class ActorData:
    meta: ActorDataMeta
    # TODO make train data a list in which each env_id has a list
    train_data: Any
