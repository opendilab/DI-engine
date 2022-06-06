from typing import Any
from dataclasses import dataclass

@dataclass
class ActorData:
    train_data: Any
    env_step: int = 0