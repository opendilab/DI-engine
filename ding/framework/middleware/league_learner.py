from dataclasses import dataclass


@dataclass
class LearnerModel:
    player_id: str
    state_dict: dict
    train_iter: int = 0
