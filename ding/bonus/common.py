from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingReturn:
    '''
    Attributions
    wandb_url: The weight & biases (wandb) project url of the trainning experiment.
    '''
    wandb_url: str


@dataclass
class EvalReturn:
    '''
    Attributions
    eval_value: The mean of evaluation return.
    eval_value_std: The standard deviation of evaluation return.
    '''
    eval_value: np.float32
    eval_value_std: np.float32
