from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseRewardModel(ABC):

    """
        the base class of reward model
    """

    @abstractmethod
    def __init__(self, cfg: dict) -> None:
        """
        cfg need to point out that the path of expert data
        and the reward data hyper para
        """
        raise NotImplementedError

    @abstractmethod
    def estimate(self, s: np.ndarray, a: np.ndarray) -> float: 
        raise NotImplementedError

    @abstractmethod
    def train(self, data) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def load_expert_data(self, data) -> None:
        raise NotImplementedError
    
    
    

