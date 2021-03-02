from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseRewardModel(ABC):

    """
        the base class of reward model
    """
    
    @abstractmethod
    def estimate(self, s, a) -> float: 
        raise NotImplementedError

    @abstractmethod
    def train(self, data) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def load_expert_data(self, data) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def collect_data(self,data) -> None:
        raise NotImplementedError
    
    
    

