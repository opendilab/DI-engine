import numpy as np
import pickle
import scipy.stats as stats 
from sklearn.svm import SVC
# from base_reward_estimate import BaseRewardModel
from abc import ABC, abstractmethod


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


class PdeilRewardModel(BaseRewardModel):

    def __init__(self, cfg: dict) -> None:
        self.config: dict = cfg
        self.e_u_s = None
        self.e_sigma_s = None
        self.svm = None
        self.launch()
        self.p_u_s = None
        self.p_sigma_s = None
        self.expert_data = None
    
    

    def load_expert_data(self) -> None:
        expert_data_path: str = self.config["expert_data_path"]
        with open(expert_data_path, 'rb') as f:
            self.expert_data: list = pickle.load(f)

    def launch(self) -> None:
        self.load_expert_data()
        states: list = []
        actions: list = []
        for item in self.expert_data:
            states.append(item[0])
            actions.append(item[1])
        states: np.ndarray = np.array(states, dtype=np.float32)
        actions: np.ndarray = np.array(actions, dtype=np.int64)
        self.e_u_s: np.ndarray = np.mean(states, axis=0)
        self.e_sigma_s: np.ndarray = np.cov(states, rowvar=False)
        self.svm: SVC = SVC(probability=True)
        self.svm.fit(states, actions)
    
    def train(self, data: list) -> None:
        # 这里的data， 我们只需要收集当前策略的状态
        states: np.ndarray = np.array(data)
        self.p_u_s = np.mean(states, axis=0)
        self.p_sigma_s = np.cov(states, rowvar=False)
    
    def estimate(self, s, a):
        if self.p_u_s is None:
            print("you need to train you reward model first")
            return 0
        else:
            rho_1 = stats.multivariate_normal.pdf(x=s, mean=self.e_u_s, cov=self.e_sigma_s, allow_singular=False)
            rho_2 = stats.multivariate_normal.pdf(x=s, mean=self.p_u_s, cov=self.p_sigma_s, allow_singular=False)
            state = s.reshape((1, -1))
            rho_3 = self.svm.predict_proba(state)[0][a]
            alpha = self.config['alpha']
            beta =  1 - alpha
            den = rho_1 * rho_3
            frac = alpha * rho_1 + beta * rho_2
            if frac == 0:
                # 这个东西需要新增
                return 0
            else:
                return den / frac
        
        


        

