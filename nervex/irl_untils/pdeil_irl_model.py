import numpy as np
import pickle
import scipy.stats as stats
from sklearn.svm import SVC
from .base_reward_estimate import BaseRewardModel
# from abc import ABC, abstractmethod


class PdeilRewardModel(BaseRewardModel):

    def __init__(self, cfg: dict) -> None:
        super(PdeilRewardModel, self).__init__()
        self.config: dict = cfg
        self.e_u_s = None
        self.e_sigma_s = None
        if cfg['discrete_action']:
            self.svm = None
        else:
            self.e_u_s_a = None
            self.e_sigma_s_a = None
        self.launch()
        self.p_u_s = None
        self.p_sigma_s = None
        self.expert_data = None
        self.train_data: list = []

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
        if self.config['discrete_action']:
            self.svm: SVC = SVC(probability=True)
            self.svm.fit(states, actions)
        else:
            # states action conjuct
            state_actions = np.concatenate((states, actions), axis=1)
            self.e_u_s_a = np.mean(state_actions, axis=0)
            self.e_sigma_s_a = np.cov(state_actions, rowvar=False)

    def _train(self, data: list) -> None:
        # 这里的data， 我们只需要收集当前策略的状态
        states: np.ndarray = np.array(data)
        self.p_u_s = np.mean(states, axis=0)
        self.p_sigma_s = np.cov(states, rowvar=False)

    def train(self):
        self._train(self.train_data)

    def estimate(self, s, a) -> float:
        if self.p_u_s is None:
            print("you need to train you reward model first")
            return 0
        else:
            rho_1 = stats.multivariate_normal.pdf(x=s, mean=self.e_u_s, cov=self.e_sigma_s, allow_singular=False)
            rho_2 = stats.multivariate_normal.pdf(x=s, mean=self.p_u_s, cov=self.p_sigma_s, allow_singular=False)
            state = s.reshape((1, -1))
            if self.config['discrete_action']:
                rho_3 = self.svm.predict_proba(state)[0][a]
            else:
                s_a = np.concatenate([s, a])
                rho_3 = stats.multivariate_normal.pdf(
                    x=s_a, mean=self.e_u_s_a, cov=self.e_sigma_s_a, allow_singular=False
                )
                rho_3 = rho_3 / rho_1
            alpha = self.config['alpha']
            beta = 1 - alpha
            den = rho_1 * rho_3
            frac = alpha * rho_1 + beta * rho_2
            if frac == 0:
                # 这个东西需要新增
                return 0.0
            else:
                return den / frac

    def collect_data(self, item):
        self.train_data.append(item)

    def clear_data(self):
        self.train_data.clear()
