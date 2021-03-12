import numpy as np
import torch
import pickle
import scipy.stats as stats
try:
    from sklearn.svm import SVC
except ImportError:
    SVC = None
from nervex.torch_utils import cov
from .base_reward_estimate import BaseRewardModel


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
        self.p_u_s = None
        self.p_sigma_s = None
        self.expert_data = None
        self.train_data: list = []
        # pedil default use cpu device
        self.device = 'cpu'

    def load_expert_data(self) -> None:
        expert_data_path: str = self.config["expert_data_path"]
        with open(expert_data_path, 'rb') as f:
            self.expert_data: list = pickle.load(f)

    def start(self) -> None:
        self.load_expert_data()
        states: list = []
        actions: list = []
        for item in self.expert_data:
            states.append(item['obs'])
            actions.append(item['action'])
        states: torch.Tensor = torch.stack(states, dim=0)
        actions: torch.Tensor = torch.stack(actions, dim=0)
        self.e_u_s: torch.Tensor = torch.mean(states, axis=0)
        self.e_sigma_s: torch.Tensor = cov(states, rowvar=False)
        if self.config["discrete_action"] and SVC is None:
            import logging
            logging.warning("You are using discrete action while the SVC is not installed!")
        if self.config['discrete_action'] and SVC is not None:
            self.svm: SVC = SVC(probability=True)
            self.svm.fit(states.cpu().numpy(), actions.cpu().numpy())
        else:
            # states action conjuct
            state_actions = torch.cat((states, actions.float()), dim=-1)
            self.e_u_s_a = torch.mean(state_actions, axis=0)
            self.e_sigma_s_a = cov(state_actions, rowvar=False)

    def _train(self, states: torch.Tensor) -> None:
        # we only need to collect the current policy state
        self.p_u_s = torch.mean(states, axis=0)
        self.p_sigma_s = cov(states, rowvar=False)

    def train(self):
        states = torch.stack([item['obs'] for item in self.train_data], dim=0)
        self._train(states)

    def _batch_mn_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        ret = []
        for b in range(x.shape[0]):
            ret.append(stats.multivariate_normal.pdf(x[b], mean=mean, cov=cov, allow_singular=False))
        return np.array(ret).astype(np.float32)

    def estimate(self, data: list) -> None:
        """Modify reward inplace"""
        s = torch.stack([item['obs'] for item in data], dim=0)
        a = torch.stack([item['action'] for item in data], dim=0)
        if self.p_u_s is None:
            print("you need to train you reward model first")
            for item in data:
                item['reward'].zero_()
        else:
            rho_1 = self._batch_mn_pdf(s.cpu().numpy(), self.e_u_s.cpu().numpy(), self.e_sigma_s.cpu().numpy())
            rho_1 = torch.from_numpy(rho_1)
            rho_2 = self._batch_mn_pdf(s.cpu().numpy(), self.p_u_s.cpu().numpy(), self.p_sigma_s.cpu().numpy())
            rho_2 = torch.from_numpy(rho_2)
            if self.config['discrete_action']:
                rho_3 = self.svm.predict_proba(s.cpu().numpy())[a.cpu().numpy()]
                rho_3 = torch.from_numpy(rho_3)
            else:
                s_a = torch.cat([s, a.float()], dim=-1)
                rho_3 = self._batch_mn_pdf(
                    s_a.cpu().numpy(),
                    self.e_u_s_a.cpu().numpy(),
                    self.e_sigma_s_a.cpu().numpy()
                )
                rho_3 = torch.from_numpy(rho_3)
                rho_3 = rho_3 / rho_1
            alpha = self.config['alpha']
            beta = 1 - alpha
            den = rho_1 * rho_3
            frac = alpha * rho_1 + beta * rho_2
            if frac.abs().max() < 1e-4:
                for item in data:
                    item['reward'].zero_()
            else:
                reward = den / frac
                reward = torch.chunk(reward, reward.shape[0], dim=0)
                for item, rew in zip(data, reward):
                    item['reward'] = rew

    def collect_data(self, item: list):
        self.train_data.extend(item)

    def clear_data(self):
        self.train_data.clear()
