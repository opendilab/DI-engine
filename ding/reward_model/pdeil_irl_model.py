import numpy as np
import torch
import pickle
from typing import List, Dict
import scipy.stats as stats
try:
    from sklearn.svm import SVC
except ImportError:
    SVC = None
from ding.torch_utils import cov
from ding.utils import REWARD_MODEL_REGISTRY, one_time_warning
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('pdeil')
class PdeilRewardModel(BaseRewardModel):
    """
    Overview:
        The Pdeil reward model class
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``, ``_batch_mn_pdf``
    """
    config = dict(
        type='pdeil',
        # expert_data_path='expert_data.pkl',
        discrete_action=False,
        alpha=0.5,
    )

    def __init__(self, cfg: dict, device, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
            Some rules in naming the attributes of ``self.``:

                - ``e_`` : expert values
                - ``_sigma_`` : standard division values
                - ``p_`` : current policy values
                - ``_s_`` : states
                - ``_a_`` : actions
        Arguments:
            - cfg (:obj:`Dict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`str`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(PdeilRewardModel, self).__init__()
        self.cfg: dict = cfg
        self.e_u_s = None
        self.e_sigma_s = None
        if cfg.discrete_action:
            self.svm = None
        else:
            self.e_u_s_a = None
            self.e_sigma_s_a = None
        self.p_u_s = None
        self.p_sigma_s = None
        self.expert_data = None
        self.train_data: list = []
        assert device in ["cpu", "cuda"] or "cuda" in device
        # pedil default use cpu device
        self.device = 'cpu'

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
        if self.cfg.discrete_action and SVC is None:
            one_time_warning("You are using discrete action while the SVC is not installed!")
        if self.cfg.discrete_action and SVC is not None:
            self.svm: SVC = SVC(probability=True)
            self.svm.fit(states.cpu().numpy(), actions.cpu().numpy())
        else:
            # states action conjuct
            state_actions = torch.cat((states, actions.float()), dim=-1)
            self.e_u_s_a = torch.mean(state_actions, axis=0)
            self.e_sigma_s_a = cov(state_actions, rowvar=False)

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config['expert_data_path']`` attribute in self.
        Effects:
            This is a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``)
        """
        expert_data_path: str = self.cfg.expert_data_path
        with open(expert_data_path, 'rb') as f:
            self.expert_data: list = pickle.load(f)

    def _train(self, states: torch.Tensor) -> None:
        """
        Overview:
            Helper function for ``train`` which caclulates loss for train data and expert data.
        Arguments:
            - states (:obj:`torch.Tensor`): current policy states
        Effects:
            - Update attributes of ``p_u_s`` and ``p_sigma_s``
        """
        # we only need to collect the current policy state
        self.p_u_s = torch.mean(states, axis=0)
        self.p_sigma_s = cov(states, rowvar=False)

    def train(self):
        """
        Overview:
            Training the Pdeil reward model.
        """
        states = torch.stack([item['obs'] for item in self.train_data], dim=0)
        self._train(states)

    def _batch_mn_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Overview:
           Get multivariate normal pdf of given np array.
        """
        return np.asarray(stats.multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=False), dtype=np.float32)

    def estimate(self, data: list) -> List[Dict]:
        """
        Overview:
            Estimate reward by rewriting the reward keys.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation,\
                 with at least ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)
        s = torch.stack([item['obs'] for item in train_data_augmented], dim=0)
        a = torch.stack([item['action'] for item in train_data_augmented], dim=0)
        if self.p_u_s is None:
            print("you need to train you reward model first")
            for item in train_data_augmented:
                item['reward'].zero_()
        else:
            rho_1 = self._batch_mn_pdf(s.cpu().numpy(), self.e_u_s.cpu().numpy(), self.e_sigma_s.cpu().numpy())
            rho_1 = torch.from_numpy(rho_1)
            rho_2 = self._batch_mn_pdf(s.cpu().numpy(), self.p_u_s.cpu().numpy(), self.p_sigma_s.cpu().numpy())
            rho_2 = torch.from_numpy(rho_2)
            if self.cfg.discrete_action:
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
            alpha = self.cfg.alpha
            beta = 1 - alpha
            den = rho_1 * rho_3
            frac = alpha * rho_1 + beta * rho_2
            if frac.abs().max() < 1e-4:
                for item in train_data_augmented:
                    item['reward'].zero_()
            else:
                reward = den / frac
                reward = torch.chunk(reward, reward.shape[0], dim=0)
                for item, rew in zip(train_data_augmented, reward):
                    item['reward'] = rew
        return train_data_augmented

    def collect_data(self, item: list):
        """
        Overview:
            Collecting training data by iterating data items in the input list
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self`` by \
                iterating data items in the input data items' list
        """
        self.train_data.extend(item)

    def clear_data(self):
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.train_data.clear()
