import numpy as np
import torch
import pickle
import scipy.stats as stats
from ding.utils import REWARD_MODEL_REGISTRY, one_time_warning
from .base_reward_model import BaseRewardModel


@REWARD_MODEL_REGISTRY.register('countbased')
class CountBasedRewardModel(BaseRewardModel):
    """
    Overview:
        The Pdeil reward model class
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``, ``_batch_mn_pdf``
    """
    config = dict(
        type='countbased',
        counter_type='SimHash',
        bonus_coefficent=0.5,
        state_dim=2,
        hash_dim=32,
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
        super(CountBasedRewardModel, self).__init__()
        self.cfg: dict = cfg
        self._beta = cfg.bonus_coefficent
        self._counter_type = cfg.counter_type
        self.device = device
        self.tb_logger = tb_logger
        assert self._counter_type in ['SimHash', 'AutoEncoder']
        if self._counter_type == 'SimHash':
            print(cfg)
            self._counter = SimHash(cfg.state_dim, cfg.hash_dim, self.device)
        elif self._counter_type == 'AutoEncoder':
            self._counter = AutoEncoder(cfg.state_dim,cfg.hash_dim,self.device)

    def train(self):
        """
        Overview:
            Training the Pdeil reward model.
        """
        pass

    def estimate(self, data: list) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward keys.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation,\
                 with at least ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        if self._counter_type == 'SimHash':
            s = torch.stack([item['obs'] for item in data], dim=0).to(self.device)
            hash_cnts = self._counter.update(s)
            for item, cnt in zip(data, hash_cnts):
                item['reward'] = item['reward']+self._beta/np.sqrt(cnt)
        

    def collect_data(self, data: list):
        """
        Overview:
            Collecting training data by iterating data items in the input list
        Arguments:
            - data (:obj:`list`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self`` by \
                iterating data items in the input data items' list
        """
        pass
        # if self._counter_type == 'SimHash':
        #     s = torch.stack([item['obs'] for item in data], dim=0).to(self.device)
        #     hash_cnts = self._counter.update(s)
        #     for item, cnt in zip(data, hash_cnts):
        #         item['reward'] = item['reward']+self._beta/np.sqrt(cnt)

    def clear_data(self):
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        pass


class SimHash():
    def __init__(self, state_emb, k, device):
        self.hash = {}
        self.fc = torch.nn.Linear(state_emb, k, bias=False).to(device)
        self.fc.weight.data = torch.randn_like(self.fc.weight)
        self.device = device

    def update(self, states):
        counts = []
        keys = np.sign(self.fc(states).to('cpu').detach().numpy()).tolist()
        for idx,key in enumerate(keys):
            key = str(key)
            keys[idx]=key
            self.hash[key] = self.hash.get(key, 0)+1
        for key in keys:
            counts.append(self.hash[key])
        return counts

class AutoEncoder():
    def __init__(self,state_dim,k,device):
        pass
    
