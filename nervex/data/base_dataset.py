import torch
from torch.utils.data import Dataset

META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'


class BaseDataset(Dataset):
    def __init__(self, cfg):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()

    def step(self, index=None):
        """We can assume everydataset has step function."""
        pass

    def reset_step(self, index=None):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError()
