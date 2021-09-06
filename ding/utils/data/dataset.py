from typing import List, Dict
import pickle
import torch
import numpy as np

from easydict import EasyDict
from torch.utils.data import Dataset

from ding.utils import DATASET_REGISTRY, import_module


@DATASET_REGISTRY.register('naive')
class NaiveRLDataset(Dataset):

    def __init__(self, cfg) -> None:
        assert type(cfg) in [str, EasyDict], "invalid cfg type: {}".format(type(cfg))
        if isinstance(cfg, EasyDict):
            self._data_path = cfg.policy.learn.data_path
        elif isinstance(cfg, str):
            self._data_path = cfg
        with open(self._data_path, 'rb') as f:
            self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


@DATASET_REGISTRY.register('d4rl')
class D4RLDataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        import gym
        import logging
        try:
            import d4rl  # register d4rl enviroments with open ai gym
        except ImportError:
            logging.warning("not found d4rl env, please install it, refer to https://github.com/rail-berkeley/d4rl")

        # Init parameters
        data_path = cfg.policy.learn.get('data_path', None)
        env_id = cfg.env.env_id

        # Create the environment
        if data_path:
            d4rl.set_dataset_path(data_path)
        env = gym.make(env_id)
        dataset = d4rl.qlearning_dataset(env)
        self._data = []
        self._load_d4rl(dataset)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _load_d4rl(self, dataset: Dict[str, np.ndarray]) -> None:
        for i in range(len(dataset['observations'])):
            trans_data = {}
            trans_data['obs'] = torch.from_numpy(dataset['observations'][i])
            trans_data['next_obs'] = torch.from_numpy(dataset['next_observations'][i])
            trans_data['action'] = torch.from_numpy(dataset['actions'][i])
            trans_data['reward'] = torch.tensor(dataset['rewards'][i])
            trans_data['done'] = dataset['terminals'][i]
            trans_data['collect_iter'] = 0
            self._data.append(trans_data)


def create_dataset(cfg, **kwargs) -> Dataset:
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return DATASET_REGISTRY.build(cfg.policy.learn.data_type, cfg=cfg, **kwargs)
