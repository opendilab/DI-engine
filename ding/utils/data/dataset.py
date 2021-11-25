from typing import List, Dict
import pickle
import torch
import numpy as np
import logging

from easydict import EasyDict
from torch.utils.data import Dataset

from ding.utils import DATASET_REGISTRY, import_module


@DATASET_REGISTRY.register('naive')
class NaiveRLDataset(Dataset):

    def __init__(self, cfg) -> None:
        assert type(cfg) in [str, EasyDict], "invalid cfg type: {}".format(type(cfg))
        if isinstance(cfg, EasyDict):
            self._data_path = cfg.policy.collect.data_path
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
        data_path = cfg.policy.collect.get('data_path', None)
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


@DATASET_REGISTRY.register('hdf5')
class HDF5Dataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        try:
            import h5py
        except ImportError:
            logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
        data_path = cfg.policy.collect.get('data_path', None)
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        if cfg.policy.collect.get('normalize_states', None):
            self._normalize_states()

    def __len__(self) -> int:
        return len(self._data['obs'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    def _load_data(self, dataset: Dict[str, np.ndarray]) -> None:
        self._data = {}
        for k in dataset.keys():
            logging.info(f'Load {k} data.')
            self._data[k] = dataset[k][:]

    def _normalize_states(self, eps=1e-3):
        self._mean = self._data['obs'].mean(0, keepdims=True)
        self._std = self._data['obs'].std(0, keepdims=True) + eps
        self._data['obs'] = (self._data['obs'] - self._mean) / self._std
        self._data['next_obs'] = (self._data['next_obs'] - self._mean) / self._std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


def hdf5_save(exp_data, expert_data_path):
    try:
        import h5py
    except ImportError:
        logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
    import numpy as np
    dataset = dataset = h5py.File('%s_demos.hdf5' % expert_data_path.replace('.pkl', ''), 'w')
    dataset.create_dataset('obs', data=np.array([d['obs'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('action', data=np.array([d['action'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('reward', data=np.array([d['reward'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('done', data=np.array([d['done'] for d in exp_data]), compression='gzip')
    dataset.create_dataset('collect_iter', data=np.array([d['collect_iter'] for d in exp_data]), compression='gzip')
    dataset.create_dataset('next_obs', data=np.array([d['next_obs'].numpy() for d in exp_data]), compression='gzip')


def naive_save(exp_data, expert_data_path):
    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)


def offline_data_save_type(exp_data, expert_data_path, data_type='naive'):
    globals()[data_type + '_save'](exp_data, expert_data_path)


def create_dataset(cfg, **kwargs) -> Dataset:
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return DATASET_REGISTRY.build(cfg.policy.collect.data_type, cfg=cfg, **kwargs)
