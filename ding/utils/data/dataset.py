from typing import List, Dict
import pickle
import torch
import numpy as np
import random
from ditk import logging

from easydict import EasyDict
from torch.utils.data import Dataset

from ding.utils import DATASET_REGISTRY, import_module
from ding.rl_utils import discount_cumsum


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
        if cfg.policy.collect.get('normalize_states', None):
            dataset = self._normalize_states(dataset)
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
            self._data.append(trans_data)

    def _normalize_states(self, dataset, eps=1e-3):
        self._mean = dataset['observations'].mean(0, keepdims=True)
        self._std = dataset['observations'].std(0, keepdims=True) + eps
        dataset['observations'] = (dataset['observations'] - self._mean) / self._std
        dataset['next_observations'] = (dataset['next_observations'] - self._mean) / self._std
        return dataset

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


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


@DATASET_REGISTRY.register('d4rl_trajectory')
class D4RLTrajectoryDataset(Dataset):

    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        if isinstance(self.trajectories[0], list):
            # for our collected dataset, e.g. cartpole/lunarlander case
            self.trajectories_tmp = {}
            self.trajectories_tmp = [
                {
                    'observations': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['obs']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    'next_observations': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['next_obs']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    'actions': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['action']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    'rewards': np.stack(
                        [
                            self.trajectories[eps_index][transition_index]['reward']
                            for transition_index in range(len(self.trajectories[eps_index]))
                        ],
                        axis=0
                    ),
                    # 'dones':
                    #     np.stack([
                    #     int(self.trajectories[eps_index][transition_index]['done']) for transition_index in range(len(self.trajectories[eps_index]))
                    # ], axis=0)
                } for eps_index in range(len(self.trajectories))
            ]

            self.trajectories = self.trajectories_tmp

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si:si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si:si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si:si + self.context_len])
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat(
                [states, torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)], dim=0
            )

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat(
                [actions, torch.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype)], dim=0
            )

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(([padding_len] + list(returns_to_go.shape[1:])), dtype=returns_to_go.dtype)
                ],
                dim=0
            )

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [torch.ones(traj_len, dtype=torch.long),
                 torch.zeros(padding_len, dtype=torch.long)], dim=0
            )

        return timesteps, states, actions, returns_to_go, traj_mask


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
