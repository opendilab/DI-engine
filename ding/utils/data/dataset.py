from typing import List, Dict
import pickle
import torch
from torch.utils.data import Dataset


class NaiveRLDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        with open(self._data_path, 'rb') as f:
            self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

class OfflineRLDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        self._data = torch.load(self._data_path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


class D4RLDataset(Dataset):

    def __init__(self, env_id: str, device: str) -> None:
        import warnings
        import gym
        try:
            import d4rl  # register d4rl enviroments with open ai gym
        except ImportError:
            warnings.warn("not found d4rl env, please install it, refer to https://github.com/rail-berkeley/d4rl")

        # Create the environment
        self._device = device
        env = gym.make(env_id)
        if 'random-expert' in env_id:
            dataset = d4rl.basic_dataset(env)
        else:
            dataset = d4rl.qlearning_dataset(env)
        self._data = []
        self._load_d4rl(dataset)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _load_d4rl(self, dataset):
        for i in range(len(dataset['observations'])):
            # if i >1000:
            #     break
            trans_data={}
            # import ipdb;ipdb.set_trace()
            trans_data['obs'] = torch.from_numpy(dataset['observations'][i]).to(self._device)
            trans_data['next_obs'] = torch.from_numpy(dataset['next_observations'][i]).to(self._device)
            trans_data['action'] = torch.from_numpy(dataset['actions'][i]).to(self._device)
            trans_data['reward'] = torch.tensor(dataset['rewards'][i]).to(self._device)
            trans_data['done'] = dataset['terminals'][i]
            trans_data['collect_iter'] = 0
            self._data.append(trans_data)