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
