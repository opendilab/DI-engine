from .offline import ReplayDataset
from .tests.fake_dataset import FakeReplayDataset


def build_dataset(dataset_config, train_dataset):
    if dataset_config.dataset_type == "fake":
        return FakeReplayDataset(dataset_config)
    return ReplayDataset(dataset_config, train_mode=train_dataset)
