from .offline import ReplayDataset
from .fake_dataset import FakeReplayDataset, FakeActorDataset


def build_dataset(dataset_config, train_dataset):
    if dataset_config.dataset_type == "fake":
        return FakeReplayDataset(dataset_config)
    elif dataset_config.dataset_type == "fake_actor":
        return FakeActorDataset(dataset_config)
    return ReplayDataset(dataset_config, train_mode=train_dataset)
