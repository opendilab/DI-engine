from .offline import ReplayDataset
from .tests.fake_dataset import FakeReplayDataset


def build_dataset(cfg, train_dataset):
    if cfg.dataset_type == "fake":
        return FakeReplayDataset(cfg)
    return ReplayDataset(cfg, train_mode=train_dataset)
