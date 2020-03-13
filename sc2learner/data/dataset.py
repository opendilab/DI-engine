from .offline import ReplayDataset, ReplayEvalDataset
from .tests.fake_dataset import FakeReplayDataset


def build_dataset(cfg):
    dataset_type = cfg.dataset_type
    assert(dataset_type in ['replay', 'replay_eval', 'fake'])
    if dataset_type == 'replay':
        return ReplayDataset(cfg)
    elif dataset_type == 'replay_eval':
        return ReplayEvalDataset(cfg)
    elif dataset_type == 'fake':
        return FakeReplayDataset(cfg)
    else:
        raise KeyError("invalid dataset_type: {}".format(dataset_type))
