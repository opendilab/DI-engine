from .offline import ReplayDataset, ReplayEvalDataset


def build_dataset(cfg):
    dataset_type = cfg.dataset_type
    assert(dataset_type in ['replay', 'replay_eval'])
    if dataset_type == 'replay':
        return ReplayDataset(cfg)
    elif dataset_type == 'replay_eval':
        return ReplayEvalDataset(cfg)
    else:
        raise KeyError("invalid dataset_type: {}".format(dataset_type))
