from torch.utils.data import DataLoader
from .online_dataloader import OnlineDataLoader, unroll_split_collate_fn
from .replay_dataset import policy_collate_fn
from .sampler import DistributedSampler
from .replay_dataloader import ReplayIterationDataLoader


def build_dataloader(cfg, dataset):
    dataloader_type = cfg.dataloader_type
    assert(dataloader_type in ['epoch', 'iter'])
    if dataloader_type == 'epoch':
        sampler = DistributedSampler(dataset, round_up=False) if cfg.use_distributed else None
        shuffle = False if cfg.use_distributed else True
        # set num_workers=0 for preventing ceph reading file bug
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=False, num_workers=0,
                                sampler=sampler, shuffle=shuffle, drop_last=False, collate_fn=policy_collate_fn)
    elif dataloader_type == 'iter':
        assert(cfg.use_distributed)
        dataloader = ReplayIterationDataLoader(dataset, cfg.batch_size)
    return dataloader
