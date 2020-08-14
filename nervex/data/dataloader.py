from torch.utils.data import DataLoader

from .offline import ReplayIterationDataLoader
from .collate_fn import policy_collate_fn, actor_collate_fn
from .online import OnlineIteratorDataLoader
from .sampler import DistributedSampler


def build_dataloader(dataset, dataloader_type, batch_size, use_distributed, read_data_fn=None):
    dataloader_type = dataloader_type
    if dataloader_type == 'epoch':
        sampler = DistributedSampler(dataset, round_up=False) if use_distributed else None
        shuffle = False if use_distributed else True
        # set num_workers=0 for preventing ceph reading file bug
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=0,
            sampler=sampler,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=policy_collate_fn
        )
    elif dataloader_type == 'iter':
        # FIXME ReplayIterationDataLoader is not tested locally at all!
        # assert use_distributed
        dataloader = ReplayIterationDataLoader(dataset, batch_size, collate_fn=policy_collate_fn)
    elif dataloader_type == 'online':
        dataloader = OnlineIteratorDataLoader(
            dataset, batch_size=batch_size, collate_fn=actor_collate_fn, read_data_fn=read_data_fn
        )
    elif dataloader_type == 'fake':
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=policy_collate_fn, num_workers=0)
    elif dataloader_type == 'fake_actor':
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=actor_collate_fn, num_workers=0)
    else:
        raise NotImplementedError()
    return dataloader
