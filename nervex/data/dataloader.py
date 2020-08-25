from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .online import OnlineIteratorDataLoader
from .sampler import DistributedSampler


def build_dataloader(
    dataset, dataloader_type, batch_size, use_distributed, read_data_fn=None, collate_fn=default_collate
):
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
            collate_fn=collate_fn
        )
    elif dataloader_type == 'online':
        dataloader = OnlineIteratorDataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, read_data_fn=read_data_fn
        )
    else:
        raise NotImplementedError("invalid dataloader_type: {}".format(dataloader_type))
    return dataloader
