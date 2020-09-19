from collections.abc import Sequence, Mapping
from numbers import Integral
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from nervex.utils import list_dict2dict_list


def diff_shape_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if any([isinstance(elem, type(None)) for elem in batch]):
        return batch
    elif isinstance(elem, torch.Tensor):
        shapes = [e.shape for e in batch]
        if len(set(shapes)) != 1:
            return batch
        else:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            return diff_shape_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, Integral):
        return batch
    elif isinstance(elem, Mapping):
        return {key: diff_shape_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(diff_shape_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [diff_shape_collate(samples) for samples in transposed]

    raise TypeError('not support element type: {}'.format(elem_type))


def sumo_dqn_collate_fn(data):
    batchs = data

    # print("data of collect:", data)

    obs_batch = torch.cat([x['obs'].unsqueeze(0) for x in batchs], 0)
    nextobs_batch = torch.cat([x['next_obs'].unsqueeze(0) for x in batchs], 0)
    action_batch = torch.cat([torch.LongTensor([x['action']]) for x in batchs])
    reward_batch = default_collate([x['reward'] for x in batchs])
    reward_batch = {k: v.squeeze(1) for k, v in reward_batch.items()}
    done_batch = torch.cat([torch.Tensor([x['done']]) for x in batchs])

    reward = reward_batch
    action = action_batch
    action = list(zip(*action))
    action = [torch.stack(t) for t in action]
    done = done_batch

    # print("state_batch = ", state_batch)
    # print("state_batch_shape = ", state_batch.shape)

    return {'obs': obs_batch, 'next_obs': nextobs_batch, 'action': action, 'reward': reward, 'done': done}
