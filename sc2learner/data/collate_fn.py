from collections.abc import Sequence, Mapping
from numbers import Integral
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from pysc2.lib.static_data import ACTIONS_REORDER
from sc2learner.utils import list_dict2dict_list


def policy_collate_fn(batch, max_delay=63, action_type_transform=True):
    data_item = {
        'spatial_info': False,  # special op
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'actions': False,
        'map_size': False,
        'start_step': False
    }

    def merge_func(data):
        valid_data = [t for t in data if t is not None]
        new_data = list_dict2dict_list(valid_data)
        for k, merge in data_item.items():
            if merge:
                new_data[k] = default_collate(new_data[k])
            if k == 'spatial_info':
                shape = [t.shape for t in new_data[k]]
                if len(set(shape)) != 1:
                    tmp_shape = list(zip(*shape))
                    H, W = max(tmp_shape[1]), max(tmp_shape[2])
                    new_spatial_info = []
                    for item in new_data[k]:
                        h, w = item.shape[-2:]
                        new_spatial_info.append(F.pad(item, [0, W - w, 0, H - h], "constant", 0))
                    new_data[k] = default_collate(new_spatial_info)
                else:
                    new_data[k] = default_collate(new_data[k])
            if k == 'actions':
                new_data[k] = list_dict2dict_list(new_data[k])
                new_data[k]['delay'] = [torch.clamp(x, 0, max_delay) for x in new_data[k]['delay']]  # clip
                if action_type_transform:
                    action_type = [t.item() for t in new_data[k]['action_type']]
                    L = len(action_type)
                    for i in range(L):
                        action_type[i] = ACTIONS_REORDER[action_type[i]]
                    action_type = torch.LongTensor(action_type)
                    new_data[k]['action_type'] = list(torch.chunk(action_type, L, dim=0))
        new_data['end_index'] = [idx for idx, t in enumerate(data) if t is None]
        return new_data

    # sequence, batch
    b_len = [len(b) for b in batch]
    max_len = max(b_len)
    min_len = min(b_len)
    if max_len != min_len:
        seq = []
        for i in range(max_len):
            tmp = []
            for j in range(len(batch)):
                if i >= b_len[j]:
                    tmp.append(None)
                else:
                    tmp.append(batch[j][i])
            seq.append(tmp)

    seq = list(zip(*batch))
    for s in range(len(seq)):
        seq[s] = merge_func(seq[s])
    return seq


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


def actor_collate_fn(batch, max_delay=63, action_type_transform=True):
    data_item = {
        'prev_state': False,  # special op
        'spatial_info': False,  # special op
        'scalar_info': True,
        'entity_info': False,
        'entity_raw': False,
        'map_size': False,
        'actions': True,
        'teacher_actions': True,
        'rewards': True,
        'baselines': True,
        'game_seconds': True,
        'target_outputs': True,
        'teacher_outputs': True,
        'behaviour_outputs': True,
        'score_cumulative': True,
    }
    data_keys = ['home', 'away', 'home_next', 'away_next']

    def merge_func(data):
        new_data = list_dict2dict_list(data)
        for k, merge in data_item.items():
            if k not in new_data.keys():
                continue
            if merge:
                new_data[k] = diff_shape_collate(new_data[k])
            if k == 'spatial_info':
                shape = [t.shape for t in new_data[k]]
                if len(set(shape)) != 1:
                    tmp_shape = list(zip(*shape))
                    H, W = max(tmp_shape[1]), max(tmp_shape[2])
                    new_spatial_info = []
                    for item in new_data[k]:
                        h, w = item.shape[-2:]
                        new_spatial_info.append(F.pad(item, [0, W - w, 0, H - h], "constant", 0))
                    new_data[k] = diff_shape_collate(new_spatial_info)
                else:
                    new_data[k] = diff_shape_collate(new_data[k])
        return new_data

    seq = list(zip(*batch))
    for s in range(len(seq)):
        step_data = list_dict2dict_list(seq[s])
        tmp = {}
        for k in step_data.keys():
            if k in data_keys:
                tmp[k] = merge_func(step_data[k])
            else:
                tmp[k] = step_data[k]
        seq[s] = tmp
    return seq
