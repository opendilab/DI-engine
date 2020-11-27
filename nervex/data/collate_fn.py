from collections.abc import Sequence, Mapping
from numbers import Integral
from typing import List, Dict, Union, Any

import torch
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch: Sequence) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Put each data field into a tensor with outer dimension batch size.
        Examples: (list different ``batch``s and their corresponding collated result)

            - a list with B tensors with shape :math:`(m, n)` -->> a tensor with shape :math:`(B, m, n)`
            - a list with B lists, each list contains m elements -->> a list of m tensors, each with shape :math:`(B)`
            - a list with B dicts, whose values are tensors with shape :math:`(m, n)` -->> a dict, \
                whose values are tensors with shape :math:`(B, m, n)`
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, directly concatenate into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if elem.shape == (1, ):
            # reshape (B, 1) -> (B)
            return torch.cat(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        dtype = torch.bool if isinstance(elem, bool) else torch.int64
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def timestep_collate(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, list]]:
    """
    Overview:
        Put each timestepped data field into a tensor with outer dimension batch size using ``default_collate``.
        For short, this process can be represented by:
        [len=B, ele={dict_key: [len=T, ele=Tensor(any_dims)]}] -> {dict_key: Tensor([T, B, any_dims])}
    Arguments:
        - batch (:obj:`List[Dict[str, Any]]`): a list of dicts with length B, each element is {some_key: some_seq} \
            ('prev_state' should be a key in the dict); \
            some_seq is a sequence with length T, each element is a torch.Tensor with any shape.
    Returns:
        - ret (:obj:`Dict[str, Union[torch.Tensor, list]]`): the collated data, with timestep and batch size \
            into each data field. By using ``default_collate``, timestep would come to the first dim. \
            So the final shape is :math:`(T, B, dim1, dim2, ...)`
    """
    elem = batch[0]
    assert isinstance(elem, container_abcs.Mapping), type(elem)
    prev_state = [b.pop('prev_state') for b in batch]
    batch = default_collate(batch)  # -> {some_key: T lists}, each list is [B, some_dim]
    for k in batch:
        if isinstance(batch[k], container_abcs.Sequence) and isinstance(batch[k][0], torch.Tensor):
            batch[k] = torch.stack(batch[k])  # -> {some_key: [T, B, some_dim]}
    batch['prev_state'] = list(zip(*prev_state))
    return batch


def diff_shape_collate(batch: Sequence) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Similar to ``default_collate``, put each data field into a tensor with outer dimension batch size.
        The main difference is that, ``diff_shape_collate`` allows tensors in the batch have `None`,
        which is quite common StarCraft observation.
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]
    elem_type = type(elem)
    if any([isinstance(elem, type(None)) for elem in batch]):
        return batch
    elif isinstance(elem, torch.Tensor):
        shapes = [e.shape for e in batch]
        if len(set(shapes)) != 1:
            return batch
        else:
            return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            return diff_shape_collate([torch.as_tensor(b) for b in batch])  # todo
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        dtype = torch.bool if isinstance(elem, bool) else torch.int64
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, Mapping):
        return {key: diff_shape_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(diff_shape_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [diff_shape_collate(samples) for samples in transposed]

    raise TypeError('not support element type: {}'.format(elem_type))


def default_decollate(batch: Union[torch.Tensor, Sequence, Mapping], ignore: List[str] = ['prev_state']) -> List[Any]:
    """
    Overview:
        Drag out batch_size collated data's batch size to decollate it,
        which is the reverse operation of ``default_collate``.
    Arguments:
        - batch (:obj:`Union[torch.Tensor, Sequence, Mapping]`): can reference the Returns of ``default_collate``
        - ignore(:obj:`List[str]`): a list of names to be ignored, only function if input ``batch`` is a dict. \
            If key is in this list, its value would stay the same with no decollation.
    Returns:
        - ret (:obj:`List[Any]`): a list with B elements.
    """
    if isinstance(batch, torch.Tensor):
        batch = torch.split(batch, 1, dim=0)
        # squeeze if original batch's shape is like (B, dim1, dim2, ...);
        # otherwise directly return the list.
        if len(batch[0].shape) > 1:
            batch = [elem.squeeze(0) for elem in batch]
        return list(batch)
    elif isinstance(batch, Sequence):
        return list(zip(*[default_decollate(e) for e in batch]))
    elif isinstance(batch, Mapping):
        tmp = {k: v if k in ignore else default_decollate(v) for k, v in batch.items()}
        B = len(list(tmp.values())[0])
        return [{k: tmp[k][i] for k in tmp.keys()} for i in range(B)]

    raise TypeError("not support batch type: {}".format(type(batch)))
