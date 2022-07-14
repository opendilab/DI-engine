from collections.abc import Sequence, Mapping
from typing import List, Dict, Union, Any

import torch
import treetensor.torch as ttorch
import re
from torch._six import string_classes
import collections.abc as container_abcs
from ding.compatibility import torch_ge_131

int_classes = int
np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def ttorch_collate(x, json=False):

    def inplace_fn(t):
        for k in t.keys():
            if isinstance(t[k], torch.Tensor):
                if len(t[k].shape) == 2 and t[k].shape[1] == 1:  # reshape (B, 1) -> (B)
                    t[k] = t[k].squeeze(-1)
            else:
                inplace_fn(t[k])

    x = ttorch.stack(x)
    inplace_fn(x)
    if json:
        x = x.json()
    return x


def default_collate(batch: Sequence,
                    cat_1dim: bool = True,
                    ignore_prefix: list = ['collate_ignore']) -> Union[torch.Tensor, Mapping, Sequence]:
    """
    Overview:
        Put each data field into a tensor with outer dimension batch size.
    Example:
        >>> # a list with B tensors shaped (m, n) -->> a tensor shaped (B, m, n)
        >>> a = [torch.zeros(2,3) for _ in range(4)]
        >>> default_collate(a).shape
        torch.Size([4, 2, 3])
        >>>
        >>> # a list with B lists, each list contains m elements -->> a list of m tensors, each with shape (B, )
        >>> a = [[0 for __ in range(3)] for _ in range(4)]
        >>> default_collate(a)
        [tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0])]
        >>>
        >>> # a list with B dicts, whose values are tensors shaped :math:`(m, n)` -->>
        >>> # a dict whose values are tensors with shape :math:`(B, m, n)`
        >>> a = [{i: torch.zeros(i,i+1) for i in range(2, 4)} for _ in range(4)]
        >>> print(a[0][2].shape, a[0][3].shape)
        torch.Size([2, 3]) torch.Size([3, 4])
        >>> b = default_collate(a)
        >>> print(b[2].shape, b[3].shape)
        torch.Size([4, 2, 3]) torch.Size([4, 3, 4])
    Arguments:
        - batch (:obj:`Sequence`): a data sequence, whose length is batch size, whose element is one piece of data
    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data field.\
            the return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].
    """
    elem = batch[0]

    elem_type = type(elem)
    if isinstance(batch, ttorch.Tensor):
        return batch.json()
    if isinstance(elem, torch.Tensor):
        out = None
        if torch_ge_131() and torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, directly concatenate into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        if elem.shape == (1, ) and cat_1dim:
            # reshape (B, 1) -> (B)
            return torch.cat(batch, 0, out=out)
            # return torch.stack(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif isinstance(elem, ttorch.Tensor):
        return ttorch_collate(batch, json=True)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch], cat_1dim=cat_1dim)
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
        ret = {}
        for key in elem:
            if any([key.startswith(t) for t in ignore_prefix]):
                ret[key] = [d[key] for d in batch]
            else:
                ret[key] = default_collate([d[key] for d in batch], cat_1dim=cat_1dim)
        return ret
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples, cat_1dim=cat_1dim) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples, cat_1dim=cat_1dim) for samples in transposed]

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

    def stack(data):
        if isinstance(data, container_abcs.Mapping):
            return {k: stack(data[k]) for k in data}
        elif isinstance(data, container_abcs.Sequence) and isinstance(data[0], torch.Tensor):
            return torch.stack(data)
        else:
            return data

    elem = batch[0]
    assert isinstance(elem, (container_abcs.Mapping, list)), type(elem)
    if isinstance(batch[0], list):  # new pipeline + treetensor
        prev_state = [[b[i].get('prev_state') for b in batch] for i in range(len(batch[0]))]
        batch_data = ttorch.stack([ttorch_collate(b) for b in batch])  # (B, T, *)
        del batch_data.prev_state
        batch_data = batch_data.transpose(1, 0)
        batch_data.prev_state = prev_state
    else:
        prev_state = [b.pop('prev_state') for b in batch]
        batch_data = default_collate(batch)  # -> {some_key: T lists}, each list is [B, some_dim]
        batch_data = stack(batch_data)  # -> {some_key: [T, B, some_dim]}
        transformed_prev_state = list(zip(*prev_state))
        batch_data['prev_state'] = transformed_prev_state
        # append back prev_state, avoiding multi batch share the same data bug
        for i in range(len(batch)):
            batch[i]['prev_state'] = prev_state[i]
    return batch_data


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


def default_decollate(
        batch: Union[torch.Tensor, Sequence, Mapping],
        ignore: List[str] = ['prev_state', 'prev_actor_state', 'prev_critic_state']
) -> List[Any]:
    """
    Overview:
        Drag out batch_size collated data's batch size to decollate it,
        which is the reverse operation of ``default_collate``.
    Arguments:
        - batch (:obj:`Union[torch.Tensor, Sequence, Mapping]`): can refer to the Returns of ``default_collate``
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
