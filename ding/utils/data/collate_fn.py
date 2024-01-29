from collections.abc import Sequence, Mapping
from typing import List, Dict, Union, Any

import torch
import treetensor.torch as ttorch
import re
import collections.abc as container_abcs
from ding.compatibility import torch_ge_131

int_classes = int
string_classes = (str, bytes)
np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def ttorch_collate(x, json: bool = False, cat_1dim: bool = True):
    """
    Overview:
        Collates a list of tensors or nested dictionaries of tensors into a single tensor or nested \
            dictionary of tensors.

    Arguments:
        - x : The input list of tensors or nested dictionaries of tensors.
        - json (:obj:`bool`): If True, converts the output to JSON format. Defaults to False.
        - cat_1dim (:obj:`bool`): If True, concatenates tensors with shape (B, 1) along the last dimension. \
            Defaults to True.

    Returns:
        The collated output tensor or nested dictionary of tensors.

    Examples:
        >>> # case 1: Collate a list of tensors
        >>> tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
        >>> collated = ttorch_collate(tensors)
        collated = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> # case 2: Collate a nested dictionary of tensors
        >>> nested_dict = {
                'a': torch.tensor([1, 2, 3]),
                'b': torch.tensor([4, 5, 6]),
                'c': torch.tensor([7, 8, 9])
            }
        >>> collated = ttorch_collate(nested_dict)
        collated = {
            'a': torch.tensor([1, 2, 3]),
            'b': torch.tensor([4, 5, 6]),
            'c': torch.tensor([7, 8, 9])
        }
        >>> # case 3: Collate a list of nested dictionaries of tensors
        >>> nested_dicts = [
                {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])},
                {'a': torch.tensor([7, 8, 9]), 'b': torch.tensor([10, 11, 12])}
            ]
        >>> collated = ttorch_collate(nested_dicts)
        collated = {
            'a': torch.tensor([[1, 2, 3], [7, 8, 9]]),
            'b': torch.tensor([[4, 5, 6], [10, 11, 12]])
        }
    """

    def inplace_fn(t):
        for k in t.keys():
            if isinstance(t[k], torch.Tensor):
                if len(t[k].shape) == 2 and t[k].shape[1] == 1:  # reshape (B, 1) -> (B)
                    t[k] = t[k].squeeze(-1)
            else:
                inplace_fn(t[k])

    x = ttorch.stack(x)
    if cat_1dim:
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

    Arguments:
        - batch (:obj:`Sequence`): A data sequence, whose length is batch size, whose element is one piece of data.
        - cat_1dim (:obj:`bool`): Whether to concatenate tensors with shape (B, 1) to (B), defaults to True.
        - ignore_prefix (:obj:`list`): A list of prefixes to ignore when collating dictionaries, \
            defaults to ['collate_ignore'].

    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): the collated data, with batch size into each data \
            field. The return dtype depends on the original element dtype, can be [torch.Tensor, Mapping, Sequence].

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
    """

    if isinstance(batch, ttorch.Tensor):
        return batch.json()

    elem = batch[0]
    elem_type = type(elem)
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
        return ttorch_collate(batch, json=True, cat_1dim=cat_1dim)
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
        Collates a batch of timestepped data fields into tensors with the outer dimension being the batch size. \
        Each timestepped data field is represented as a tensor with shape [T, B, any_dims], where T is the length \
        of the sequence, B is the batch size, and any_dims represents the shape of the tensor at each timestep.

    Arguments:
        - batch(:obj:`List[Dict[str, Any]]`): A list of dictionaries with length B, where each dictionary represents \
            a timestepped data field. Each dictionary contains a key-value pair, where the key is the name of the \
            data field and the value is a sequence of torch.Tensor objects with any shape.

    Returns:
        - ret(:obj:`Dict[str, Union[torch.Tensor, list]]`): The collated data, with the timestep and batch size \
            incorporated into each data field. The shape of each data field is [T, B, dim1, dim2, ...].

    Examples:
        >>> batch = [
                {'data0': [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]},
                {'data1': [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]}
            ]
        >>> collated_data = timestep_collate(batch)
        >>> print(collated_data['data'].shape)
        torch.Size([2, 2, 3])
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
        Collates a batch of data with different shapes.
        This function is similar to `default_collate`, but it allows tensors in the batch to have `None` values, \
        which is common in StarCraft observations.

    Arguments:
        - batch (:obj:`Sequence`): A sequence of data, where each element is a piece of data.

    Returns:
        - ret (:obj:`Union[torch.Tensor, Mapping, Sequence]`): The collated data, with the batch size applied \
            to each data field. The return type depends on the original element type and can be a torch.Tensor, \
            Mapping, or Sequence.

    Examples:
        >>> # a list with B tensors shaped (m, n) -->> a tensor shaped (B, m, n)
        >>> a = [torch.zeros(2,3) for _ in range(4)]
        >>> diff_shape_collate(a).shape
        torch.Size([4, 2, 3])
        >>>
        >>> # a list with B lists, each list contains m elements -->> a list of m tensors, each with shape (B, )
        >>> a = [[0 for __ in range(3)] for _ in range(4)]
        >>> diff_shape_collate(a)
        [tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0]), tensor([0, 0, 0, 0])]
        >>>
        >>> # a list with B dicts, whose values are tensors shaped :math:`(m, n)` -->>
        >>> # a dict whose values are tensors with shape :math:`(B, m, n)`
        >>> a = [{i: torch.zeros(i,i+1) for i in range(2, 4)} for _ in range(4)]
        >>> print(a[0][2].shape, a[0][3].shape)
        torch.Size([2, 3]) torch.Size([3, 4])
        >>> b = diff_shape_collate(a)
        >>> print(b[2].shape, b[3].shape)
        torch.Size([4, 2, 3]) torch.Size([4, 3, 4])
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
        Drag out batch_size collated data's batch size to decollate it, which is the reverse operation of \
        ``default_collate``.

    Arguments:
        - batch (:obj:`Union[torch.Tensor, Sequence, Mapping]`): The collated data batch. It can be a tensor, \
            sequence, or mapping.
        - ignore(:obj:`List[str]`): A list of names to be ignored. Only applicable if the input ``batch`` is a \
            dictionary. If a key is in this list, its value will remain the same without decollation. Defaults to \
            ['prev_state', 'prev_actor_state', 'prev_critic_state'].

    Returns:
        - ret (:obj:`List[Any]`): A list with B elements, where B is the batch size.

    Examples:
        >>> batch = {
            'a': [
                [1, 2, 3],
                [4, 5, 6]
            ],
            'b': [
                [7, 8, 9],
                [10, 11, 12]
            ]}
        >>> default_decollate(batch)
        {
            0: {'a': [1, 2, 3], 'b': [7, 8, 9]},
            1: {'a': [4, 5, 6], 'b': [10, 11, 12]},
        }
    """
    if isinstance(batch, torch.Tensor):
        batch = torch.split(batch, 1, dim=0)
        # Squeeze if the original batch's shape is like (B, dim1, dim2, ...);
        # otherwise, directly return the list.
        if len(batch[0].shape) > 1:
            batch = [elem.squeeze(0) for elem in batch]
        return list(batch)
    elif isinstance(batch, Sequence):
        return list(zip(*[default_decollate(e) for e in batch]))
    elif isinstance(batch, Mapping):
        tmp = {k: v if k in ignore else default_decollate(v) for k, v in batch.items()}
        B = len(list(tmp.values())[0])
        return [{k: tmp[k][i] for k in tmp.keys()} for i in range(B)]
    elif isinstance(batch, torch.distributions.Distribution):  # For compatibility
        return [None for _ in range(batch.batch_shape[0])]

    raise TypeError("Not supported batch type: {}".format(type(batch)))
