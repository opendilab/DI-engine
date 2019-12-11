import torch
import numbers


def to_device(item, device):
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list) or isinstance(item, tuple):
        return [to_device(t, device) for t in item]
    elif isinstance(item, dict):
        return {k: to_device(item[k], device) for k in item.keys()}
    elif isinstance(item, numbers.Integral) or isinstance(item, numbers.Real):
        return item
    elif item is None:
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))
