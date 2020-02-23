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
    elif item is None or isinstance(item, str):
        return item
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_tensor(item, dtype):
    def transform(d):
        return torch.tensor(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_tensor(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        else:
            new_data = []
            for t in item:
                new_data.append(to_tensor(t, dtype))
            return new_data
    elif item is None:
        return 'none'  # for convenience in dataloader
    else:
        raise TypeError("not support item type: {}".format(type(item)))
