from easydict import EasyDict
import jax
import jax.numpy as jnp


def to_raw(x):
    if isinstance(x, dict):
        return {k: to_raw(v) for k, v in x.items()}
    elif x.shape == (1, ) or x.shape == ():
        return x.item()
    elif len(x.shape) == 1:
        return x.tolist()
    else:
        raise RuntimeError(x)


def collate_fn_jax(data: list):
    data = {k: jnp.stack([d[k] for d in data]) for k in data[0].keys()}
    data = jax.tree_util.tree_map(lambda x: jnp.stack(x), data)
    data = EasyDict(data)
    for k in data:
        if len(data[k].shape) == 2 and data[k].shape[1] == 1:
            data[k] = data[k].squeeze(1)
    return data
