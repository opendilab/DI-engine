from torch.utils.data import _utils
import random


class OnlineDataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn=None):
        self.dataset = dataset
        if collate_fn is None:
            self.collate_fn = _utils.collate.default_collate
        else:
            self.collate_fn = collate_fn
        self.batch_size = batch_size

    def _get_indices(self):
        indices = random.sample([i for i in range(self.dataset.maxlen)], self.batch_size)
        return indices

    def __next__(self):
        indices = self._get_indices()
        batch, avg_usage, push_count, avg_model_index = self.dataset.get_indice_data(indices, self._cur_model_index)
        batch = self.collate_fn(batch)
        return batch, avg_usage, push_count, avg_model_index

    @property
    def cur_model_index(self, cur_model_index):
        self._cur_model_index = cur_model_index


def unroll_split_collate_fn(*args, collate_fn=_utils.collate.default_collate, **kwargs):
    result = collate_fn(*args, **kwargs)
    if isinstance(result, list) or isinstance(result, list):
        B0, B1 = result[0].shape[:2]  # first element result must be torch.Tensor
        new_result = []
        for item in result:
            other_shape = item.shape[2:]
            new_result.append(item.reshape(B0*B1, *other_shape))
    elif isinstance(result, dict):
        new_result = {}
        for k, v in result.items():
            if isinstance(v, list) and v[0] == 'none':
                new_result[k] = None
            else:
                B0, B1 = v.shape[:2]
                other_shape = v.shape[2:]
                new_result[k] = v.reshape(B0*B1, *other_shape)
    else:
        raise TypeError("invalid dataset item type: {}".format(type(result)))
    return new_result
