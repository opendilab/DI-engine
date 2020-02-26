import torch
from torch.utils.data import _utils


class OnlineDataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn=None):
        self.dataset = dataset
        if collate_fn is None:
            self.collate_fn = _utils.collate.default_collate
        else:
            self.collate_fn = collate_fn
        self.batch_size = batch_size

    def __next__(self):
        batch, avg_usage, push_count, avg_model_index = \
               self.dataset.get_sample_batch(self.batch_size, self.cur_model_index)
        batch = self.collate_fn(batch)
        return batch, avg_usage, push_count, avg_model_index

    @property
    def cur_model_index(self):
        return self._cur_model_index

    @cur_model_index.setter
    def cur_model_index(self, cur_model_index):
        self._cur_model_index = cur_model_index


def unroll_split_collate_fn(*args, collate_fn=_utils.collate.default_collate, **kwargs):
    # TODO: replace this hacky workaround for non unique sized data chunks
    # result = collate_fn(*args, **kwargs)
    # Expecting a list of dict as input
    # there are multiple samples in each key of dict (as a list or Tensor)
    # Returning a single dict with all samples in each key (as Tensor if possible)
    result = args[0]
    new_result = {}
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)
        for k,v in item.items():
            if isinstance(v, list) and v[0] == 'none':
                new_result[k] = None
            elif isinstance(v, str) and v == 'none':
                new_result[k] = None
            elif isinstance(v, torch.Tensor):
                if k in new_result:
                    new_result[k] = torch.cat((new_result[k], v))
                else:
                    new_result[k] = v
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                if k in new_result:
                    new_result[k] = torch.cat((new_result[k], torch.cat(v)))
                else:
                    new_result[k] = torch.cat(v)
            elif isinstance(v, list):
                if k in new_result:
                    new_result[k].extend(v)
                else:
                    new_result[k] = v
            else:
                print('WARNING: item {} of type {} in data discarded'.format(k, str(type(v))))
    return new_result
