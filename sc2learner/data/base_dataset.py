import torch
from torch.utils.data import Dataset

from sc2learner.utils import deepcopy

META_SUFFIX = '.meta'
DATA_SUFFIX = '.step'
STAT_SUFFIX = '.stat_processed'


class SenseStarBaseDataset(Dataset):
    def __init__(self, cfg):
        super(SenseStarBaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()

    def step(self, index=None):
        """We can assume everydataset has step function."""
        pass

    def reset_step(self, index=None):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError()

    def action_unit_id_transform(self, data):
        # TODO(pzh)  @NYZ to see if this is work for any type of data (RL & SL)
        new_data = []
        for idx, item in enumerate(data):
            valid = True
            item = deepcopy(data[idx])
            id_list = item['entity_raw']['id']
            action = item['actions']
            if isinstance(action['selected_units'], torch.Tensor):
                unit_ids = []
                for unit in action['selected_units']:
                    val = unit.item()
                    if val in id_list:
                        unit_ids.append(id_list.index(val))
                    else:
                        print("not found selected_units id({}) in nearest observation".format(val))
                        valid = False
                        break
                item['actions']['selected_units'] = torch.LongTensor(unit_ids)
            if isinstance(action['target_units'], torch.Tensor):
                unit_ids = []
                for unit in action['target_units']:
                    val = unit.item()
                    if val in id_list:
                        unit_ids.append(id_list.index(val))
                    else:
                        print("not found target_units id({}) in nearest observation".format(val))
                        valid = False
                        break
                item['actions']['target_units'] = torch.LongTensor(unit_ids)
            if valid:
                new_data.append(item)
        return new_data
