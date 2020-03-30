Code Review
===================

.. toctree::
    :maxdepth: 3

Code Comment Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have the following comment format:

  1. file head 
      - goal: copyright, license and the file main function
  2. line comment 
      - goal: explain the implementation or emphasize the code detail
  3. class and function comment
      - goal: explain the main function, input and output type and effect
      - format: Overview, Arguments, Returns (if it doesn't have the corresponding attribute, you can omit it)
  4. TODO
      - goal: set a todo task
      - format: `# TODO(assignee name) task statement`
  5. ISSUE
      - goal: issue some doubt about the code
      - format: `# ISSUE(questioner name) issue statement`

And you can know about the specific format from next code example

.. code::

    '''
    Copyright 2020 Sensetime X-lab. All Rights Reserved

    Main Function:
        1. dataset design for supervised learning from replay data, Pytorch
    '''
    import os
    import torch
    import numpy as np
    import numbers
    import random
    from torch.utils.data import Dataset
    from torch.utils.data._utils.collate import default_collate
    from sc2learner.envs.observations.alphastar_obs_wrapper import decompress_obs


    META_SUFFIX = '.meta'  # the suffix of meta data info, such as race, mmr and so on
    DATA_SUFFIX = '.step'  # the suffix of each step game data
    STAT_SUFFIX = '.stat_processed'  # the suffix of processed statistics data


    class ReplayDataset(Dataset):
        '''
            Overview: map-style dataset for replay data
            Interface: __init__, __len__, __getitem__, state_dict, load_state_dict, step
        '''
        def __init__(self, cfg):
            '''
                Overview: initialization method, parse config and prepare related arguments
                Arguments:
                    - cfg (:obj:`dict`): dataset config
            '''
            super(ReplayDataset, self).__init__()
            assert(cfg.data.trajectory_type in ['random', 'slide_window'])
            with open(cfg.data.replay_list, 'r') as f:
                path_list = f.readlines()
            self.path_list = [{'name': p[:-1], 'count': 0} for idx, p in enumerate(path_list)]
            self.trajectory_len = cfg.data.trajectory_len
            self.trajectory_type = cfg.data.trajectory_type
            self.slide_window_step = cfg.data.slide_window_step
            self.use_stat = cfg.data.use_stat
            self.beginning_build_order_num = cfg.data.beginning_build_order_num
            self.beginning_build_order_prob = cfg.data.beginning_build_order_prob
            self.cumulative_stat_prob = cfg.data.cumulative_stat_prob

        def __len__(self):
            '''
                Overview: get the length of the dataset
                Returns:
                    - (:obj`int`): the length of the dataset
            '''
            return len(self.path_list)

        def state_dict(self):
            # ISSUE(nyz) why the dataset need to save state_dict
            return self.path_list

        def load_state_dict(self, state_dict):
            self.path_list = state_dict

        def copy(self, data):
            if isinstance(data, dict):
                new_data = {}
                for k, v in data.items():
                    new_data[k] = self.copy(v)
            elif isinstance(data, list) or isinstance(data, tuple):
                new_data = []
                for item in data:
                    new_data.append(self.copy(item))
            elif isinstance(data, torch.Tensor):
                new_data = data.clone()
            elif isinstance(data, np.ndarray):
                new_data = np.copy(data)
            elif isinstance(data, str) or isinstance(data, numbers.Integral):
                new_data = data
            else:
                raise TypeError("invalid data type:{}".format(type(data)))
            return new_data

        def action_unit_id_transform(self, data):
            new_data = []
            for idx, item in enumerate(data):
                valid = True
                # deepcopy the data in order to avoid the bug occurred during modify data
                item = self.copy(data[idx])
                id_list = item['entity_raw']['id']
                action = item['actions']
                if isinstance(action['selected_units'], torch.Tensor):
                    unit_ids = []
                    for unit in action['selected_units']:
                        val = unit.item()
                        if val in id_list:
                            unit_ids.append(id_list.index(val))
                        else:
                            # TODO(nyz) add this type error message into logger
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

        def step(self):
            for i in range(len(self.path_list)):
                handle = self.path_list[i]
                if 'step_num' not in handle.keys():
                    meta = torch.load(handle['name'] + META_SUFFIX)
                    step_num = meta['step_num']
                    handle['step_num'] = step_num
                else:
                    step_num = handle['step_num']
                assert(handle['step_num'] >= self.trajectory_len)
                if self.trajectory_type == 'random':
                    handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
                elif self.trajectory_type == 'slide_window':
                    if 'cur_step' not in handle.keys():
                        handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
                    else:
                        next_step = handle['cur_step'] + self.slide_window_step
                        if next_step >= step_num - self.trajectory_len:
                            handle['cur_step'] = random.randint(0, step_num - self.trajectory_len)
                        else:
                            handle['cur_step'] = next_step

        def _load_stat(self, handle):
            stat = torch.load(handle['name'] + STAT_SUFFIX)
            mmr = stat['mmr']
            beginning_build_order = stat['beginning_build_order']
            # first self.beginning_build_order_num item
            beginning_build_order = beginning_build_order[:self.beginning_build_order_num]
            if beginning_build_order.shape[0] < self.beginning_build_order_num:
                B, N = beginning_build_order.shape
                B0 = self.beginning_build_order_num - B
                beginning_build_order = torch.cat([beginning_build_order, torch.zeros(B0, N)])
            cumulative_stat = stat['cumulative_stat']
            bool_bo = float(np.random.rand() < self.beginning_build_order_prob)
            bool_cum = float(np.random.rand() < self.cumulative_stat_prob)
            beginning_build_order = bool_bo * beginning_build_order
            cumulative_stat = {k: bool_cum * v for k, v in cumulative_stat.items()}
            return beginning_build_order, cumulative_stat, mmr

        def __getitem__(self, idx):
            handle = self.path_list[idx]
            data = torch.load(handle['name'] + DATA_SUFFIX)
            start = handle['cur_step']
            end = start + self.trajectory_len
            sample_data = data[start:end]
            # if unit id transform deletes some data frames,
            # collate_fn will use the minimum number of data frame to compose a batch
            sample_data = self.action_unit_id_transform(sample_data)
            # decompress_obs data from bits data to the float32 type data
            sample_data = [decompress_obs(d) for d in sample_data]
            if self.use_stat:
                beginning_build_order, cumulative_stat, mmr = self._load_stat(handle)
                for i in range(len(sample_data)):
                    sample_data[i]['scalar_info']['beginning_build_order'] = beginning_build_order
                    sample_data[i]['scalar_info']['cumulative_stat'] = cumulative_stat
                    sample_data[i]['scalar_info']['mmr'] = mmr

            return sample_data

.. note::
    Usually, we add line comment behind the corresponding code line, but sometimes it will surpass the max-line-length(default 120),
    and you can solve this problem by adding **# noqa** at the end of this line, which is the ignore sign of the codestyle checker.
