import torch
from ding.torch_utils import flatten, sequence_mask
from ding.utils.data import default_collate
from dizoo.distar.envs import MAX_SELECTED_UNITS_NUM

MASK_INF = -1e9


def padding_entity_info(traj_data, max_entity_num):
    traj_data.pop('map_name', None)
    entity_padding_num = max_entity_num - len(traj_data['entity_info']['x'])
    if 'entity_embeddings' in traj_data.keys():
        traj_data['entity_embeddings'] = torch.nn.functional.pad(
            traj_data['entity_embeddings'], (0, 0, 0, entity_padding_num), 'constant', 0
        )

    for k in traj_data['entity_info'].keys():
        traj_data['entity_info'][k] = torch.nn.functional.pad(
            traj_data['entity_info'][k], (0, entity_padding_num), 'constant', 0
        )
    if 'action_info' in traj_data:
        su_padding_num = MAX_SELECTED_UNITS_NUM - traj_data['teacher_logit']['selected_units'].shape[0]

        traj_data['mask']['selected_units_mask'] = sequence_mask(
            traj_data['selected_units_num'].unsqueeze(dim=0), max_len=MAX_SELECTED_UNITS_NUM
        ).squeeze(dim=0)
        traj_data['action_info']['selected_units'] = torch.nn.functional.pad(
            traj_data['action_info']['selected_units'],
            (0, MAX_SELECTED_UNITS_NUM - traj_data['action_info']['selected_units'].shape[-1]), 'constant', 0
        )

        traj_data['behaviour_logp']['selected_units'] = torch.nn.functional.pad(
            traj_data['behaviour_logp']['selected_units'], (
                0,
                su_padding_num,
            ), 'constant', MASK_INF
        )

        traj_data['teacher_logit']['selected_units'] = torch.nn.functional.pad(
            traj_data['teacher_logit']['selected_units'], (
                0,
                entity_padding_num,
                0,
                su_padding_num,
            ), 'constant', MASK_INF
        )
        traj_data['teacher_logit']['target_unit'] = torch.nn.functional.pad(
            traj_data['teacher_logit']['target_unit'], (0, entity_padding_num), 'constant', MASK_INF
        )

        traj_data['mask']['selected_units_logits_mask'] = sequence_mask(
            traj_data['entity_num'].unsqueeze(dim=0) + 1, max_len=max_entity_num + 1
        ).squeeze(dim=0)
        traj_data['mask']['target_units_logits_mask'] = sequence_mask(
            traj_data['entity_num'].unsqueeze(dim=0), max_len=max_entity_num
        ).squeeze(dim=0)

    return traj_data


def collate_fn_learn(traj_batch):
    # data list of list, with shape batch_size, unroll_len
    # find max_entity_num in data_batch
    max_entity_num = max(
        [len(traj_data['entity_info']['x']) for traj_data_list in traj_batch for traj_data in traj_data_list]
    )

    # padding entity_info in observation, target_unit, selected_units, mask
    traj_batch = [
        [padding_entity_info(traj_data, max_entity_num) for traj_data in traj_data_list]
        for traj_data_list in traj_batch
    ]

    data = [default_collate(traj_data_list, allow_key_mismatch=True) for traj_data_list in traj_batch]

    batch_size = len(data)
    unroll_len = len(data[0]['step'])
    data = default_collate(data, dim=1)

    new_data = {}
    for k, val in data.items():
        if k in ['spatial_info', 'entity_info', 'scalar_info', 'entity_num', 'entity_location', 'hidden_state',
                 'value_feature']:
            new_data[k] = flatten(val)
        else:
            new_data[k] = val
    new_data['aux_type'] = batch_size
    new_data['batch_size'] = batch_size
    new_data['unroll_len'] = unroll_len
    return new_data
