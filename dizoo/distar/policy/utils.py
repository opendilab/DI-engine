import torch
import torch.nn.functional as F
from ding.torch_utils import flatten, sequence_mask
from ding.utils.data import default_collate
from dizoo.distar.envs import MAX_SELECTED_UNITS_NUM

MASK_INF = -1e9
EPS = 1e-9


def padding_entity_info(traj_data, max_entity_num):
    # traj_data.pop('map_name', None)
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


def entropy_error(target_policy_probs_dict, target_policy_log_probs_dict, mask, head_weights_dict):
    total_entropy_loss = 0.
    entropy_dict = {}
    for head_type in ['action_type', 'queued', 'delay', 'selected_units', 'target_unit', 'target_location']:
        ent = -target_policy_probs_dict[head_type] * target_policy_log_probs_dict[head_type]
        if head_type == 'selected_units':
            ent = ent.sum(dim=-1) / (
                EPS + torch.log(mask['selected_units_logits_mask'].float().sum(dim=-1) + 1).unsqueeze(-1)
            )  # normalize
            ent = (ent * mask['selected_units_mask']).sum(-1)
            ent = ent.div(mask['selected_units_mask'].sum(-1) + EPS)
        elif head_type == 'target_unit':
            # normalize by unit
            ent = ent.sum(dim=-1) / (EPS + torch.log(mask['target_units_logits_mask'].float().sum(dim=-1) + 1))
        else:
            ent = ent.sum(dim=-1) / torch.log(torch.FloatTensor([ent.shape[-1]]).to(ent.device))
        if head_type not in ['action_type', 'delay']:
            ent = ent * mask['actions_mask'][head_type]
        entropy = ent.mean()
        entropy_dict['entropy/' + head_type] = entropy.item()
        total_entropy_loss += (-entropy * head_weights_dict[head_type])
    return total_entropy_loss, entropy_dict


def kl_error(
    target_policy_log_probs_dict, teacher_policy_logits_dict, mask, game_steps, action_type_kl_steps, head_weights_dict
):
    total_kl_loss = 0.
    kl_loss_dict = {}

    for head_type in ['action_type', 'queued', 'delay', 'selected_units', 'target_unit', 'target_location']:
        target_policy_log_probs = target_policy_log_probs_dict[head_type]
        teacher_policy_logits = teacher_policy_logits_dict[head_type]

        teacher_policy_log_probs = F.log_softmax(teacher_policy_logits, dim=-1)
        teacher_policy_probs = torch.exp(teacher_policy_log_probs)
        kl = teacher_policy_probs * (teacher_policy_log_probs - target_policy_log_probs)

        kl = kl.sum(dim=-1)
        if head_type == 'selected_units':
            kl = (kl * mask['selected_units_mask']).sum(-1)
        if head_type not in ['action_type', 'delay']:
            kl = kl * mask['actions_mask'][head_type]
        if head_type == 'action_type':
            flag = game_steps < action_type_kl_steps
            action_type_kl = kl * flag * mask['cum_action_mask']
            action_type_kl_loss = action_type_kl.mean()
            kl_loss_dict['kl/extra_at'] = action_type_kl_loss.item()
        kl_loss = kl.mean()
        total_kl_loss += (kl_loss * head_weights_dict[head_type])
        kl_loss_dict['kl/' + head_type] = kl_loss.item()
    return total_kl_loss, action_type_kl_loss, kl_loss_dict
