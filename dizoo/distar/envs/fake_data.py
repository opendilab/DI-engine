from typing import Sequence
import torch

from ding.utils.data import default_collate
from .meta import MAX_DELAY, MAX_ENTITY_NUM, NUM_ACTIONS, NUM_UNIT_TYPES, NUM_UPGRADES, NUM_CUMULATIVE_STAT_ACTIONS, \
    NUM_BEGINNING_ORDER_ACTIONS, NUM_UNIT_MIX_ABILITIES, NUM_QUEUE_ACTION, NUM_BUFFS, NUM_ADDON, \
    MAX_SELECTED_UNITS_NUM, DEFAULT_SPATIAL_SIZE

H, W = DEFAULT_SPATIAL_SIZE


def spatial_info():
    return {
        'height_map': torch.rand(H, W),
        'visibility_map': torch.randint(0, 4, size=(H, W), dtype=torch.float),
        'creep': torch.randint(0, 2, size=(H, W), dtype=torch.float),
        'player_relative': torch.randint(0, 5, size=(H, W), dtype=torch.float),
        'alerts': torch.randint(0, 2, size=(H, W), dtype=torch.float),
        'pathable': torch.randint(0, 2, size=(H, W), dtype=torch.float),
        'buildable': torch.randint(0, 2, size=(H, W), dtype=torch.float),
        'effect_PsiStorm': torch.randint(0, min(H, W), size=(2, ), dtype=torch.float),
        'effect_NukeDot': torch.randint(0, min(H, W), size=(2, ), dtype=torch.float),
        'effect_LiberatorDefenderZone': torch.randint(0, min(H, W), size=(2, ), dtype=torch.float),
        'effect_BlindingCloud': torch.randint(0, min(H, W), size=(2, ), dtype=torch.float),
        'effect_CorrosiveBile': torch.randint(0, min(H, W), size=(2, ), dtype=torch.float),
        'effect_LurkerSpines': torch.randint(0, min(H, W), size=(2, ), dtype=torch.float),
    }


def entity_info():
    data = {
        'unit_type': torch.randint(0, NUM_UNIT_TYPES, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'alliance': torch.randint(0, 5, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'cargo_space_taken': torch.randint(0, 9, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'build_progress': torch.rand(MAX_ENTITY_NUM),
        'health_ratio': torch.rand(MAX_ENTITY_NUM),
        'shield_ratio': torch.rand(MAX_ENTITY_NUM),
        'energy_ratio': torch.rand(MAX_ENTITY_NUM),
        'display_type': torch.randint(0, 5, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'x': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'y': torch.randint(0, 11, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'cloak': torch.randint(0, 5, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'is_blip': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'is_powered': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'mineral_contents': torch.rand(MAX_ENTITY_NUM),
        'vespene_contents': torch.rand(MAX_ENTITY_NUM),
        'cargo_space_max': torch.randint(0, 9, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'assigned_harvesters': torch.randint(0, 24, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'weapon_cooldown': torch.randint(0, 32, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'order_length': torch.randint(0, 9, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'order_id_0': torch.randint(0, NUM_ACTIONS, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'order_id_1': torch.randint(0, NUM_QUEUE_ACTION, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'is_hallucination': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'buff_id_0': torch.randint(0, NUM_BUFFS, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'buff_id_1': torch.randint(0, NUM_BUFFS, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'addon_unit_type': torch.randint(0, NUM_ADDON, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'is_active': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'order_progress_0': torch.rand(MAX_ENTITY_NUM),
        'order_progress_1': torch.rand(MAX_ENTITY_NUM),
        'order_id_2': torch.randint(0, NUM_QUEUE_ACTION, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'order_id_3': torch.randint(0, NUM_QUEUE_ACTION, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'is_in_cargo': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'attack_upgrade_level': torch.randint(0, 4, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'armor_upgrade_level': torch.randint(0, 4, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'shield_upgrade_level': torch.randint(0, 4, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'last_selected_units': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
        'last_targeted_unit': torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.float),
    }
    return data


def scalar_info():
    data = {
        'home_race': torch.randint(0, 4, size=(), dtype=torch.float),
        'away_race': torch.randint(0, 4, size=(), dtype=torch.float),
        'agent_statistics': torch.rand(10),
        'time': torch.randint(0, 100, size=(), dtype=torch.float),
        'unit_counts_bow': torch.randint(0, 10, size=(NUM_UNIT_TYPES, ), dtype=torch.float),
        'beginning_build_order': torch.randint(0, 20, size=(20, ), dtype=torch.float),
        'cumulative_stat': torch.randint(0, 2, size=(NUM_CUMULATIVE_STAT_ACTIONS, ), dtype=torch.float),
        'last_delay': torch.randint(0, MAX_DELAY, size=(), dtype=torch.float),
        'last_queued': torch.randint(0, 2, size=(), dtype=torch.float),
        'last_action_type': torch.randint(0, NUM_ACTIONS, size=(), dtype=torch.float),
        'upgrades': torch.randint(0, 2, size=(NUM_UPGRADES, ), dtype=torch.float),
        'beginning_order': torch.randint(0, NUM_BEGINNING_ORDER_ACTIONS, size=(20, ), dtype=torch.float),
        'bo_location': torch.randint(0, 100 * 100, size=(20, ), dtype=torch.float),
        'unit_type_bool': torch.randint(0, 2, size=(NUM_UNIT_TYPES, ), dtype=torch.float),
        'enemy_unit_type_bool': torch.randint(0, 2, size=(NUM_UNIT_TYPES, ), dtype=torch.float),
        'unit_order_type': torch.randint(0, 2, size=(NUM_UNIT_MIX_ABILITIES, ), dtype=torch.float)
    }
    return data


def get_mask(action):
    mask = {
        'action_type': torch.ones(1, dtype=torch.long).squeeze(),
        'delay': torch.ones(1, dtype=torch.long).squeeze(),
        'queued': torch.ones(1, dtype=torch.long).squeeze(),
        'selected_units': torch.randint(0, 2, size=(), dtype=torch.long),
        'target_unit': torch.randint(0, 2, size=(), dtype=torch.long),
        'target_location': torch.randint(0, 2, size=(), dtype=torch.long)
    }
    selected_units_logits_mask = torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.long)
    target_units_logits_mask = torch.randint(0, 2, size=(MAX_ENTITY_NUM, ), dtype=torch.long)
    target_units_logits_mask[action['target_unit']] = 1

    return {
        'actions_mask': mask,
        'selected_units_logits_mask': selected_units_logits_mask,
        'target_units_logits_mask': target_units_logits_mask,
    }


def action_info():
    data = {
        'action_type': torch.randint(0, NUM_ACTIONS, size=(), dtype=torch.long),
        'delay': torch.randint(0, MAX_DELAY, size=(), dtype=torch.long),
        'queued': torch.randint(0, 2, size=(), dtype=torch.long),
        'selected_units': torch.randint(0, 5, size=(MAX_SELECTED_UNITS_NUM, ), dtype=torch.long),
        'target_unit': torch.randint(0, MAX_ENTITY_NUM, size=(), dtype=torch.long),
        'target_location': torch.randint(0, H, size=(), dtype=torch.long)
    }
    mask = get_mask(data)
    return data, mask


def action_logits(logp=False, action=None):
    data = {
        'action_type': torch.rand(size=(NUM_ACTIONS, )) - 0.5,
        'delay': torch.rand(size=(MAX_DELAY, )) - 0.5,
        'queued': torch.rand(size=(2, )) - 0.5,
        'selected_units': torch.rand(size=(MAX_SELECTED_UNITS_NUM, MAX_ENTITY_NUM + 1)) - 0.5,
        'target_unit': torch.rand(size=(MAX_ENTITY_NUM, )) - 0.5,
        'target_location': torch.rand(size=(H * W, )) - 0.5
    }
    if logp:
        for k in data:
            dist = torch.distributions.Categorical(logits=data[k])
            data[k] = dist.log_prob(action[k])
    return data


def rl_step_data(last=False):
    action, mask = action_info()
    teacher_action_logits = action_logits()
    data = {
        'spatial_info': spatial_info(),
        'entity_info': entity_info(),
        'scalar_info': scalar_info(),
        'entity_num': torch.randint(5, 100, size=(1, ), dtype=torch.long),
        'selected_units_num': torch.randint(0, MAX_SELECTED_UNITS_NUM, size=(), dtype=torch.long),
        'entity_location': torch.randint(0, H, size=(512, 2), dtype=torch.long),
        'hidden_state': [(torch.zeros(size=(384, )), torch.zeros(size=(384, ))) for _ in range(3)],
        'action_info': action,
        'behaviour_logp': action_logits(logp=True, action=action),
        'teacher_logit': action_logits(logp=False),
        'reward': {
            'winloss': torch.randint(-1, 1, size=(), dtype=torch.float),
            'build_order': torch.randint(-1, 1, size=(), dtype=torch.float),
            'built_unit': torch.randint(-1, 1, size=(), dtype=torch.float),
            'effect': torch.randint(-1, 1, size=(), dtype=torch.float),
            'upgrade': torch.randint(-1, 1, size=(), dtype=torch.float),
            'battle': torch.randint(-1, 1, size=(), dtype=torch.float),
        },
        'step': torch.randint(100, 1000, size=(), dtype=torch.long),
        'mask': mask,
    }
    if last:
        for k in ['mask', 'action_info', 'teacher_logit', 'behaviour_logp', 'selected_units_num', 'reward', 'step']:
            data.pop(k)
    return data


def fake_rl_data_batch_with_last(unroll_len=4):
    list_step_data = []
    for i in range(unroll_len):
        step_data = rl_step_data()
        list_step_data.append(step_data)
    list_step_data.append(rl_step_data(last=True))  # last step
    return list_step_data


def get_fake_rl_trajectory(batch_size=3, unroll_len=4):
    data_batch = [fake_rl_data_batch_with_last(unroll_len) for _ in range(batch_size)]
    return data_batch
