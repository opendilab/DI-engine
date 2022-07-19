from typing import Optional, List
from copy import deepcopy
from collections import defaultdict
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib import named_array, colors, point
from pysc2.lib.features import Effects, ScoreCategories, FeatureType, Feature
from torch import int8, uint8, int16, int32, float32, float16, int64

import six
import collections
import enum
import json
import random
import numpy as np
import torch
import os.path as osp

from ding.envs import BaseEnv, BaseEnvTimestep
try:
    from distar.envs.env import SC2Env
except ImportError:

    class SC2Env:
        pass
from .meta import MAX_DELAY, MAX_SELECTED_UNITS_NUM, DEFAULT_SPATIAL_SIZE, MAX_ENTITY_NUM, NUM_UPGRADES, NUM_ACTIONS, \
    NUM_UNIT_TYPES, NUM_UNIT_MIX_ABILITIES, EFFECT_LENGTH, UPGRADE_LENGTH, BEGINNING_ORDER_LENGTH
from .static_data import ACTIONS_STAT, RACE_DICT, UNIT_TYPES_REORDER_ARRAY, UPGRADES_REORDER_ARRAY, \
    CUMULATIVE_STAT_ACTIONS, UNIT_ABILITY_REORDER, ABILITY_TO_QUEUE_ACTION, BUFFS_REORDER_ARRAY, \
    ADDON_REORDER_ARRAY


class DIStarEnv(SC2Env, BaseEnv):

    def __init__(self, cfg):
        super(DIStarEnv, self).__init__(cfg)

    def reset(self):
        observations, game_info, map_name = super(DIStarEnv, self).reset()

        for policy_id, policy_obs in observations.items():
            policy_obs['game_info'] = game_info[policy_id]
            map_size = game_info[policy_id].start_raw.map_size
            policy_obs['map_name'] = map_name
            policy_obs['map_size'] = map_size

        return observations

    def close(self):
        super(DIStarEnv, self).close()

    def step(self, actions):
        # In DI-engine, the return of BaseEnv.step is ('obs', 'reward', 'done', 'info')
        # Here in DI-star, the return is ({'raw_obs': self._obs[agent_idx], 'opponent_obs': opponent_obs,
        # 'action_result': self._action_result[agent_idx]}, reward, episode_complete)
        next_observations, reward, done = super(DIStarEnv, self).step(actions)
        # next_observations 和 observations 格式一样
        # reward 是 list [policy reward 1, policy reward 2]
        # done 是 一个 bool 值
        info = {}
        for policy_id in range(self._num_agents):
            info[policy_id] = {}
            if done:
                info[policy_id]['final_eval_reward'] = reward[policy_id]
                info[policy_id]['result'] = 'draws'
                if reward[policy_id] == 1:
                    info[policy_id]['result'] = 'wins'
                elif reward[policy_id] == -1:
                    info[policy_id]['result'] = 'losses'
        timestep = BaseEnvTimestep(obs=next_observations, reward=reward, done=done, info=info)
        return timestep

    def seed(self, seed, dynamic_seed=False):
        self._random_seed = seed

    @property
    def game_info(self):
        return self._game_info

    @property
    def map_name(self):
        return self._map_name

    @property
    def observation_space(self):
        #TODO
        pass

    @property
    def action_space(self):
        #TODO
        pass

    @classmethod
    def random_action(cls, obs):
        raw = obs['raw_obs'].observation.raw_data

        all_unit_types = set()
        self_unit_types = set()

        for u in raw.units:
            # Here we select the units except “buildings that are in building progress” for simplification
            if u.build_progress == 1:
                all_unit_types.add(u.unit_type)
                if u.alliance == 1:
                    self_unit_types.add(u.unit_type)

        avail_actions = [
            {
                0: {
                    'exist_selected_types': [],
                    'exist_target_types': []
                }
            }, {
                168: {
                    'exist_selected_types': [],
                    'exist_target_types': []
                }
            }
        ]  # no_op and raw_move_camera don't have seleted_units

        for action_id, action in ACTIONS_STAT.items():
            exist_selected_types = list(self_unit_types.intersection(set(action['selected_type'])))
            exist_target_types = list(all_unit_types.intersection(set(action['target_type'])))

            # if an action should have target, but we don't have valid target in this observation,
            # then discard this action
            if len(action['target_type']) != 0 and len(exist_target_types) == 0:
                continue

            if len(exist_selected_types) > 0:
                avail_actions.append(
                    {
                        action_id: {
                            'exist_selected_types': exist_selected_types,
                            'exist_target_types': exist_target_types
                        }
                    }
                )

        current_action = random.choice(avail_actions)
        func_id, exist_types = current_action.popitem()

        if func_id not in [0, 168]:
            correspond_selected_units = [
                u.tag for u in raw.units if u.unit_type in exist_types['exist_selected_types'] and u.build_progress == 1
            ]
            correspond_targets = [
                u.tag for u in raw.units if u.unit_type in exist_types['exist_target_types'] and u.build_progress == 1
            ]

            num_selected_unit = random.randint(0, min(MAX_SELECTED_UNITS_NUM, len(correspond_selected_units)))

            unit_tags = random.sample(correspond_selected_units, num_selected_unit)
            target_unit_tag = random.choice(correspond_targets) if len(correspond_targets) > 0 else None

        else:
            unit_tags = []
            target_unit_tag = None

        data = {
            'func_id': func_id,
            'skip_steps': random.randint(0, MAX_DELAY - 1),
            # 'skip_steps': 8,
            'queued': random.randint(0, 1),
            'unit_tags': unit_tags,
            'target_unit_tag': target_unit_tag,
            'location': (
                random.randint(0, DEFAULT_SPATIAL_SIZE[0] - 1), random.randint(0, DEFAULT_SPATIAL_SIZE[1] - 1)
            )
        }
        return [data]

    @property
    def reward_space(self):
        #TODO
        pass

    def __repr__(self):
        return "DI-engine DI-star Env"


def parse_new_game(data, z_path: str, z_idx: Optional[None] = List):
    # init Z
    z_path = osp.join(osp.dirname(__file__), "z_files", z_path)
    with open(z_path, 'r') as f:
        z_data = json.load(f)

    raw_ob = data['raw_obs']
    game_info = data['game_info']
    map_size = data['map_size']
    if isinstance(map_size, list):
        map_size = point.Point(map_size[0], map_size[1])
    map_name = data['map_name']
    requested_races = {
        info.player_id: info.race_requested
        for info in game_info.player_info if info.type != sc_pb.Observer
    }
    location = []
    for i in raw_ob.observation.raw_data.units:
        if i.unit_type == 59 or i.unit_type == 18 or i.unit_type == 86:
            location.append([i.pos.x, i.pos.y])
    assert len(location) == 1, 'no fog of war, check game version!'
    _born_location = deepcopy(location[0])
    born_location = location[0]
    born_location[0] = int(born_location[0])
    born_location[1] = int(map_size.y - born_location[1])
    born_location_str = str(born_location[0] + born_location[1] * 160)

    z_type = None
    idx = None
    race = RACE_DICT[requested_races[raw_ob.observation.player_common.player_id]]
    opponent_id = 1 if raw_ob.observation.player_common.player_id == 2 else 2
    opponent_race = RACE_DICT[requested_races[opponent_id]]
    if race == opponent_race:
        mix_race = race
    else:
        mix_race = race + opponent_race
    if z_idx is not None:
        idx, z_type = random.choice(z_idx[map_name][mix_race][born_location_str])
        z = z_data[map_name][mix_race][born_location_str][idx]
    else:
        z = random.choice(z_data[map_name][mix_race][born_location_str])

    if len(z) == 5:
        target_building_order, target_cumulative_stat, bo_location, target_z_loop, z_type = z
    else:
        target_building_order, target_cumulative_stat, bo_location, target_z_loop = z
    return race, requested_races, map_size, target_building_order, target_cumulative_stat, bo_location, target_z_loop, z_type, _born_location


class FeatureUnit(enum.IntEnum):
    """Indices for the `feature_unit` observations."""
    unit_type = 0
    alliance = 1
    cargo_space_taken = 2
    build_progress = 3
    health_max = 4
    shield_max = 5
    energy_max = 6
    display_type = 7
    owner = 8
    x = 9
    y = 10
    cloak = 11
    is_blip = 12
    is_powered = 13
    mineral_contents = 14
    vespene_contents = 15
    cargo_space_max = 16
    assigned_harvesters = 17
    weapon_cooldown = 18
    order_length = 19  # If zero, the unit is idle.
    order_id_0 = 20
    order_id_1 = 21
    # tag = 22  # Unique identifier for a unit (only populated for raw units).
    is_hallucination = 22
    buff_id_0 = 23
    buff_id_1 = 24
    addon_unit_type = 25
    is_active = 26
    order_progress_0 = 27
    order_progress_1 = 28
    order_id_2 = 29
    order_id_3 = 30
    is_in_cargo = 31
    attack_upgrade_level = 32
    armor_upgrade_level = 33
    shield_upgrade_level = 34
    health = 35
    shield = 36
    energy = 37


class MinimapFeatures(collections.namedtuple(
        "MinimapFeatures",
    ["height_map", "visibility_map", "creep", "player_relative", "alerts", "pathable", "buildable"])):
    """The set of minimap feature layers."""
    __slots__ = ()

    def __new__(cls, **kwargs):
        feats = {}
        for name, (scale, type_, palette) in six.iteritems(kwargs):
            feats[name] = Feature(
                index=MinimapFeatures._fields.index(name),
                name=name,
                layer_set="minimap_renders",
                full_name="minimap " + name,
                scale=scale,
                type=type_,
                palette=palette(scale) if callable(palette) else palette,
                clip=False
            )
        return super(MinimapFeatures, cls).__new__(cls, **feats)  # pytype: disable=missing-parameter


MINIMAP_FEATURES = MinimapFeatures(
    height_map=(256, FeatureType.SCALAR, colors.height_map),
    visibility_map=(4, FeatureType.CATEGORICAL, colors.VISIBILITY_PALETTE),
    creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE),
    player_relative=(5, FeatureType.CATEGORICAL, colors.PLAYER_RELATIVE_PALETTE),
    alerts=(2, FeatureType.CATEGORICAL, colors.winter),
    pathable=(2, FeatureType.CATEGORICAL, colors.winter),
    buildable=(2, FeatureType.CATEGORICAL, colors.winter),
)

SPATIAL_INFO = [
    ('height_map', uint8), ('visibility_map', uint8), ('creep', uint8), ('player_relative', uint8), ('alerts', uint8),
    ('pathable', uint8), ('buildable', uint8), ('effect_PsiStorm', int16), ('effect_NukeDot', int16),
    ('effect_LiberatorDefenderZone', int16), ('effect_BlindingCloud', int16), ('effect_CorrosiveBile', int16),
    ('effect_LurkerSpines', int16)
]

# (name, dtype, size)
SCALAR_INFO = [
    ('home_race', uint8, ()), ('away_race', uint8, ()), ('upgrades', int16, (NUM_UPGRADES, )), ('time', float32, ()),
    ('unit_counts_bow', uint8, (NUM_UNIT_TYPES, )), ('agent_statistics', float32, (10, )),
    ('cumulative_stat', uint8, (len(CUMULATIVE_STAT_ACTIONS), )),
    ('beginning_order', int16, (BEGINNING_ORDER_LENGTH, )), ('last_queued', int16, ()), ('last_delay', int16, ()),
    ('last_action_type', int16, ()), ('bo_location', int16, (BEGINNING_ORDER_LENGTH, )),
    ('unit_order_type', uint8, (NUM_UNIT_MIX_ABILITIES, )), ('unit_type_bool', uint8, (NUM_UNIT_TYPES, )),
    ('enemy_unit_type_bool', uint8, (NUM_UNIT_TYPES, ))
]

ENTITY_INFO = [
    ('unit_type', int16), ('alliance', uint8), ('cargo_space_taken', uint8), ('build_progress', float16),
    ('health_ratio', float16), ('shield_ratio', float16), ('energy_ratio', float16), ('display_type', uint8),
    ('x', uint8), ('y', uint8), ('cloak', uint8), ('is_blip', uint8), ('is_powered', uint8),
    ('mineral_contents', float16), ('vespene_contents', float16), ('cargo_space_max', uint8),
    ('assigned_harvesters', uint8), ('weapon_cooldown', uint8), ('order_length', uint8), ('order_id_0', int16),
    ('order_id_1', int16), ('is_hallucination', uint8), ('buff_id_0', uint8), ('buff_id_1', uint8),
    ('addon_unit_type', uint8), ('is_active', uint8), ('order_progress_0', float16), ('order_progress_1', float16),
    ('order_id_2', int16), ('order_id_3', int16), ('is_in_cargo', uint8), ('attack_upgrade_level', uint8),
    ('armor_upgrade_level', uint8), ('shield_upgrade_level', uint8), ('last_selected_units', int8),
    ('last_targeted_unit', int8)
]

ACTION_INFO = {
    'action_type': torch.tensor(0, dtype=torch.long),
    'delay': torch.tensor(0, dtype=torch.long),
    'queued': torch.tensor(0, dtype=torch.long),
    'selected_units': torch.zeros((MAX_SELECTED_UNITS_NUM, ), dtype=torch.long),
    'target_unit': torch.tensor(0, dtype=torch.long),
    'target_location': torch.tensor(0, dtype=torch.long)
}

ACTION_LOGP = {
    'action_type': torch.tensor(0, dtype=torch.float),
    'delay': torch.tensor(0, dtype=torch.float),
    'queued': torch.tensor(0, dtype=torch.float),
    'selected_units': torch.zeros((MAX_SELECTED_UNITS_NUM, ), dtype=torch.float),
    'target_unit': torch.tensor(0, dtype=torch.float),
    'target_location': torch.tensor(0, dtype=torch.float)
}

ACTION_LOGIT = {
    'action_type': torch.zeros(NUM_ACTIONS, dtype=torch.float),
    'delay': torch.zeros(MAX_DELAY + 1, dtype=torch.float),
    'queued': torch.zeros(2, dtype=torch.float),
    'selected_units': torch.zeros((MAX_SELECTED_UNITS_NUM, MAX_ENTITY_NUM + 1), dtype=torch.float),
    'target_unit': torch.zeros(MAX_ENTITY_NUM, dtype=torch.float),
    'target_location': torch.zeros(DEFAULT_SPATIAL_SIZE[0] * DEFAULT_SPATIAL_SIZE[1], dtype=torch.float)
}


def compute_battle_score(obs):
    if obs is None:
        return 0.
    score_details = obs.observation.score.score_details
    killed_mineral, killed_vespene = 0., 0.
    for s in ScoreCategories:
        killed_mineral += getattr(score_details.killed_minerals, s.name)
        killed_vespene += getattr(score_details.killed_vespene, s.name)
    battle_score = killed_mineral + 1.5 * killed_vespene
    return battle_score


def transform_obs(obs, map_size, requested_races, padding_spatial=False, opponent_obs=None):
    spatial_info = defaultdict(list)
    scalar_info = {}
    entity_info = {}
    game_info = {}

    raw = obs.observation.raw_data
    # spatial info
    for f in MINIMAP_FEATURES:
        d = f.unpack(obs.observation).copy()
        d = torch.from_numpy(d)
        padding_y = DEFAULT_SPATIAL_SIZE[0] - d.shape[0]
        padding_x = DEFAULT_SPATIAL_SIZE[1] - d.shape[1]
        if (padding_y != 0 or padding_x != 0) and padding_spatial:
            d = torch.nn.functional.pad(d, (0, padding_x, 0, padding_y), 'constant', 0)
        spatial_info[f.name] = d
    for e in raw.effects:
        name = Effects(e.effect_id).name
        if name in ['LiberatorDefenderZone', 'LurkerSpines'] and e.owner == 1:
            continue
        for p in e.pos:
            location = int(p.x) + int(map_size.y - p.y) * DEFAULT_SPATIAL_SIZE[1]
            spatial_info['effect_' + name].append(location)
    for k, _ in SPATIAL_INFO:
        if 'effect' in k:
            padding_num = EFFECT_LENGTH - len(spatial_info[k])
            if padding_num > 0:
                spatial_info[k] += [0] * padding_num
            else:
                spatial_info[k] = spatial_info[k][:EFFECT_LENGTH]
            spatial_info[k] = torch.as_tensor(spatial_info[k], dtype=int16)

    # entity info
    tag_types = {}  # Only populate the cache if it's needed.

    def get_addon_type(tag):
        if not tag_types:
            for u in raw.units:
                tag_types[u.tag] = u.unit_type
        return tag_types.get(tag, 0)

    tags = []
    units = []
    for u in raw.units:
        tags.append(u.tag)
        units.append(
            [
                u.unit_type,
                u.alliance,  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4
                u.cargo_space_taken,
                u.build_progress,
                u.health_max,
                u.shield_max,
                u.energy_max,
                u.display_type,  # Visible = 1, Snapshot = 2, Hidden = 3
                u.owner,  # 1-15, 16 = neutral
                u.pos.x,
                u.pos.y,
                u.cloak,  # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
                u.is_blip,
                u.is_powered,
                u.mineral_contents,
                u.vespene_contents,
                # Not populated for enemies or neutral
                u.cargo_space_max,
                u.assigned_harvesters,
                u.weapon_cooldown,
                len(u.orders),
                u.orders[0].ability_id if len(u.orders) > 0 else 0,
                u.orders[1].ability_id if len(u.orders) > 1 else 0,
                u.is_hallucination,
                u.buff_ids[0] if len(u.buff_ids) >= 1 else 0,
                u.buff_ids[1] if len(u.buff_ids) >= 2 else 0,
                get_addon_type(u.add_on_tag) if u.add_on_tag else 0,
                u.is_active,
                u.orders[0].progress if len(u.orders) >= 1 else 0,
                u.orders[1].progress if len(u.orders) >= 2 else 0,
                u.orders[2].ability_id if len(u.orders) > 2 else 0,
                u.orders[3].ability_id if len(u.orders) > 3 else 0,
                0,
                u.attack_upgrade_level,
                u.armor_upgrade_level,
                u.shield_upgrade_level,
                u.health,
                u.shield,
                u.energy,
            ]
        )
        for v in u.passengers:
            tags.append(v.tag)
            units.append(
                [
                    v.unit_type,
                    u.alliance,  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4
                    0,
                    0,
                    v.health_max,
                    v.shield_max,
                    v.energy_max,
                    0,  # Visible = 1, Snapshot = 2, Hidden = 3
                    u.owner,  # 1-15, 16 = neutral
                    u.pos.x,
                    u.pos.y,
                    0,  # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
                    0,
                    0,
                    0,
                    0,
                    # Not populated for enemies or neutral
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    v.health,
                    v.shield,
                    v.energy,
                ]
            )
    units = units[:MAX_ENTITY_NUM]
    tags = tags[:MAX_ENTITY_NUM]
    raw_entity_info = named_array.NamedNumpyArray(units, [None, FeatureUnit], dtype=np.float32)

    for k, dtype in ENTITY_INFO:
        if 'last' in k:
            pass
        elif k == 'unit_type':
            entity_info[k] = UNIT_TYPES_REORDER_ARRAY[raw_entity_info[:, 'unit_type']].short()
        elif 'order_id' in k:
            order_idx = int(k.split('_')[-1])
            if order_idx == 0:
                entity_info[k] = UNIT_ABILITY_REORDER[raw_entity_info[:, k]].short()
                invalid_actions = entity_info[k] == -1
                if invalid_actions.any():
                    print('[ERROR] invalid unit ability', raw_entity_info[invalid_actions, k])
            else:
                entity_info[k] = ABILITY_TO_QUEUE_ACTION[raw_entity_info[:, k]].short()
                invalid_actions = entity_info[k] == -1
                if invalid_actions.any():
                    print('[ERROR] invalid queue ability', raw_entity_info[invalid_actions, k])
        elif 'buff_id' in k:
            entity_info[k] = BUFFS_REORDER_ARRAY[raw_entity_info[:, k]].short()
        elif k == 'addon_unit_type':
            entity_info[k] = ADDON_REORDER_ARRAY[raw_entity_info[:, k]].short()
        elif k == 'cargo_space_taken':
            entity_info[k] = torch.as_tensor(raw_entity_info[:, 'cargo_space_taken'], dtype=dtype).clamp_(min=0, max=8)
        elif k == 'cargo_space_max':
            entity_info[k] = torch.as_tensor(raw_entity_info[:, 'cargo_space_max'], dtype=dtype).clamp_(min=0, max=8)
        elif k == 'health_ratio':
            entity_info[k] = torch.as_tensor(
                raw_entity_info[:, 'health'], dtype=dtype
            ) / (torch.as_tensor(raw_entity_info[:, 'health_max'], dtype=dtype) + 1e-6)
        elif k == 'shield_ratio':
            entity_info[k] = torch.as_tensor(
                raw_entity_info[:, 'shield'], dtype=dtype
            ) / (torch.as_tensor(raw_entity_info[:, 'shield_max'], dtype=dtype) + 1e-6)
        elif k == 'energy_ratio':
            entity_info[k] = torch.as_tensor(
                raw_entity_info[:, 'energy'], dtype=dtype
            ) / (torch.as_tensor(raw_entity_info[:, 'energy_max'], dtype=dtype) + 1e-6)
        elif k == 'mineral_contents':
            entity_info[k] = torch.as_tensor(raw_entity_info[:, 'mineral_contents'], dtype=dtype) / 1800
        elif k == 'vespene_contents':
            entity_info[k] = torch.as_tensor(raw_entity_info[:, 'vespene_contents'], dtype=dtype) / 2500
        elif k == 'y':
            entity_info[k] = torch.as_tensor(map_size.y - raw_entity_info[:, 'y'], dtype=dtype)
        else:
            entity_info[k] = torch.as_tensor(raw_entity_info[:, k], dtype=dtype)

    # scalar info
    scalar_info['time'] = torch.tensor(obs.observation.game_loop, dtype=torch.float)
    player = obs.observation.player_common
    scalar_info['agent_statistics'] = torch.tensor(
        [
            player.minerals, player.vespene, player.food_used, player.food_cap, player.food_army, player.food_workers,
            player.idle_worker_count, player.army_count, player.warp_gate_count, player.larva_count
        ],
        dtype=torch.float
    )
    scalar_info['agent_statistics'] = torch.log(scalar_info['agent_statistics'] + 1)

    scalar_info["home_race"] = torch.tensor(requested_races[player.player_id], dtype=torch.uint8)
    for player_id, race in requested_races.items():
        if player_id != player.player_id:
            scalar_info["away_race"] = torch.tensor(race, dtype=torch.uint8)

    upgrades = torch.zeros(NUM_UPGRADES, dtype=torch.uint8)
    raw_upgrades = UPGRADES_REORDER_ARRAY[raw.player.upgrade_ids[:UPGRADE_LENGTH]]
    upgrades.scatter_(dim=0, index=raw_upgrades, value=1.)
    scalar_info["upgrades"] = upgrades

    unit_counts_bow = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
    scalar_info['unit_type_bool'] = torch.zeros(NUM_UNIT_TYPES, dtype=uint8)
    own_unit_types = entity_info['unit_type'][entity_info['alliance'] == 1]
    scalar_info['unit_counts_bow'] = torch.scatter_add(
        unit_counts_bow, dim=0, index=own_unit_types.long(), src=torch.ones_like(own_unit_types, dtype=torch.uint8)
    )
    scalar_info['unit_type_bool'] = (scalar_info['unit_counts_bow'] > 0).to(uint8)

    scalar_info['unit_order_type'] = torch.zeros(NUM_UNIT_MIX_ABILITIES, dtype=uint8)
    own_unit_orders = entity_info['order_id_0'][entity_info['alliance'] == 1]
    scalar_info['unit_order_type'].scatter_(0, own_unit_orders.long(), torch.ones_like(own_unit_orders, dtype=uint8))

    enemy_unit_types = entity_info['unit_type'][entity_info['alliance'] == 4]
    enemy_unit_type_bool = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
    scalar_info['enemy_unit_type_bool'] = torch.scatter(
        enemy_unit_type_bool,
        dim=0,
        index=enemy_unit_types.long(),
        src=torch.ones_like(enemy_unit_types, dtype=torch.uint8)
    )

    # game info
    game_info['action_result'] = [o.result for o in obs.action_errors]
    game_info['game_loop'] = obs.observation.game_loop
    game_info['tags'] = tags
    game_info['battle_score'] = compute_battle_score(obs)
    game_info['opponent_battle_score'] = 0.
    ret = {
        'spatial_info': spatial_info,
        'scalar_info': scalar_info,
        'entity_num': torch.tensor(len(entity_info['unit_type']), dtype=torch.long),
        'entity_info': entity_info,
        'game_info': game_info,
    }

    # value feature
    if opponent_obs:
        raw = opponent_obs.observation.raw_data
        enemy_unit_counts_bow = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
        enemy_x = []
        enemy_y = []
        enemy_unit_type = []
        unit_alliance = []
        for u in raw.units:
            if u.alliance == 1:
                enemy_x.append(u.pos.x)
                enemy_y.append(u.pos.y)
                enemy_unit_type.append(u.unit_type)
                unit_alliance.append(1)
        enemy_unit_type = UNIT_TYPES_REORDER_ARRAY[enemy_unit_type].short()
        enemy_unit_counts_bow = torch.scatter_add(
            enemy_unit_counts_bow,
            dim=0,
            index=enemy_unit_type.long(),
            src=torch.ones_like(enemy_unit_type, dtype=torch.uint8)
        )
        enemy_unit_type_bool = (enemy_unit_counts_bow > 0).to(uint8)

        unit_type = torch.cat([enemy_unit_type, own_unit_types], dim=0)
        enemy_x = torch.as_tensor(enemy_x, dtype=uint8)
        unit_x = torch.cat([enemy_x, entity_info['x'][entity_info['alliance'] == 1]], dim=0)

        enemy_y = torch.as_tensor(enemy_y, dtype=float32)
        enemy_y = torch.as_tensor(map_size.y - enemy_y, dtype=uint8)
        unit_y = torch.cat([enemy_y, entity_info['y'][entity_info['alliance'] == 1]], dim=0)
        total_unit_count = len(unit_y)
        unit_alliance += [0] * (total_unit_count - len(unit_alliance))
        unit_alliance = torch.as_tensor(unit_alliance, dtype=torch.bool)

        padding_num = MAX_ENTITY_NUM - total_unit_count
        if padding_num > 0:
            unit_x = torch.nn.functional.pad(unit_x, (0, padding_num), 'constant', 0)
            unit_y = torch.nn.functional.pad(unit_y, (0, padding_num), 'constant', 0)
            unit_type = torch.nn.functional.pad(unit_type, (0, padding_num), 'constant', 0)
            unit_alliance = torch.nn.functional.pad(unit_alliance, (0, padding_num), 'constant', 0)
        else:
            unit_x = unit_x[:MAX_ENTITY_NUM]
            unit_y = unit_y[:MAX_ENTITY_NUM]
            unit_type = unit_type[:MAX_ENTITY_NUM]
            unit_alliance = unit_alliance[:MAX_ENTITY_NUM]

        total_unit_count = torch.tensor(total_unit_count, dtype=torch.long)

        player = opponent_obs.observation.player_common
        enemy_agent_statistics = torch.tensor(
            [
                player.minerals, player.vespene, player.food_used, player.food_cap, player.food_army,
                player.food_workers, player.idle_worker_count, player.army_count, player.warp_gate_count,
                player.larva_count
            ],
            dtype=torch.float
        )
        enemy_agent_statistics = torch.log(enemy_agent_statistics + 1)
        enemy_raw_upgrades = UPGRADES_REORDER_ARRAY[raw.player.upgrade_ids[:UPGRADE_LENGTH]]
        enemy_upgrades = torch.zeros(NUM_UPGRADES, dtype=torch.uint8)
        enemy_upgrades.scatter_(dim=0, index=enemy_raw_upgrades, value=1.)

        d = MINIMAP_FEATURES.player_relative.unpack(opponent_obs.observation).copy()
        d = torch.from_numpy(d)
        padding_y = DEFAULT_SPATIAL_SIZE[0] - d.shape[0]
        padding_x = DEFAULT_SPATIAL_SIZE[1] - d.shape[1]
        if (padding_y != 0 or padding_x != 0) and padding_spatial:
            d = torch.nn.functional.pad(d, (0, padding_x, 0, padding_y), 'constant', 0)
        enemy_units_spatial = d == 1
        own_units_spatial = ret['spatial_info']['player_relative'] == 1
        value_feature = {
            'unit_type': unit_type,
            'enemy_unit_counts_bow': enemy_unit_counts_bow,
            'enemy_unit_type_bool': enemy_unit_type_bool,
            'unit_x': unit_x,
            'unit_y': unit_y,
            'unit_alliance': unit_alliance,
            'total_unit_count': total_unit_count,
            'enemy_agent_statistics': enemy_agent_statistics,
            'enemy_upgrades': enemy_upgrades,
            'own_units_spatial': own_units_spatial.unsqueeze(dim=0),
            'enemy_units_spatial': enemy_units_spatial.unsqueeze(dim=0)
        }
        ret['value_feature'] = value_feature
        game_info['opponent_battle_score'] = compute_battle_score(opponent_obs)
    return ret
