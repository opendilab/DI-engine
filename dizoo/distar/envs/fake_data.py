from typing import Sequence
import math
import six
import numpy as np
import torch

from pysc2.lib import features, actions, point, units
from pysc2.maps.melee import Melee
from s2clientprotocol import raw_pb2
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import score_pb2, common_pb2

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
    cum_action_mask= torch.tensor(1.0,dtype=torch.float)

    return {
        'actions_mask': mask,
        'selected_units_logits_mask': selected_units_logits_mask,
        'target_units_logits_mask': target_units_logits_mask,
        'cum_action_mask': cum_action_mask
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
        'entity_num': torch.randint(5, 100, size=(), dtype=torch.long),
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


def fake_rl_traj_with_last(unroll_len=4):
    list_step_data = []
    for i in range(unroll_len):
        step_data = rl_step_data()
        list_step_data.append(step_data)
    list_step_data.append(rl_step_data(last=True))  # last step
    return list_step_data


def get_fake_rl_batch(batch_size=3, unroll_len=4):
    data_batch = [fake_rl_traj_with_last(unroll_len) for _ in range(batch_size)]
    return data_batch


class Unit(object):
    """Class to hold unit data for the builder."""

    def __init__(
        self,
        unit_type,  # see lib/units.py
        player_relative,  # features.PlayerRelative,
        health,
        shields=0,
        energy=0,
        transport_slots_taken=0,
        build_progress=1.0
    ):

        self.unit_type = unit_type
        self.player_relative = player_relative
        self.health = health
        self.shields = shields
        self.energy = energy
        self.transport_slots_taken = transport_slots_taken
        self.build_progress = build_progress

    def fill(self, unit_proto):
        """Fill a proto unit data object from this Unit."""
        unit_proto.unit_type = self.unit_type
        unit_proto.player_relative = self.player_relative
        unit_proto.health = self.health
        unit_proto.shields = self.shields
        unit_proto.energy = self.energy
        unit_proto.transport_slots_taken = self.transport_slots_taken
        unit_proto.build_progress = self.build_progress

    def as_array(self):
        """Return the unit represented as a numpy array."""
        return np.array(
            [
                self.unit_type, self.player_relative, self.health, self.shields, self.energy,
                self.transport_slots_taken,
                int(self.build_progress * 100)
            ],
            dtype=np.int32
        )

    def as_dict(self):
        return vars(self)


class FeatureUnit(object):
    """Class to hold feature unit data for the builder."""

    def __init__(
        self,
        unit_type,  # see lib/units
        alliance,  # features.PlayerRelative,
        owner,  # 1-15, 16=neutral
        pos,  # common_pb2.Point,
        radius,
        health,
        health_max,
        is_on_screen,
        shield=0,
        shield_max=0,
        energy=0,
        energy_max=0,
        cargo_space_taken=0,
        cargo_space_max=0,
        build_progress=1.0,
        facing=0.0,
        display_type=raw_pb2.Visible,  # raw_pb.DisplayType
        cloak=raw_pb2.NotCloaked,  # raw_pb.CloakState
        is_selected=False,
        is_blip=False,
        is_powered=True,
        mineral_contents=0,
        vespene_contents=0,
        assigned_harvesters=0,
        ideal_harvesters=0,
        weapon_cooldown=0.0,
        orders=None,
        is_flying=False,
        is_burrowed=False,
        is_hallucination=False,
        is_active=False,
        attack_upgrade_level=0,
        armor_upgrade_level=0,
        shield_upgrade_level=0,
    ):

        self.unit_type = unit_type
        self.alliance = alliance
        self.owner = owner
        self.pos = pos
        self.radius = radius
        self.health = health
        self.health_max = health_max
        self.is_on_screen = is_on_screen
        self.shield = shield
        self.shield_max = shield_max
        self.energy = energy
        self.energy_max = energy_max
        self.cargo_space_taken = cargo_space_taken
        self.cargo_space_max = cargo_space_max
        self.build_progress = build_progress
        self.facing = facing
        self.display_type = display_type
        self.cloak = cloak
        self.is_selected = is_selected
        self.is_blip = is_blip
        self.is_powered = is_powered
        self.mineral_contents = mineral_contents
        self.vespene_contents = vespene_contents
        self.assigned_harvesters = assigned_harvesters
        self.ideal_harvesters = ideal_harvesters
        self.weapon_cooldown = weapon_cooldown
        self.is_flying = is_flying
        self.is_burrowed = is_burrowed
        self.is_hallucination = is_hallucination
        self.is_active = is_active
        self.attack_upgrade_level = attack_upgrade_level
        self.armor_upgrade_level = armor_upgrade_level
        self.shield_upgrade_level = shield_upgrade_level
        if orders is not None:
            self.orders = orders

    def as_dict(self):
        return vars(self)


class FeatureResource(object):
    """Class to hold feature unit data for the builder."""

    def __init__(
        self,
        unit_type,  # see lib/units
        alliance,  # features.PlayerRelative,
        owner,  # 1-15, 16=neutral
        pos,  # common_pb2.Point,
        radius,
        is_on_screen,
        build_progress=1.0,
        facing=0.0,
        display_type=raw_pb2.Visible,  # raw_pb.DisplayType
        cloak=raw_pb2.NotCloaked,  # raw_pb.CloakState
        is_blip=False,
        is_powered=True,
    ):

        self.unit_type = unit_type
        self.alliance = alliance
        self.owner = owner
        self.pos = pos
        self.radius = radius
        self.is_on_screen = is_on_screen
        self.build_progress = build_progress
        self.facing = facing
        self.display_type = display_type
        self.cloak = cloak
        self.is_blip = is_blip
        self.is_powered = is_powered

    def as_dict(self):
        return vars(self)


class Builder(object):
    """For test code - build a dummy ResponseObservation proto."""

    def __init__(self, obs_spec):
        self._game_loop = 1
        self._player_common = sc_pb.PlayerCommon(
            player_id=1,
            minerals=20,
            vespene=50,
            food_cap=36,
            food_used=21,
            food_army=6,
            food_workers=15,
            idle_worker_count=2,
            army_count=6,
            warp_gate_count=0,
        )

        self._score = 300
        self._score_type = Melee
        self._score_details = score_pb2.ScoreDetails(
            idle_production_time=0,
            idle_worker_time=0,
            total_value_units=190,
            total_value_structures=230,
            killed_value_units=0,
            killed_value_structures=0,
            collected_minerals=2130,
            collected_vespene=560,
            collection_rate_minerals=50,
            collection_rate_vespene=20,
            spent_minerals=2000,
            spent_vespene=500,
            food_used=score_pb2.CategoryScoreDetails(
                none=0.0, army=0.0, economy=self._player_common.food_used, technology=0.0, upgrade=0.0
            ),
            killed_minerals=score_pb2.CategoryScoreDetails(
                none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0
            ),
            killed_vespene=score_pb2.CategoryScoreDetails(none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0),
            lost_minerals=score_pb2.CategoryScoreDetails(none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0),
            lost_vespene=score_pb2.CategoryScoreDetails(none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0),
            friendly_fire_minerals=score_pb2.CategoryScoreDetails(
                none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0
            ),
            friendly_fire_vespene=score_pb2.CategoryScoreDetails(
                none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0
            ),
            used_minerals=score_pb2.CategoryScoreDetails(none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0),
            used_vespene=score_pb2.CategoryScoreDetails(none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0),
            total_used_minerals=score_pb2.CategoryScoreDetails(
                none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0
            ),
            total_used_vespene=score_pb2.CategoryScoreDetails(
                none=0.0, army=0.0, economy=0.0, technology=0.0, upgrade=0.0
            ),
            total_damage_dealt=score_pb2.VitalScoreDetails(life=0.0, shields=0.0, energy=0.0),
            total_damage_taken=score_pb2.VitalScoreDetails(life=0.0, shields=0.0, energy=0.0),
            total_healed=score_pb2.VitalScoreDetails(life=0.0, shields=0.0, energy=0.0),
        )

        self._obs_spec = obs_spec
        self._single_select = None
        self._multi_select = None
        self._build_queue = None
        self._production = None
        self._feature_units = None

    def game_loop(self, game_loop):
        self._game_loop = game_loop
        return self

    # pylint:disable=unused-argument
    def player_common(
        self,
        player_id=None,
        minerals=None,
        vespene=None,
        food_cap=None,
        food_used=None,
        food_army=None,
        food_workers=None,
        idle_worker_count=None,
        army_count=None,
        warp_gate_count=None,
        larva_count=None
    ):
        """Update some or all of the fields in the PlayerCommon data."""

        args = dict(locals())
        for key, value in six.iteritems(args):
            if value is not None and key != 'self':
                setattr(self._player_common, key, value)
        return self

    def score(self, score):
        self._score = score
        return self

    def score_details(
        self,
        idle_production_time=None,
        idle_worker_time=None,
        total_value_units=None,
        total_value_structures=None,
        killed_value_units=None,
        killed_value_structures=None,
        collected_minerals=None,
        collected_vespene=None,
        collection_rate_minerals=None,
        collection_rate_vespene=None,
        spent_minerals=None,
        spent_vespene=None
    ):
        """Update some or all of the fields in the ScoreDetails data."""

        args = dict(locals())
        for key, value in six.iteritems(args):
            if value is not None and key != 'self':
                setattr(self._score_details, key, value)
        return self

    # pylint:enable=unused-argument

    def score_by_category(self, entry_name, none, army, economy, technology, upgrade):

        field = getattr(self._score_details, entry_name)
        field.CopyFrom(
            score_pb2.CategoryScoreDetails(
                none=none, army=army, economy=economy, technology=technology, upgrade=upgrade
            )
        )

    def score_by_vital(self, entry_name, life, shields, energy):
        field = getattr(self._score_details, entry_name)
        field.CopyFrom(score_pb2.VitalScoreDetails(life=life, shields=shields, energy=energy))

    def single_select(self, unit):
        self._single_select = unit
        return self

    def multi_select(self, units):
        self._multi_select = units
        return self

    def build_queue(self, build_queue, production=None):
        self._build_queue = build_queue
        self._production = production
        return self

    def feature_units(self, feature_units):
        self._feature_units = feature_units
        return self

    def build(self):
        """Builds and returns a proto ResponseObservation."""
        response_observation = sc_pb.ResponseObservation()
        obs = response_observation.observation

        obs.game_loop = self._game_loop
        obs.player_common.CopyFrom(self._player_common)

        obs.score.score_type = 2
        obs.score.score = self._score
        obs.score.score_details.CopyFrom(self._score_details)

        def fill(image_data, size, bits):
            image_data.bits_per_pixel = bits
            image_data.size.y = size[0]
            image_data.size.x = size[1]
            image_data.data = b'\0' * int(math.ceil(size[0] * size[1] * bits / 8))

        if 'feature_screen' in self._obs_spec:
            for feature in features.SCREEN_FEATURES:
                fill(getattr(obs.feature_layer_data.renders, feature.name), self._obs_spec['feature_screen'][1:], 8)

        if 'feature_minimap' in self._obs_spec:
            for feature in features.MINIMAP_FEATURES:
                fill(
                    getattr(obs.feature_layer_data.minimap_renders, feature.name),
                    self._obs_spec['feature_minimap'][1:], 8
                )

        # if 'rgb_screen' in self._obs_spec:
        #   fill(obs.render_data.map, self._obs_spec['rgb_screen'][:2], 24)

        # if 'rgb_minimap' in self._obs_spec:
        #   fill(obs.render_data.minimap, self._obs_spec['rgb_minimap'][:2], 24)

        if self._single_select:
            self._single_select.fill(obs.ui_data.single.unit)

        if self._multi_select:
            for unit in self._multi_select:
                obs.ui_data.multi.units.add(**unit.as_dict())

        if self._build_queue:
            for unit in self._build_queue:
                obs.ui_data.production.build_queue.add(**unit.as_dict())

        if self._production:
            for item in self._production:
                obs.ui_data.production.production_queue.add(**item)

        if self._feature_units:
            for tag, feature_unit in enumerate(self._feature_units, 1):
                args = dict(tag=tag)
                args.update(feature_unit.as_dict())
                obs.raw_data.units.add(**args)

        return response_observation


def fake_raw_obs():
    _features = features.Features(
        features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=(1, 1), minimap=(W, H)),
            rgb_dimensions=features.Dimensions(screen=(128, 124), minimap=(1, 1)),
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True
        ),
        map_size=point.Point(256, 256)
    )
    obs_spec = _features.observation_spec()
    builder = Builder(obs_spec).game_loop(0)
    feature_units = [
        FeatureResource(
            units.Neutral.MineralField,
            features.PlayerRelative.NEUTRAL,
            owner=16,
            pos=common_pb2.Point(x=10, y=10, z=0),
            facing=0.0,
            radius=1.125,
            build_progress=1.0,
            cloak=raw_pb2.CloakedUnknown,
            is_on_screen=True,
            is_blip=False,
            is_powered=False,
        ),
        FeatureResource(
            units.Neutral.MineralField,
            features.PlayerRelative.NEUTRAL,
            owner=16,
            pos=common_pb2.Point(x=120, y=10, z=0),
            facing=0.0,
            radius=1.125,
            build_progress=1.0,
            cloak=raw_pb2.CloakedUnknown,
            is_on_screen=False,
            is_blip=False,
            is_powered=False,
        ),
        FeatureResource(
            units.Neutral.MineralField,
            features.PlayerRelative.NEUTRAL,
            owner=16,
            pos=common_pb2.Point(x=10, y=120, z=0),
            facing=0.0,
            radius=1.125,
            build_progress=1.0,
            cloak=raw_pb2.CloakedUnknown,
            is_on_screen=False,
            is_blip=False,
            is_powered=False,
        ),
        FeatureResource(
            units.Neutral.MineralField,
            features.PlayerRelative.NEUTRAL,
            owner=16,
            pos=common_pb2.Point(x=120, y=120, z=0),
            facing=0.0,
            radius=1.125,
            build_progress=1.0,
            cloak=raw_pb2.CloakedUnknown,
            is_on_screen=False,
            is_blip=False,
            is_powered=False,
        ),
        FeatureUnit(
            units.Zerg.Drone,
            features.PlayerRelative.SELF,
            owner=1,
            display_type=raw_pb2.Visible,
            pos=common_pb2.Point(x=10, y=11, z=0),
            radius=0.375,
            facing=3,
            cloak=raw_pb2.NotCloaked,
            is_selected=False,
            is_on_screen=True,
            is_blip=False,
            health_max=40,
            health=40,
            is_flying=False,
            is_burrowed=False
        ),
    ]

    builder.feature_units(feature_units)
    return builder.build()


def get_fake_env_step_data():
    return {'raw_obs': fake_raw_obs(), 'opponent_obs': None, 'action_result': [0]}


def get_fake_env_reset_data():
    import os
    import pickle
    with open(os.path.join(os.path.dirname(__file__), 'fake_reset.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data
