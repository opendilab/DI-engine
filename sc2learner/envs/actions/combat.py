from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import ABILITY_ID as ABILITY
from pysc2.lib.typeenums import UPGRADE_ID as UPGRADE

from sc2learner.envs.actions.function import Function
import sc2learner.envs.common.utils as utils
from sc2learner.envs.common.const import ATTACK_FORCE
from sc2learner.envs.common.const import ALLY_TYPE
from sc2learner.envs.common.const import PRIORITIZED_ATTACK

Region = namedtuple('Region', ('ranges', 'rally_point_a', 'rally_point_b'))


class CombatActions(object):
    def __init__(self):
        self._regions = [
            Region([(0, 0, 200, 176)], (161.5, 21.5), (38.5, 122.5)),
            Region([(0, 88, 80, 176)], (68, 108), (68, 108)),
            Region([(80, 88, 120, 176)], (100, 113.5), (100, 113.5)),
            Region([(120, 88, 200, 176)], (147.5, 113.5), (147.5, 113.5)),
            Region([(0, 55, 80, 88)], (36.5, 76.5), (36.5, 76.5)),
            Region([(80, 55, 120, 88)], (100, 71.5), (100, 71.5)),
            Region([(120, 55, 200, 88)], (163.5, 66.5), (163.5, 66.5)),
            Region([(0, 0, 80, 55)], (52.5, 30), (52.5, 30)),
            Region([(80, 0, 120, 55)], (100, 30), (100, 30)),
            Region([(120, 0, 200, 55)], (133, 36), (133, 36))
        ]
        self._flip_region = lambda r_id: 10 - r_id if r_id > 0 else r_id

        self._attack_tasks = {}

    def reset(self):
        self._attack_tasks.clear()

    def action(self, source_region_id, target_region_id):
        assert source_region_id < len(self._regions)
        assert target_region_id < len(self._regions)
        return Function(
            name=("combats_in_region_%d_attack_region_%d" % (source_region_id, target_region_id)),
            function=self._attack_region(source_region_id, target_region_id),
            is_valid=self._is_valid_attack_region(source_region_id, target_region_id)
        )

    @property
    def num_regions(self):
        return len(self._regions)

    @property
    def action_rally_new_combat_units(self):
        return Function(
            name="rally_new_combat_units",
            function=self._rally_new_combat_units,
            is_valid=self._is_valid_rally_new_combat_units
        )

    @property
    def action_framewise_rally_and_attack(self):
        return Function(
            name="framewise_rally_and_attack", function=self._framewise_rally_and_attack, is_valid=lambda dc: True
        )

    def _attack_region(self, source_region_id, target_region_id):
        def act(dc):
            flip = True if self._player_position(dc) == 0 else False
            src_id = self._flip_region(source_region_id) if flip else source_region_id
            tgt_id = self._flip_region(target_region_id) if flip else target_region_id
            combat_unit = [u for u in dc.combat_units if self._is_in_region(u, src_id)]
            self._set_attack_task(combat_unit, tgt_id)
            return []

        return act

    def _is_valid_attack_region(self, source_region_id, target_region_id):
        def is_valid(dc):
            flip = True if self._player_position(dc) == 0 else False
            src_id = self._flip_region(source_region_id) if flip else source_region_id
            combat_unit = [u for u in dc.combat_units if self._is_in_region(u, src_id)]
            return len(combat_unit) >= 3

        return is_valid

    def _rally_new_combat_units(self, dc):
        new_combat_units = [u for u in dc.combat_units if dc.is_new_unit(u)]
        if self._player_position(dc) == 0:
            self._set_attack_task(new_combat_units, 1)
        else:
            self._set_attack_task(new_combat_units, 9)
        return []

    def _is_valid_rally_new_combat_units(self, dc):
        new_combat_units = [u for u in dc.combat_units if dc.is_new_unit(u)]
        if len(new_combat_units) > 0:
            return True
        else:
            return False

    def _framewise_rally_and_attack(self, dc):
        actions = []
        for region_id in range(len(self._regions)):
            units_with_task = [
                u for u in dc.combat_units if (u.tag in self._attack_tasks and self._attack_tasks[u.tag] == region_id)
            ]
            if len(units_with_task) > 0:
                target_enemies = [
                    u for u in dc.units_of_alliance(ALLY_TYPE.ENEMY.value) if self._is_in_region(u, region_id)
                ]
                if len(target_enemies) > 0:
                    actions.extend(self._micro_attack(units_with_task, target_enemies, dc))
                else:
                    if self._player_position(dc) == 0:
                        rally_point = self._regions[region_id].rally_point_a
                    else:
                        rally_point = self._regions[region_id].rally_point_b
                    actions.extend(self._micro_rally(units_with_task, rally_point, dc))
        return actions

    def _micro_attack(self, combat_units, enemy_units, dc):
        def prioritized_attack(unit, target_units):
            assert len(target_units) > 0
            prioritized_target_units = [u for u in target_units if u.unit_type in PRIORITIZED_ATTACK]
            if len(prioritized_target_units) > 0:
                closest_target = utils.closest_unit(unit, prioritized_target_units)
            else:
                closest_target = utils.closest_unit(unit, target_units)
            target_pos = (closest_target.float_attr.pos_x, closest_target.float_attr.pos_y)
            return self._unit_attack(unit, target_pos, dc)

        def flee_or_fight(unit, target_units):
            assert len(target_units) > 0
            closest_target = utils.closest_unit(unit, target_units)
            closest_dist = utils.closest_distance(unit, enemy_units)
            strongest_health = utils.strongest_health(combat_units)
            if (closest_dist < 5.0 and unit.float_attr.health / unit.float_attr.health_max < 0.3
                    and strongest_health > 0.9):
                x = unit.float_attr.pos_x + (unit.float_attr.pos_x - closest_target.float_attr.pos_x) * 0.2
                y = unit.float_attr.pos_y + (unit.float_attr.pos_y - closest_target.float_attr.pos_y) * 0.2
                target_pos = (x, y)
                return self._unit_move(unit, target_pos, dc)
            else:
                target_pos = (closest_target.float_attr.pos_x, closest_target.float_attr.pos_y)
                return self._unit_attack(unit, target_pos, dc)

        air_combat_units = [
            u for u in combat_units
            if (ATTACK_FORCE[u.unit_type].can_attack_air and not ATTACK_FORCE[u.unit_type].can_attack_ground)
        ]
        ground_combat_units = [
            u for u in combat_units
            if (not ATTACK_FORCE[u.unit_type].can_attack_air and ATTACK_FORCE[u.unit_type].can_attack_ground)
        ]
        air_ground_combat_units = [
            u for u in combat_units
            if (ATTACK_FORCE[u.unit_type].can_attack_air and ATTACK_FORCE[u.unit_type].can_attack_ground)
        ]
        air_enemy_units = [u for u in enemy_units if u.bool_attr.is_flying]
        ground_enemy_units = [u for u in enemy_units if not u.bool_attr.is_flying]
        actions = []
        for unit in air_combat_units:
            if len(air_enemy_units) > 0:
                actions.extend(prioritized_attack(unit, air_enemy_units))
        for unit in ground_combat_units:
            if len(ground_enemy_units) > 0:
                actions.extend(prioritized_attack(unit, ground_enemy_units))
        for unit in air_ground_combat_units:
            if len(enemy_units) > 0:
                actions.extend(prioritized_attack(unit, enemy_units))
        return actions

    def _micro_rally(self, units, rally_point, dc):
        actions = []
        for unit in units:
            actions.extend(self._unit_attack(unit, rally_point, dc))
        return actions

    def _unit_attack(self, unit, target_pos, dc):
        # move with attack
        if unit.unit_type == UNIT_TYPE.ZERG_RAVAGER.value:
            return self._ravager_unit_attack(unit, target_pos, dc)
        # elif (unit.unit_type == UNIT_TYPE.ZERG_ROACH.value or
        # unit.unit_type == UNIT_TYPE.ZERG_ROACHBURROWED.value):
        # return self._roach_unit_attack(unit, target_pos, dc)
        elif (unit.unit_type == UNIT_TYPE.ZERG_LURKERMP.value
              or unit.unit_type == UNIT_TYPE.ZERG_LURKERMPBURROWED.value):
            return self._lurker_unit_attack(unit, target_pos, dc)
        else:
            return self._normal_unit_attack(unit, target_pos)

    def _unit_move(self, unit, target_pos, dc):
        # move without attack
        if unit.unit_type == UNIT_TYPE.ZERG_LURKERMPBURROWED.value:
            return self._lurker_unit_move(unit, target_pos)
        # elif unit.unit_type == UNIT_TYPE.ZERG_ROACHBURROWED.value:
        # return self._roach_unit_move(unit, target_pos, dc)
        else:
            return self._normal_unit_move(unit, target_pos)

    def _normal_unit_attack(self, unit, target_pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(unit.tag)
        action.action_raw.unit_command.ability_id = ABILITY.ATTACK_ATTACK.value
        action.action_raw.unit_command.target_world_space_pos.x = target_pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = target_pos[1]
        return [action]

    def _normal_unit_move(self, unit, target_pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.unit_tags.append(unit.tag)
        action.action_raw.unit_command.ability_id = ABILITY.MOVE.value
        action.action_raw.unit_command.target_world_space_pos.x = target_pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = target_pos[1]
        return [action]

    def _roach_unit_attack(self, unit, target_pos, dc):
        actions = []
        ground_enemies = [u for u in dc.units_of_alliance(ALLY_TYPE.ENEMY.value) if not u.bool_attr.is_flying]
        if len(utils.units_nearby(unit, ground_enemies, max_distance=4)) > 0:
            if unit.unit_type == UNIT_TYPE.ZERG_ROACHBURROWED.value:
                action = sc_pb.Action()
                action.action_raw.unit_command.unit_tags.append(unit.tag)
                action.action_raw.unit_command.ability_id = ABILITY.BURROWUP_ROACH.value
                actions.append(action)
            actions.extend(self._normal_unit_attack(unit, target_pos))
        else:
            actions.extend(self._roach_unit_move(unit, target_pos, dc))
        return actions

    def _roach_unit_move(self, unit, target_pos, dc):
        actions = []
        if (UPGRADE.TUNNELINGCLAWS.value in dc.upgraded_techs and UPGRADE.BURROW.value in dc.upgraded_techs
                and unit.unit_type == UNIT_TYPE.ZERG_ROACH.value):
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(unit.tag)
            action.action_raw.unit_command.ability_id = ABILITY.BURROWDOWN_ROACH.value
            actions.append(action)
        actions.extend(self._normal_unit_move(unit, target_pos))
        return actions

    def _lurker_unit_attack(self, unit, target_pos, dc):
        actions = []
        ground_enemies = [u for u in dc.units_of_alliance(ALLY_TYPE.ENEMY.value) if not u.bool_attr.is_flying]
        if len(utils.units_nearby(unit, ground_enemies, max_distance=8)) > 0:
            if unit.unit_type == UNIT_TYPE.ZERG_LURKERMP.value:
                action = sc_pb.Action()
                action.action_raw.unit_command.unit_tags.append(unit.tag)
                action.action_raw.unit_command.ability_id = \
                    ABILITY.BURROWDOWN_LURKER.value
                actions.append(action)
        else:
            actions.extend(self._lurker_unit_move(unit, target_pos))
        return actions

    def _lurker_unit_move(self, unit, target_pos):
        actions = []
        if unit.unit_type == UNIT_TYPE.ZERG_LURKERMPBURROWED.value:
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(unit.tag)
            action.action_raw.unit_command.ability_id = ABILITY.BURROWUP_LURKER.value
            actions.append(action)
        actions.extend(self._normal_unit_move(unit, target_pos))
        return actions

    def _ravager_unit_attack(self, unit, target_pos, dc):
        actions = []
        ground_units = [u for u in dc.units_of_alliance(ALLY_TYPE.SELF.value) if not u.bool_attr.is_flying]
        if len(utils.units_nearby(target_pos, ground_units, max_distance=2)) == 0:
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(unit.tag)
            action.action_raw.unit_command.ability_id = \
                ABILITY.EFFECT_CORROSIVEBILE.value
            action.action_raw.unit_command.target_world_space_pos.x = target_pos[0]
            action.action_raw.unit_command.target_world_space_pos.y = target_pos[1]
            actions.append(action)
        actions.extend(self._normal_unit_attack(unit, target_pos))
        return actions

    def _set_attack_task(self, units, target_region_id):
        for u in units:
            self._attack_tasks[u.tag] = target_region_id

    def _is_in_region(self, unit, region_id):
        return any(
            [
                (
                    unit.float_attr.pos_x >= r[0] and unit.float_attr.pos_x < r[2] and unit.float_attr.pos_y >= r[1]
                    and unit.float_attr.pos_y < r[3]
                ) for r in self._regions[region_id].ranges
            ]
        )

    def _player_position(self, dc):
        if dc.init_base_pos[0] < 100:
            return 0
        else:
            return 1
