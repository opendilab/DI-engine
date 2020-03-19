from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.tech_tree import TechTree
from pysc2.lib.unit_controls import Unit
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import ABILITY_ID as ABILITY

from sc2learner.envs.actions.function import Function
from sc2learner.envs.actions.placer import Placer
import sc2learner.envs.common.utils as utils
from sc2learner.envs.common.const import MAXIMUM_NUM


class BuildActions(object):
    def __init__(self, game_version='4.1.2'):
        self._placer = Placer()
        self._tech_tree = TechTree()
        self._tech_tree.update_version(game_version)

    def action(self, func_name, type_id):
        return Function(name=func_name, function=self._build_unit(type_id), is_valid=self._is_valid_build_unit(type_id))

    def _build_unit(self, type_id):
        def act(dc):
            tech = self._tech_tree.getUnitData(type_id)
            pos = self._placer.get_building_position(type_id, dc)
            if pos is None:
                return []
            extractor_tags = set(u.tag for u in dc.units_of_type(UNIT_TYPE.ZERG_EXTRACTOR.value))
            builders = dc.units_of_types(tech.whatBuilds)
            prefered_builders = [
                u for u in builders if (
                    u.unit_type != UNIT_TYPE.ZERG_DRONE.value or len(u.orders) == 0 or (
                        u.orders[0].ability_id == ABILITY.HARVEST_GATHER_DRONE.value
                        and u.orders[0].target_tag not in extractor_tags
                    )
                )
            ]
            if len(prefered_builders) > 0:
                builder = utils.closest_unit(pos, prefered_builders)
            else:
                if len(builders) == 0:
                    return []
                builder = utils.closest_unit(pos, builders)
            action = sc_pb.Action()
            action.action_raw.unit_command.unit_tags.append(builder.tag)
            action.action_raw.unit_command.ability_id = tech.buildAbility
            if isinstance(pos, Unit):
                action.action_raw.unit_command.target_unit_tag = pos.tag
            else:
                action.action_raw.unit_command.target_world_space_pos.x = pos[0]
                action.action_raw.unit_command.target_world_space_pos.y = pos[1]
            return [action]

        return act

    def _is_valid_build_unit(self, type_id):
        def is_valid(dc):
            tech = self._tech_tree.getUnitData(type_id)
            has_required_units = any([len(dc.mature_units_of_type(u)) > 0
                                      for u in tech.requiredUnits]) \
                if len(tech.requiredUnits) > 0 else True
            has_required_upgrades = all([t in dc.upgraded_techs for t in tech.requiredUpgrades])
            current_num = len(dc.units_of_type(type_id)) + \
                len(dc.units_with_task(tech.buildAbility))
            overquota = current_num >= MAXIMUM_NUM[type_id] \
                if type_id in MAXIMUM_NUM else False

            if (has_required_units and has_required_upgrades and not overquota and dc.mineral_count >= tech.mineralCost
                    and dc.gas_count >= tech.gasCost and dc.supply_count >= tech.supplyCost
                    and len(dc.units_of_types(tech.whatBuilds)) > 0 and len(dc.units_with_task(tech.buildAbility)) == 0
                    and self._placer.can_build(type_id, dc)):
                return True
            else:
                return False

        return is_valid
