from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.tech_tree import TechTree

from sc2learner.envs.actions.function import Function


class UpgradeActions(object):

  def __init__(self, game_version='4.1.2'):
    self._tech_tree = TechTree()
    self._tech_tree.update_version(game_version)

  def action(self, func_name, upgrade_id):
    return Function(name=func_name,
                    function=self._upgrade_unit(upgrade_id),
                    is_valid=self._is_valid_upgrade_unit(upgrade_id))

  def _upgrade_unit(self, upgrade_id):

    def act(dc):
      tech = self._tech_tree.getUpgradeData(upgrade_id)
      if len(dc.idle_units_of_types(tech.whatBuilds)) == 0: return []
      upgrader = random.choice(dc.idle_units_of_types(tech.whatBuilds))
      action = sc_pb.Action()
      action.action_raw.unit_command.unit_tags.append(upgrader.tag)
      action.action_raw.unit_command.ability_id = tech.buildAbility
      return [action]

    return act

  def _is_valid_upgrade_unit(self, upgrade_id):

    def is_valid(dc):
      tech = self._tech_tree.getUpgradeData(upgrade_id)
      has_required_units = any([len(dc.mature_units_of_type(u)) > 0
                                for u in tech.requiredUnits]) \
          if len(tech.requiredUnits) > 0 else True
      has_required_upgrades = all([t in dc.upgraded_techs
                                   for t in tech.requiredUpgrades])
      if (has_required_units and
          has_required_upgrades and
          upgrade_id not in dc.upgraded_techs and
          len(dc.units_with_task(tech.buildAbility)) == 0 and
          dc.mineral_count >= tech.mineralCost and
          dc.gas_count >= tech.gasCost and
          dc.supply_count >= tech.supplyCost and
          len(dc.idle_units_of_types(tech.whatBuilds)) > 0):
        return True
      else:
        return False

    return is_valid
