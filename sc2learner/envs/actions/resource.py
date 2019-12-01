from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import ABILITY_ID as ABILITY

from sc2learner.envs.actions.function import Function
import sc2learner.envs.common.utils as utils


class ResourceActions(object):

  @property
  def action_queens_inject_larva(self):
    return Function(name="queens_inject_larva",
                    function=self._all_idle_queens_inject_larva,
                    is_valid=self._is_valid_all_idle_queens_inject_larva)

  @property
  def action_idle_workers_gather_minerals(self):
    return Function(name="idle_workers_gather_minerals",
                    function=self._all_idle_workers_gather_minerals,
                    is_valid=self._is_valid_all_idle_workers_gather_minerals)

  @property
  def action_assign_workers_gather_gas(self):
    return Function(name="assign_workers_gather_gas",
                    function=self._assign_workers_gather_gas,
                    is_valid=self._is_valid_assign_workers_gather_gas)

  @property
  def action_assign_workers_gather_minerals(self):
    return Function(name="assign_workers_gather_minerals",
                    function=self._assign_workers_gather_minerals,
                    is_valid=self._is_valid_assign_workers_gather_minerals)

  def _all_idle_queens_inject_larva(self, dc):
    injectable_queens = [
        # TODO: -->idle_units_of_type
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_QUEEN.value)
        if u.float_attr.energy >= 25
    ]
    bases = dc.mature_units_of_types([UNIT_TYPE.ZERG_HATCHERY.value,
                                      UNIT_TYPE.ZERG_LAIR.value,
                                      UNIT_TYPE.ZERG_HIVE.value])
    actions = []
    for queen in injectable_queens:
      action = sc_pb.Action()
      action.action_raw.unit_command.unit_tags.append(queen.tag)
      action.action_raw.unit_command.ability_id = \
          ABILITY.EFFECT_INJECTLARVA.value
      base = utils.closest_unit(queen, bases)
      action.action_raw.unit_command.target_unit_tag = base.tag
      actions.append(action)
    return actions

  def _is_valid_all_idle_queens_inject_larva(self, dc):
    injectable_queens = [
        # TODO: -->idle_units_of_type
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_QUEEN.value)
        if u.float_attr.energy >= 25
    ]
    bases = dc.mature_units_of_types([UNIT_TYPE.ZERG_HATCHERY.value,
                                      UNIT_TYPE.ZERG_LAIR.value,
                                      UNIT_TYPE.ZERG_HIVE.value])
    if len(bases) > 0 and len(injectable_queens) > 0: return True
    else: return False

  def _all_idle_workers_gather_minerals(self, dc):
    idle_workers = dc.idle_units_of_type(UNIT_TYPE.ZERG_DRONE.value)
    actions = []
    for worker in idle_workers:
      mineral = utils.closest_unit(worker, dc.minerals)
      action = sc_pb.Action()
      action.action_raw.unit_command.unit_tags.append(worker.tag)
      action.action_raw.unit_command.ability_id = \
          ABILITY.HARVEST_GATHER_DRONE.value
      action.action_raw.unit_command.target_unit_tag = mineral.tag
      actions.append(action)
    return actions

  def _is_valid_all_idle_workers_gather_minerals(self, dc):
    if (len(dc.idle_units_of_type(UNIT_TYPE.ZERG_DRONE.value)) > 0 and
        len(dc.minerals) > 0):
      return True
    else:
      return False

  def _assign_workers_gather_gas(self, dc):
    idle_extractors =  [
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_EXTRACTOR.value)
        if u.int_attr.ideal_harvesters - u.int_attr.assigned_harvesters > 0
    ]
    if len(idle_extractors) == 0: return []
    extractor = random.choice(idle_extractors)
    num_workers_need = extractor.int_attr.ideal_harvesters - \
        extractor.int_attr.assigned_harvesters
    extractor_tags = set(u.tag for u in dc.units_of_type(
        UNIT_TYPE.ZERG_EXTRACTOR.value))
    workers = [
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_DRONE.value)
        if (len(u.orders) == 0 or
            (u.orders[0].ability_id == ABILITY.HARVEST_GATHER_DRONE.value and
             u.orders[0].target_tag not in extractor_tags))
    ]
    if len(workers) == 0: return []
    assigned_workers = utils.closest_units(extractor, workers, num_workers_need)
    action = sc_pb.Action()
    action.action_raw.unit_command.unit_tags.extend(
        [u.tag for u in assigned_workers])
    action.action_raw.unit_command.ability_id = \
        ABILITY.HARVEST_GATHER_DRONE.value
    action.action_raw.unit_command.target_unit_tag = extractor.tag
    return [action]

  def _is_valid_assign_workers_gather_gas(self, dc):
    idle_extractors =  [
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_EXTRACTOR.value)
        if u.int_attr.ideal_harvesters - u.int_attr.assigned_harvesters > 0
    ]
    extractor_tags = set(u.tag for u in dc.units_of_type(
        UNIT_TYPE.ZERG_EXTRACTOR.value))
    workers = [
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_DRONE.value)
        if (len(u.orders) == 0 or
            (u.orders[0].ability_id == ABILITY.HARVEST_GATHER_DRONE.value and
             u.orders[0].target_tag not in extractor_tags))
    ]
    if len(idle_extractors) > 0 and len(workers) > 0: return True
    else: return False

  def _assign_workers_gather_minerals(self, dc):
    extractor_tags = set(u.tag for u in dc.units_of_type(
        UNIT_TYPE.ZERG_EXTRACTOR.value))
    workers = [
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_DRONE.value)
        if (len(u.orders) == 0 or
            (u.orders[0].ability_id == ABILITY.HARVEST_GATHER_DRONE.value and
             u.orders[0].target_tag in extractor_tags))
    ]
    actions = []
    for worker in random.sample(workers, min(3, len(workers))):
      target_mineral = utils.closest_unit(worker, dc.minerals)
      action = sc_pb.Action()
      action.action_raw.unit_command.unit_tags.append(worker.tag)
      action.action_raw.unit_command.ability_id = \
          ABILITY.HARVEST_GATHER_DRONE.value
      action.action_raw.unit_command.target_unit_tag = target_mineral.tag
      actions.append(action)
    return actions

  def _is_valid_assign_workers_gather_minerals(self, dc):
    extractor_tags = set(u.tag for u in dc.units_of_type(
        UNIT_TYPE.ZERG_EXTRACTOR.value))
    workers = [
        u for u in dc.units_of_type(UNIT_TYPE.ZERG_DRONE.value)
        if (len(u.orders) == 0 or
            (u.orders[0].ability_id == ABILITY.HARVEST_GATHER_DRONE.value and
             u.orders[0].target_tag in extractor_tags))
    ]
    return len(workers) > 0
