from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

from sc2learner.envs.common.const import ALLY_TYPE
from sc2learner.envs.common.const import PLAYER_FEATURE
from sc2learner.envs.common.const import COMBAT_TYPES
import sc2learner.envs.common.utils as utils


class DataContext(object):

    def __init__(self):
        self._units = []
        self._player = None
        self._raw_data = None
        self._existed_tags = set()

    def update(self, observation):
        for u in self._units:
            self._existed_tags.add(u.tag)
        self._units = observation['units']
        self._player = observation['player']
        self._raw_data = observation['raw_data']
        self._combat_units = self.units_of_types(COMBAT_TYPES)

    def reset(self, observation):
        self._existed_tags.clear()
        self.update(observation)
        init_base = self.units_of_type(UNIT_TYPE.ZERG_HATCHERY.value)[0]
        self._init_base_pos = (init_base.float_attr.pos_x,
                               init_base.float_attr.pos_y)

    def units_of_alliance(self, ally):
        return [u for u in self._units if u.int_attr.alliance == ally]

    def units_of_type(self, type_id, ally=ALLY_TYPE.SELF.value):
        return [u for u in self.units_of_alliance(ally) if u.unit_type == type_id]

    def mature_units_of_type(self, type_id, ally=ALLY_TYPE.SELF.value):
        return [u for u in self.units_of_type(type_id, ally)
                if u.float_attr.build_progress >= 1.0]

    def idle_units_of_type(self, type_id, ally=ALLY_TYPE.SELF.value):
        return [u for u in self.mature_units_of_type(type_id, ally)
                if len(u.orders) == 0]

    def units_of_types(self, type_list, ally=ALLY_TYPE.SELF.value):
        type_set = set(type_list)
        return [u for u in self.units_of_alliance(ally) if u.unit_type in type_set]

    def mature_units_of_types(self, type_list, ally=ALLY_TYPE.SELF.value):
        return [u for u in self.units_of_types(type_list, ally)
                if u.float_attr.build_progress >= 1.0]

    def idle_units_of_types(self, type_list, ally=ALLY_TYPE.SELF.value):
        return [u for u in self.mature_units_of_types(type_list, ally)
                if len(u.orders) == 0]

    def units_with_task(self, ability_id, ally=ALLY_TYPE.SELF.value):
        return [u for u in self.units_of_alliance(ally)
                if ability_id in set([order.ability_id for order in u.orders])]

    def is_new_unit(self, unit):
        return unit.tag not in self._existed_tags

    @property
    def units(self):
        return self._units

    @property
    def combat_units(self):
        return self._combat_units

    @property
    def minerals(self):
        return [u for u in self._units
                if (u.unit_type == UNIT_TYPE.NEUTRAL_MINERALFIELD.value or
                    u.unit_type == UNIT_TYPE.NEUTRAL_MINERALFIELD750.value)]

    @property
    def unexploited_minerals(self):
        self_bases = self.units_of_types([UNIT_TYPE.ZERG_HATCHERY.value,
                                          UNIT_TYPE.ZERG_LAIR.value,
                                          UNIT_TYPE.ZERG_HIVE.value])
        enemy_bases = self.units_of_types([UNIT_TYPE.ZERG_HATCHERY.value,
                                           UNIT_TYPE.ZERG_LAIR.value,
                                           UNIT_TYPE.ZERG_HIVE.value],
                                          ALLY_TYPE.ENEMY.value)
        return [u for u in self.minerals
                if utils.closest_distance(u, self_bases + enemy_bases) > 15]

    @property
    def gas(self):
        return [u for u in self._units
                if u.unit_type == UNIT_TYPE.NEUTRAL_VESPENEGEYSER.value]

    @property
    def exploitable_gas(self):
        extractors = self.units_of_type(UNIT_TYPE.ZERG_EXTRACTOR.value) + \
            self.units_of_type(UNIT_TYPE.ZERG_EXTRACTOR.value, ALLY_TYPE.ENEMY)
        bases = self.mature_units_of_types([UNIT_TYPE.ZERG_HATCHERY.value,
                                            UNIT_TYPE.ZERG_LAIR.value,
                                            UNIT_TYPE.ZERG_HIVE.value])
        return [u for u in self.gas if (utils.closest_distance(u, bases) < 10 and
                                        utils.closest_distance(u, extractors) > 3)]

    @property
    def mineral_count(self):
        return self._player[PLAYER_FEATURE.MINERALS.value]

    @property
    def gas_count(self):
        return self._player[PLAYER_FEATURE.VESPENE.value]

    @property
    def supply_count(self):
        return self._player[PLAYER_FEATURE.FOOD_CAP.value] - \
            self._player[PLAYER_FEATURE.FOOD_USED.value]

    @property
    def upgraded_techs(self):
        return set(self._raw_data.player.upgrade_ids)

    @property
    def init_base_pos(self):
        return self._init_base_pos
